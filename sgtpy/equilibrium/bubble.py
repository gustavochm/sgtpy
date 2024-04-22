from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import gdem
from .equilibriumresult import EquilibriumResult


def bubble_sus(P_T, X, T_P, bubble_type, y_guess, eos,
               vl0, vv0, Xassl0, Xassv0,
               not_in_y=[]):

    if bubble_type == 'T':
        P = P_T
        temp_aux = T_P
    elif bubble_type == 'P':
        temp_aux = P_T
        P = T_P

    # Liquid fugacities
    lnphil, vl, Xassl = eos.logfugef_aux(X, temp_aux, P, 'L', vl0, Xassl0)

    tol = 1e-8
    error = 1
    itacc = 0
    niter = 0
    n = 5
    Y_calc = y_guess
    Y = y_guess

    # Vapour fugacities
    lnphiv, vv, Xassv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv0, Xassv0)

    while error > tol and itacc < 3:
        niter += 1

        lnK = lnphil-lnphiv
        K = np.exp(lnK)
        Y_calc_old = Y_calc
        Y_calc = X*K

        # modification for non-volatile components Ki -> 0
        Y_calc[not_in_y] = 0.

        if niter == (n-3):
            Y3 = Y_calc
        elif niter == (n-2):
            Y2 = Y_calc
        elif niter == (n-1):
            Y1 = Y_calc
        elif niter == n:
            niter = 0
            itacc += 1
            dacc = gdem(Y_calc, Y1, Y2, Y3)
            Y_calc += dacc
        error = np.linalg.norm(Y_calc-Y_calc_old)
        Y = Y_calc/Y_calc.sum()
        # Vapor fugacities
        lnphiv, vv, Xassv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv, Xassv)

    if bubble_type == 'T':
        f0 = Y_calc.sum() - 1
    elif bubble_type == 'P':
        f0 = np.log(Y_calc.sum())

    return f0, Y, lnK, vl, vv, Xassl, Xassv


def bubble_newton(inc, X, T_P, bubble_type, eos, in_y=[]):
    global vl, vv, Xassl, Xassv
    f = np.zeros_like(inc)
    lnK = inc[:-1]
    K = np.exp(lnK)

    if bubble_type == 'T':
        P = inc[-1]
        temp_aux = T_P
    elif bubble_type == 'P':
        T = inc[-1]
        temp_aux = eos.temperature_aux(T)
        P = T_P

    nc = eos.nc
    Y = np.zeros(nc)
    Y[in_y] = X[in_y] * K

    # Liquid fugacities
    lnphil, vl, Xassl = eos.logfugef_aux(X, temp_aux, P, 'L', vl, Xassl)
    # Vapour fugacities
    lnphiv, vv, Xassv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv, Xassv)

    f[:-1] = lnK + (lnphiv - lnphil)[in_y]
    f[-1] = (Y-X).sum()

    return f


def bubblePy(y_guess, P_guess, X, T, model, good_initial=False,
             v0=[None, None], Xass0=[None, None],
             not_in_y_list=[],
             full_output=False):
    """
    Bubble point (T, x) -> (P, y)

    Solves bubble point at given liquid composition and temperature. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    y_guess : array_like
        guess of vapour phase composition
    P_guess : float
        guess of equilibrium pressure [Pa].
    X : array_like
        liquid phase composition
    T : float
        absolute temperature of the liquid [K].
    model : object
        created from mixture and saftvrmie function
    good_initial: bool, optional
        if True skip succesive substitution and solves by Newton's Method.
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    Xass0 : list, optional
        if supplied X association non-bonded fraction sites initial value
    not_in_y_list : list, optional
        index of components not present in the vapour phase (default is empty)
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    Y : array_like, vector of vapour fraction moles
    P : float, equilibrium pressure [Pa]

    """
    nc = model.nc
    if len(y_guess) != nc or len(X) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    if np.any([i > (nc-1) for i in not_in_y_list]):
        raise Exception('Index of components not_in_y_list must be less than nc')

    not_in_y = np.zeros(nc, dtype=bool)
    not_in_y[not_in_y_list] = 1
    in_y = np.logical_not(not_in_y)

    global vl, vv, Xassl, Xassv
    vl0, vv0 = v0
    Xassl0, Xassv0 = Xass0

    temp_aux = model.temperature_aux(T)

    it = 0
    itmax = 10
    tol = 1e-8

    P = P_guess
    out = bubble_sus(P, X, temp_aux, 'T', y_guess, model, vl0, vv0,
                     Xassl0, Xassv0, not_in_y=not_in_y)
    f, Y, lnK, vl, vv, Xassl, Xassv = out
    error = np.abs(f)
    h = 1e-4

    method = 'quasi-newton + ASS'
    while error > tol and it <= itmax and not good_initial:
        it += 1
        out = bubble_sus(P+h, X, temp_aux, 'T', Y, model, vl, vv,
                         Xassl, Xassv, not_in_y=not_in_y)
        f1, Y1, lnK1, vl, vv, Xassl, Xassv = out
        out = bubble_sus(P, X, temp_aux, 'T', Y, model, vl, vv,
                         Xassl, Xassv, not_in_y=not_in_y)
        f, Y, lnK, vl, vv, Xassl, Xassv = out
        df = (f1-f)/h
        dP = f / df
        if dP > P:
            dP = 0.4 * P
        elif np.isnan(dP):
            dP = 0.0
            it = 1.*itmax
        P -= dP
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK[in_y], P])
        sol1 = root(bubble_newton, inc0, args=(X, temp_aux, 'T', model, in_y))
        if sol1.success:
            method = 'second-order'
            error = np.linalg.norm(sol1.fun)
            it += sol1.nfev

            sol = sol1.x
            lnK_newton = sol[:-1]
            K_newton = np.exp(lnK_newton)
            Y = np.zeros(nc)
            Y[in_y] = X[in_y] * K_newton
            Y /= np.sum(Y)
            P = sol[-1]

            rhol, Xassl = model.density_aux(X, temp_aux, P, 'L', rho0=1./vl,
                                            Xass0=Xassl)
            rhov, Xassv = model.density_aux(Y, temp_aux, P, 'V', rho0=1./vv,
                                            Xass0=Xassv)
            vl = 1./rhol
            vv = 1./rhov

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'Xassl': Xassl, 'state1': 'Liquid',
               'Y': Y, 'v2': vv, 'Xassv': Xassv, 'state2': 'Vapor',
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return Y, P


def bubbleTy(y_guess, T_guess, X, P, model, good_initial=False,
             v0=[None, None], Xass0=[None, None],
             not_in_y_list=[],
             full_output=False):
    """
    Bubble point (P, x) -> (T, y)

    Solves bubble point at given liquid composition and pressure. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    y_guess : array_like
        guess of vapour phase composition
    T_guess : float
        guess of equilibrium temperature of the liquid [K].
    X : array_like
        liquid phase composition
    P : float
        pressure of the liquid [Pa]
    model : object
        created from mixture and saftvrmie function
    good_initial: bool, optional
        if True skip succesive substitution and solves by Newton's Method.
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    Xass0 : list, optional
        if supplied X association non-bonded fraction sites initial value
    not_in_y_list : list, optional
        index of components not present in the vapour phase (default is empty)
    full_output: bool, optional
        wheter to outputs all calculation info


    Returns
    -------
    Y : array_like
        vector of vapour fraction moles
    T : float
        equilibrium temperature [K]
    """

    nc = model.nc
    if len(y_guess) != nc or len(X) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    if np.any([i > (nc-1) for i in not_in_y_list]):
        raise Exception('Index of components not_in_y_list must be less than nc')

    not_in_y = np.zeros(nc, dtype=bool)
    not_in_y[not_in_y_list] = 1
    in_y = np.logical_not(not_in_y)

    global vl, vv, Xassl, Xassv
    vl0, vv0 = v0
    Xassl0, Xassv0 = Xass0

    it = 0
    itmax = 10
    tol = 1e-8

    T = T_guess
    temp_aux = model.temperature_aux(T)
    out = bubble_sus(temp_aux, X, P, 'P', y_guess, model, vl0, vv0, Xassl0,
                     Xassv0, not_in_y=not_in_y)
    f, Y, lnK, vl, vv, Xassl, Xassv = out
    # f, Y, lnK, vl, vv = bubble_sus(T, X, P, 'P', y_guess, model, vl0, vv0)
    error = np.abs(f)
    h = 1e-4
    method = 'quasi-newton + ASS'
    while error > tol and it <= itmax and not good_initial:
        it += 1
        temp_aux = model.temperature_aux(T+h)
        out = bubble_sus(temp_aux, X, P, 'P', Y, model, vl, vv, 
                         Xassl, Xassv, not_in_y=not_in_y)
        f1, Y1, lnK1, vl, vv, Xassl, Xassv = out
        temp_aux = model.temperature_aux(T)
        out = bubble_sus(temp_aux, X, P, 'P', Y, model, vl, vv, 
                         Xassl, Xassv, not_in_y=not_in_y)
        f, Y, lnK, vl, vv, Xassl, Xassv = out
        df = (f1-f)/(h)
        T -= f/df
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK[in_y], T])
        sol1 = root(bubble_newton, inc0, args=(X, P, 'P', model, in_y))
        if sol1.success:
            method = 'second-order'
            error = np.linalg.norm(sol1.fun)
            it += sol1.nfev

            sol = sol1.x
            lnK_newton = sol[:-1]
            K_newton = np.exp(lnK_newton)
            Y = np.zeros(nc)
            Y[in_y] = X[in_y] * K_newton
            Y /= np.sum(Y)
            T = sol[-1]

            rhol, Xassl = model.density_aux(X, temp_aux, P, 'L', rho0=1./vl,
                                            Xass0=Xassl)
            rhov, Xassv = model.density_aux(Y, temp_aux, P, 'V', rho0=1./vv,
                                            Xass0=Xassv)
            vl = 1./rhol
            vv = 1./rhov

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'Xassl': Xassl, 'state1': 'Liquid',
               'Y': Y, 'v2': vv, 'Xassv': Xassv, 'state2': 'Vapor',
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return Y, T


__all__ = ['bubbleTy', 'bubblePy']