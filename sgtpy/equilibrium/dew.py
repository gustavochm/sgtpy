from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import gdem
from .equilibriumresult import EquilibriumResult


# VLE phi-phi
def dew_sus(P_T, Y, T_P, dew_type, x_guess, eos, vl0, vv0,
            Xassl0, Xassv0, not_in_x=[]):

    if dew_type == 'T':
        P = P_T
        temp_aux = T_P
    elif dew_type == 'P':
        T = P_T
        temp_aux = eos.temperature_aux(T)
        P = T_P

    # Vapour fugacities
    lnphiv, vv, Xassv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv0, Xassv0)

    tol = 1e-8
    error = 1
    itacc = 0
    niter = 0
    n = 5
    X_calc = x_guess
    X = x_guess

    # Liquid fugacities
    lnphil, vl, Xassl = eos.logfugef_aux(X, temp_aux, P, 'L', vl0, Xassl0)

    while error > tol and itacc < 3:
        niter += 1

        lnK = lnphil-lnphiv
        K = np.exp(lnK)
        X_calc_old = X_calc
        X_calc = Y/K

        # modification for non-condensable components Ki -> infty
        X_calc[not_in_x] = 0.

        if niter == (n-3):
            X3 = X_calc
        elif niter == (n-2):
            X2 = X_calc
        elif niter == (n-1):
            X1 = X_calc
        elif niter == n:
            niter = 0
            itacc += 1
            dacc = gdem(X_calc, X1, X2, X3)
            X_calc += dacc
        error = np.linalg.norm(X_calc - X_calc_old)
        X = X_calc/X_calc.sum()
        # Liquid fugacitiies
        lnphil, vl, Xassl = eos.logfugef_aux(X, temp_aux, P, 'L', vl, Xassl)

    if dew_type == 'T':
        f0 = X_calc.sum() - 1
    elif dew_type == 'P':
        f0 = np.log(X_calc.sum())

    return f0, X, lnK, vl, vv, Xassl, Xassv


def dew_newton(inc, Y, T_P, dew_type, eos, in_x=[]):

    global vl, vv, Xassl, Xassv

    f = np.zeros_like(inc)
    lnK = inc[:-1]
    K = np.exp(lnK)

    if dew_type == 'T':
        P = inc[-1]
        temp_aux = T_P
    elif dew_type == 'P':
        T = inc[-1]
        temp_aux = eos.temperature_aux(T)
        P = T_P

    nc = eos.nc
    X = np.zeros(nc)
    X[in_x] = Y[in_x] / K

    # Liquid fugacities
    lnphil, vl, Xassl = eos.logfugef_aux(X, temp_aux, P, 'L', vl, Xassl)
    # Vapor fugacities
    lnphiv, vv, Xassv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv, Xassv)

    f[:-1] = lnK + (lnphiv - lnphil)[in_x]
    f[-1] = (Y-X).sum()

    return f


def dewPx(x_guess, P_guess, y, T, model, good_initial=False,
          v0=[None, None], Xass0=[None, None],
          not_in_x_list=[],
          full_output=False):
    """
    Dew point (T, y) -> (P, y)

    Solves dew point at given vapour composition and temperature. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    x_guess : array_like
        guess of liquid phase composition
    P_guess : float
        guess of equilibrium pressure of the liquid [Pa]
    y : array_like
        vapour phase composition
    T : float
        temperaure of the vapour [K]
    model : object
        created from mixture and saftvrmie function
    good_initial: bool, optional
        if True skip succesive substitution and solves by Newton's Method.
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    Xass0 : list, optional
        if supplied X association non-bonded fraction sites initial value
    not_in_x_list : list, optional
        index of components not present in the liquid phase (default is empty)
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        liquid mole fraction vector
    P : float
        equilibrium pressure [Pa]

    """
    nc = model.nc
    if len(x_guess) != nc or len(y) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    if np.any([i > (nc-1) for i in not_in_x_list]):
        raise Exception('Index of components not_in_y_list must be less than nc')

    not_in_x = np.zeros(nc, dtype=bool)
    not_in_x[not_in_x_list] = 1
    in_x = np.logical_not(not_in_x)

    temp_aux = model.temperature_aux(T)

    global vl, vv, Xassl, Xassv
    vl0, vv0 = v0
    Xassl0, Xassv0 = Xass0

    it = 0
    itmax = 10
    tol = 1e-8

    P = P_guess
    out = dew_sus(P, y, temp_aux, 'T', x_guess, model, vl0, vv0,
                  Xassl0, Xassv0, not_in_x=not_in_x)
    f, X, lnK, vl, vv, Xassl, Xassv = out
    error = np.abs(f)
    h = 1e-3

    method = 'quasi-newton + ASS'
    while error > tol and it <= itmax and not good_initial:
        it += 1
        out = dew_sus(P+h, y, temp_aux, 'T', X, model, vl, vv, Xassl, Xassv,
                      not_in_x=not_in_x)
        f1, X1, lnK1, vl, vv, Xassl, Xassv = out
        out = dew_sus(P, y, temp_aux, 'T', X, model, vl, vv, Xassl, Xassv, 
                      not_in_x=not_in_x)
        f, X, lnK, vl, vv, Xassl, Xassv = out
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
        inc0 = np.hstack([lnK[in_x], P])
        sol1 = root(dew_newton, inc0, args=(y, temp_aux, 'T', model, in_x))
        if sol1.success:
            method = 'second-order'
            error = np.linalg.norm(sol1.fun)
            it += sol1.nfev

            sol = sol1.x
            lnK_newton = sol[:-1]
            K_newton = np.exp(lnK_newton)
            X = np.zeros(nc)
            X[in_x] = y[in_x] / K_newton
            X /= np.sum(X)
            P = sol[-1]

            rhol, Xassl = model.density_aux(X, temp_aux, P, 'L', rho0=1./vl,
                                            Xass0=Xassl)
            rhov, Xassv = model.density_aux(y, temp_aux, P, 'V', rho0=1./vv,
                                            Xass0=Xassv)
            vl = 1./rhol
            vv = 1./rhov

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'Xassl': Xassl, 'state1': 'Liquid',
               'Y': y, 'v2': vv, 'Xassv': Xassv, 'state2': 'Vapor',
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return X, P


def dewTx(x_guess, T_guess, y, P, model, good_initial=False,
          v0=[None, None], Xass0=[None, None],
          not_in_x_list=[],
          full_output=False):
    """
    Dew point (T, y) -> (P, y)

    Solves dew point at given vapour composition and pressure. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    x_guess : array_like
        guess of liquid phase composition
    T_guess : float
        guess of equilibrium temperature of the liquid [K]
    y : array_like
        vapour phase composition
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
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        liquid mole fraction vector
    T : float
        equilibrium temperature [K]

    """

    nc = model.nc
    if len(x_guess) != nc or len(y) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    if np.any([i > (nc-1) for i in not_in_x_list]):
        raise Exception('Index of components not_in_y_list must be less than nc')

    not_in_x = np.zeros(nc, dtype=bool)
    not_in_x[not_in_x_list] = 1
    in_x = np.logical_not(not_in_x)

    global vl, vv, Xassl, Xassv
    vl0, vv0 = v0
    Xassl0, Xassv0 = Xass0

    it = 0
    itmax = 10
    tol = 1e-8

    T = T_guess
    out = dew_sus(T, y, P, 'P', x_guess, model, vl0, vv0, 
                  Xassl0, Xassv0, not_in_x=not_in_x)
    f, X, lnK, vl, vv, Xassl, Xassv = out
    error = np.abs(f)
    h = 1e-4
    method = 'quasi-newton + ASS'
    while error > tol and it <= itmax and not good_initial:
        it += 1
        out = dew_sus(T+h, y, P, 'P', X, model, vl, vv, 
                      Xassl, Xassv, not_in_x=not_in_x)
        f1, X1, lnK1, vl, vv, Xassl, Xassv = out
        out = dew_sus(T, y, P, 'P', X, model, vl, vv, 
                      Xassl, Xassv, not_in_x=not_in_x)
        f, X, lnK, vl, vv, Xassl, Xassv = out
        df = (f1-f)/h
        if np.isnan(df):
            df = 0.0
            it = 1.*itmax
        T -= f/df
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK[in_x], T])
        sol1 = root(dew_newton, inc0, args=(y, P, 'P', model, in_x))
        if sol1.success:
            method = 'second-order'
            error = np.linalg.norm(sol1.fun)
            it += sol1.nfev

            sol = sol1.x
            lnK_newton = sol[:-1]
            K_newton = np.exp(lnK_newton)
            X = np.zeros(nc)
            X[in_x] = y[in_x] / K_newton
            X /= np.sum(X)
            T = sol[-1]

            temp_aux = model.temperature_aux(T)
            rhol, Xassl = model.density_aux(X, temp_aux, P, 'L', rho0=1./vl,
                                            Xass0=Xassl)
            rhov, Xassv = model.density_aux(y, temp_aux, P, 'V', rho0=1./vv,
                                            Xass0=Xassv)
            vl = 1./rhol
            vv = 1./rhov

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'Xassl': Xassl, 'state1': 'Liquid',
               'Y': y, 'v2': vv, 'Xassv': Xassv, 'state2': 'Vapor',
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return X, T


__all__ = ['dewTx', 'dewPx']