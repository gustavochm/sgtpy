from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from .equilibriumresult import EquilibriumResult


def haz_objb(inc, T_P, tipo, model):

    X, W, Y, P_T = np.array_split(inc, 4)
    P_T = P_T[0]
    if tipo == 'T':
        P = P_T
        temp_aux = T_P
    elif tipo == 'P':
        T = P_T
        temp_aux = model.temperature_aux(T)
        P = T_P

    global vx, vw, vy
    global Xassx, Xassw, Xassy

    fugX, vx, Xassx = model.logfugef_aux(X, temp_aux, P, 'L', vx, Xassx)
    fugW, vw, Xassw = model.logfugef_aux(W, temp_aux, P, 'L', vw, Xassw)
    fugY, vy, Xassy = model.logfugef_aux(Y, temp_aux, P, 'V', vy, Xassy)

    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)
    return np.hstack([K1*X-Y, K2*X-W, X.sum()-1, Y.sum()-1, W.sum()-1])


def vlleb(X0, W0, Y0, P_T, T_P, spec, model, v0=[None, None, None],
          Xass0=[None, None, None], full_output=False):
    '''
    Solves liquid liquid vapour equilibrium for binary mixtures.
    (T,P) -> (x,w,y)

    Parameters
    ----------

    X0 : array_like
        guess composition of phase 1
    W0 : array_like
        guess composition of phase 1
    Y0 : array_like
        guess composition of phase 2
    P_T : float
        absolute temperature [K] or pressure [Pa]
    T_P : floar
        absolute temperature [K] or pressure [Pa]
    spec: string
        'T' if T_P is temperature or 'P' if pressure.
    model : object
        created from mixture and saftvrmie function
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    Xass0, list, optional
        if supplied used to solve the associtaion nonbonded sites fraction
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------

    X : array_like
        liquid1 mole fraction vector
    W : array_like
        liquid2 mole fraction vector
    Y : array_like
        vapour mole fraction fector
    var: float
        temperature or pressure, depending of specification

    '''

    nc = model.nc

    if nc != 2:
        raise Exception('3 phase equilibria for binary mixtures')

    if len(X0) != nc or len(W0) != nc or len(Y0) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vx, vw, vy
    vx, vw, vy = v0

    global Xassx, Xassw, Xassy
    Xassx, Xassw, Xassy = Xass0

    if spec == 'T':
        temp_aux = model.temperature_aux(T_P)
        sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                    args=(temp_aux, spec, model))
    elif spec == 'P':
        sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                    args=(T_P, spec, model))
    else:
        raise Exception('Specification not known')

    error = np.linalg.norm(sol1.fun)
    nfev = sol1.nfev
    sol = sol1.x
    if np.any(sol < 0):
        raise Exception('negative Composition or T/P  founded')
    X, W, Y, var = np.array_split(sol, 4)
    var = var[0]
    if full_output:
        if spec == 'T':
            P = var
            T = T_P
        elif spec == 'P':
            T = var
            temp_aux = model.temperature_aux(T)
            P = T_P

        rhox, Xassx = model.density_aux(X, temp_aux, P, 'L', rho0=1./vx,
                                        Xass0=Xassx)
        rhow, Xassw = model.density_aux(W, temp_aux, P, 'L', rho0=1./vw,
                                        Xass0=Xassw)
        rhoy, Xassy = model.density_aux(Y, temp_aux, P, 'V', rho0=1./vy,
                                        Xass0=Xassy)
        vx = 1./rhox
        vw = 1./rhow
        vy = 1./rhoy

        inc = {'T': T, 'P': P, 'error': error, 'nfev': nfev,
               'X': X, 'vx': vx, 'Xassx': Xassx, 'statex': 'Liquid',
               'W': W, 'vw': vw, 'Xassw': Xassw, 'statew': 'Liquid',
               'Y': Y, 'vy': vy, 'Xassy': Xassy, 'statey': 'Vapor'}
        out = EquilibriumResult(inc)
        return out

    return X, W, Y, var
