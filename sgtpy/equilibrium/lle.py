from __future__ import division, print_function, absolute_import
import numpy as np
from .stability import tpd_minimas
from .multiflash import multiflash


def lle(x0, w0, Z, T, P, model, v0=[None, None], Xass0=[None, None],
        K_tol=1e-8, nacc=5, full_output=False, tetha_max=10.):
    """
    Liquid liquid equilibrium (z,T,P) -> (x,w,beta)

    Solves liquid liquid equilibrium from multicomponent mixtures at given
    pressure, temperature and overall composition.

    Parameters
    ----------

    x0 : array_like
        initial guess for liquid phase 1
    w0 : array_like
        initial guess for liquid phase 2
    z : array_like
        overal composition of mix
    T : float
        absolute temperature [K].
    P : float
        pressure [Pa]
    model : object
        created from mixture and saftvrmie function
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    K_tol : float, optional
        Desired accuracy of K (= W/X) vector
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
    full_output: bool, optional
        wheter to outputs all calculation info
    tetha_max : float, optional
        maximum value for stability variables. Default is 10.

    Returns
    -------
    X : array_like
        liquid 1 mole fraction vector
    W : array_like
        liquid 2 mole fraction vector
    beta : float
        phase fraction of liquid 2

    """
    nc = model.nc
    if len(x0) != nc or len(w0) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    temp_aux = model.temperature_aux(T)

    equilibrio = ['L', 'L']
    fugx, v1, Xass1 = model.logfugef_aux(x0, temp_aux, P, equilibrio[0], v0[0],
                                         Xass0[0])
    fugw, v2, Xass2 = model.logfugef_aux(w0, temp_aux, P, equilibrio[1], v0[1],
                                         Xass0[1])
    lnK = fugx - fugw
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta = (bmin + bmax)/2
    X0 = np.array([x0, w0])

    beta0 = np.array([1-beta, beta, 0.])

    out = multiflash(X0, beta0, equilibrio, Z, T, P, model,
                     [v1, v2], [Xass1, Xass2], K_tol, nacc, True, tetha_max)
    Xm, beta, tetha, v, Xass = out.X, out.beta, out.tetha, out.v, out.Xass

    if tetha > 0:
        xes, tpd_min2 = tpd_minimas(2, Xm[0], T, P, model, 'L', 'L',
                                    v[0], v[0])
        X0 = np.asarray(xes)
        beta0 = np.hstack([beta, 0.])
        out = multiflash(X0, beta0, equilibrio, Z, T, P, model, v, Xass,
                         K_tol, nacc, True, tetha_max)
        Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v

    X, W = Xm
    if tetha > 0:
        W = X.copy()

    if full_output:
        return out

    return X, W, beta[1]


__all__ = ['lle']
