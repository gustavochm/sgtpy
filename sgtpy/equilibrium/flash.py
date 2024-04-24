from __future__ import division, print_function, absolute_import
import numpy as np
from ..math import gdem
from scipy.optimize import minimize
from .equilibriumresult import EquilibriumResult


def rachfordrice(beta, K, Z, tol=1e-8, maxiter=20, not_in_x=[], not_in_y=[]):
    '''
    Solves Rachford Rice equation by Halley's method
    '''
    K1 = K-1.

    ZK = Z*K
    ZK[not_in_y] = 0.0
    ZK[not_in_x] = 1e4  # setting it to a high number

    Z_div_K = Z/K
    Z_div_K[not_in_y] = 1e4  # setting it to a high number
    Z_div_K[not_in_x] = 0.0

    g0 = np.sum(ZK) - 1.
    g1 = 1. - np.sum(Z_div_K)

    singlephase = False

    if g0 < 0:
        beta = 0.
        D = np.ones_like(Z)
        singlephase = True
    elif g1 > 0:
        beta = 1.
        D = 1 + K1
        singlephase = True
    it = 0
    e = 1.

    while e > tol and it < maxiter and not singlephase:
        it += 1
        D = 1. + beta*K1
        KD = K1/D

        # modification for not_in_y components Ki -> 0
        KD[not_in_y] = -1. / (1. - beta)

        # modification for not_in_y components Ki -> infty
        KD[not_in_x] = 1. / beta

        fo = np.dot(Z, KD)
        dfo = - np.dot(Z, KD**2)
        d2fo = 2*np.dot(Z, KD**3)
        dbeta = - (2*fo*dfo)/(2*dfo**2-fo*d2fo)
        beta += dbeta
        e = np.abs(dbeta)

    return beta, D, singlephase


def Gibbs_obj(ny_var, phases, Z, z_notzero, nx, ny, in_x, in_y,
              where_equilibria, temp_aux, P, model):
    '''
    Objective function to minimize Gibbs energy in biphasic flash
    '''
    ny[where_equilibria] = ny_var
    nx[where_equilibria] = Z[where_equilibria] - ny_var

    X = nx / np.sum(nx)
    Y = ny / np.sum(ny)

    global v1, v2, Xass1, Xass2
    with np.errstate(all='ignore'):
        lnfugx, v1, Xass1 = model.logfugef_aux(X, temp_aux, P, phases[0], v1,
                                               Xass1)
        lnfugy, v2, Xass2 = model.logfugef_aux(Y, temp_aux, P, phases[1], v2,
                                               Xass2)
        fugx = np.log(X[z_notzero]) + lnfugx[z_notzero]
        fugy = np.log(Y[z_notzero]) + lnfugy[z_notzero]

    fx = np.dot(nx[in_x], fugx[in_x])
    fy = np.dot(ny[in_y], fugy[in_y])

    f = fx + fy
    df = (fugy - fugx)[where_equilibria]
    return f, df


def flash(x_guess, y_guess, equilibrium, Z, T, P, model, v0=[None, None],
          Xass0=[None, None], K_tol=1e-8, nacc=3, accelerate_every=5,
          not_in_x_list=[], not_in_y_list=[],
          maxiter_radfordrice=10, tol_rachfordrice=1e-8,
          minimization_method='BFGS',
          full_output=False):
    """
    Isothermic isobaric flash (z, T, P) -> (x,y,beta)

    Parameters
    ----------

    x_guess : array_like
        guess composition of phase 1
    y_guess : array_like
        guess composition of phase 2
    equilibrium : string
        'LL' for LLE, 'LV' for VLE
    z : array_like
        overall system composition
    T : float
        absolute temperature [K].
    P : float
        pressure [Pa]
    model : object
        created from mixture and saftvrmie function
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    K_tol : float, optional
        Desired accuracy of K (= Y/X) vector
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
    accelerate_every : int, optional
        number of iterations to perform before acceleration in successive substitution
        must be equal or greater than 4
    not_in_x_list : list, optional
        index of components not present in phase 1 (default is empty)
    not_in_y_list : list, optional
        index of components not present in phase 2 (default is empty)
    maxiter_radfordrice : int, optional
        maximum number of iterations for Rachford-Rice equation
    tol_rachfordrice : float, optional
        tolerance for Rachford-Rice solver
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        phase 1 composition
    Y : array_like
        phase 2 composition
    beta : float
        phase 2 phase fraction
    """
    nc = model.nc
    if len(x_guess) != nc or len(y_guess) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    if np.any([i > (nc-1) for i in not_in_y_list]):
        raise Exception('Index of components not_in_y_list must be less than nc')

    if np.any([i > (nc-1) for i in not_in_y_list]):
        raise Exception('Index of components not_in_x_list must be less than nc')

    # creating list for non-condensable/non-volatiles
    not_in_x = np.zeros(nc, dtype=bool)
    not_in_x[not_in_x_list] = 1
    in_x = np.logical_not(not_in_x)

    not_in_y = np.zeros(nc, dtype=bool)
    not_in_y[not_in_y_list] = 1
    in_y = np.logical_not(not_in_y)

    in_equilibria = np.logical_and(in_x, in_y)

    temp_aux = model.temperature_aux(T)
    v10, v20 = v0
    Xass10, Xass20 = Xass0
    e1 = 1
    itacc = 0
    it = 0
    it2 = 0
    n = accelerate_every
    # nacc = 3
    X = x_guess
    Y = y_guess
    global v1, v2, Xass1, Xass2

    with np.errstate(all='ignore'):
        fugx, v1, Xass1 = model.logfugef_aux(X, temp_aux, P, equilibrium[0],
                                             v10, Xass10)
        fugy, v2, Xass2 = model.logfugef_aux(Y, temp_aux, P, equilibrium[1],
                                             v20, Xass20)
    lnK = fugx - fugy
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], Z[not_in_x], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta_new = (bmin + bmax)/2

    singlephase = False
    method = "ASS"
    while e1 > K_tol and itacc < nacc:
        it += 1
        it2 += 1
        lnK_old = lnK
        beta, D, singlephase = rachfordrice(beta_new, K, Z,
                                            tol=tol_rachfordrice,
                                            maxiter=maxiter_radfordrice,
                                            not_in_y=not_in_y,
                                            not_in_x=not_in_x)
        X = Z/D
        Y = X*K
        # modification for for non-condensable/non-volatiles
        if not singlephase:
            # modification for not_in_y components Ki -> 0
            X[not_in_y] = Z[not_in_y] / (1. - beta)
            # modification for not_in_x components Ki -> infty
            Y[not_in_x] = Z[not_in_x] / beta
        Y[not_in_y] = 0.
        X[not_in_x] = 0.
        X /= X.sum()
        Y /= Y.sum()
        with np.errstate(all='ignore'):
            fugx, v1, Xass1 = model.logfugef_aux(X, temp_aux, P, equilibrium[0],
                                                 v1, Xass1)
            fugy, v2, Xass2 = model.logfugef_aux(Y, temp_aux, P, equilibrium[1],
                                                 v2, Xass2)

        lnK = fugx - fugy
        if it == (n-3):
            lnK3 = lnK
        elif it == (n-2):
            lnK2 = lnK
        elif it == (n-1):
            lnK1 = lnK
        elif it == n:
            it = 0
            itacc += 1
            dacc = gdem(lnK, lnK1, lnK2, lnK3)
            lnK += dacc
        K = np.exp(lnK)
        e1 = np.sum((lnK-lnK_old)**2, where=in_equilibria)

        bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], Z[not_in_x], 0.]))
        bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
        if beta < bmin or beta > bmax:
            beta_new = (bmin + bmax) / 2.

    if e1 > K_tol and itacc == nacc and not singlephase:
        fobj = Gibbs_obj
        jac = True
        hess = None
        # setting up initial guesses
        z_notzero = Z != 0

        ny = np.zeros(nc)
        nx = np.zeros(nc)

        ny[not_in_y] = 0.
        ny[not_in_x] = Z[not_in_x]

        nx[not_in_x] = 0.
        nx[not_in_y] = Z[not_in_y]

        where_equilibria = np.logical_and(z_notzero, in_equilibria)
        ny_var = beta * Y[where_equilibria]

        # bounds = [(0., ny_max) for ny_max in Z[where_equilibria]]

        v1_copy = 1. * v1
        v2_copy = 1. * v2
        Xass1_copy = 1. * Xass1
        Xass2_copy = 1. * Xass2

        ny_sol = minimize(fobj, ny_var, jac=jac, method=minimization_method, hess=hess,
                          args=(equilibrium, Z, z_notzero, nx, ny, in_x, in_y, where_equilibria, temp_aux, P, model))

        if ny_sol.success:
            method = "Gibbs_minimization"
            ny_var = ny_sol.x
            ny[where_equilibria] = ny_var
            nx[where_equilibria] = Z[where_equilibria] - ny_var
            beta = np.sum(ny)
            X = nx / np.sum(nx)
            Y = ny / beta

            # updating volume roots for founded equilibria compositions
            rho1, Xass1 = model.density_aux(X, temp_aux, P, equilibrium[0],
                                            rho0=1./v1, Xass0=Xass1)
            rho2, Xass2 = model.density_aux(Y, temp_aux, P, equilibrium[1],
                                            rho0=1./v2, Xass0=Xass2)
            v1 = 1./rho1
            v2 = 1./rho2

            it2 += ny_sol.nit
            e1 = np.linalg.norm(ny_sol.jac)
        else:
            v1 = v1_copy
            v2 = v2_copy
            Xass1 = Xass1_copy
            Xass2 = Xass2_copy

    if beta == 1.0:
        X = Y.copy()
    elif beta == 0.:
        Y = X.copy()

    if full_output:
        sol = {'T': T, 'P': P, 'beta': beta, 'error': e1, 'iter': it2,
               'X': X, 'v1': v1, 'Xass1': Xass1, 'state1': equilibrium[0],
               'Y': Y, 'v2': v2, 'Xass2': Xass2, 'state2': equilibrium[1],
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return X, Y, beta
