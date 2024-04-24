from __future__ import division, print_function, absolute_import
import numpy as np
from ..math import gdem
from scipy.optimize import minimize
from .equilibriumresult import EquilibriumResult
from ..constants import Na


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


def Gibbs_obj(ny_var, phases, Z, nx, ny, where_x, where_y,
              where_equilibria, temp_aux, P, model):
    '''
    Objective function to minimize Gibbs energy in biphasic flash
    '''
    ny[where_equilibria] = ny_var
    nx[where_equilibria] = Z[where_equilibria] - ny_var

    X = nx / np.sum(nx)
    Y = ny / np.sum(ny)

    global vx, vy, Xass_x, Xass_y
    with np.errstate(all='ignore'):
        lnfugx, vx, Xass_y = model.logfugef_aux(X, temp_aux, P, phases[0], vx,
                                                Xass_x)
        lnfugy, vy, Xass_x = model.logfugef_aux(Y, temp_aux, P, phases[1], vy,
                                                Xass_y)
        fugx = np.log(X) + lnfugx
        fugy = np.log(Y) + lnfugy

    fx = np.dot(nx[where_x], fugx[where_x])
    fy = np.dot(ny[where_y], fugy[where_y])

    f = fx + fy
    df = (fugy - fugx)[where_equilibria]
    return f, df


def Qfun_obj(inc, temp_aux, Pspec, model):
    nc = model.nc
    RT = Na / temp_aux[0]
    Pspec_by_RT = Pspec / RT

    ny = inc[:nc]
    nx = inc[nc:(nc+nc)]
    lnvy_total = inc[(nc+nc)]
    lnvx_total = inc[(nc+nc+1)]
    vy_total = np.exp(lnvy_total)
    vx_total = np.exp(lnvx_total)

    ny_total = np.sum(ny)
    nx_total = np.sum(nx)
    vy = vy_total / ny_total
    vx = vx_total / nx_total

    global Xass_y, Xass_x
    #### phase Y
    y = ny / ny_total
    rho_y = 1./vy
    rhom_y = Na * rho_y
    a_y, ax_y, Xass_y = model.dafcn_dxrho_aux(y, rhom_y, temp_aux, Xass_y)
    a_y /= RT
    ax_y /= RT
    afcn_y, dafcn_y = a_y

    Zy = dafcn_y * rhom_y
    Py_by_RT = Zy / vy
    mu_nvt_y = afcn_y + ax_y - np.dot(y, ax_y) + Zy
    afcn_y *= ny_total

    #### phase X
    x = nx / nx_total
    rho_x = 1./vx
    rhom_x = Na * rho_x
    a_x, ax_x, Xass_x = model.dafcn_dxrho_aux(x, rhom_x, temp_aux, Xass_x)
    a_x /= RT
    ax_x /= RT
    afcn_x, dafcn_x = a_x

    Zx = dafcn_x * rhom_x
    Px_by_RT = Zx / vx
    mu_nvt_x = afcn_x + ax_x - np.dot(x, ax_x) + Zx
    afcn_x *= nx_total

    ## Objective function
    A_total = afcn_y + afcn_x
    PV_total = (vy_total + vx_total) * Pspec_by_RT
    Q_total = A_total + PV_total

    # gradient
    dQ = np.zeros_like(inc)
    dQ[:nc] = mu_nvt_y
    dQ[nc:(nc+nc)] = mu_nvt_x
    dQ[(nc+nc)] = vy_total * (- Py_by_RT + Pspec_by_RT)
    dQ[(nc+nc+1)] = vx_total * (- Px_by_RT + Pspec_by_RT)

    return Q_total, dQ


def Qfun_mass_balance(inc, Z, nc):
    ny = inc[:nc]
    nx = inc[nc:(nc+nc)]
    of = ny + nx - Z
    return of


def Qfun_jac_mass_balance(inc, Z, nc):
    jac = np.zeros([nc, (nc+nc+2)])
    index0 = np.array([i for i in range(nc)])
    index1 = np.array([i+nc for i in range(nc)])

    jac[index0, index0] = 1.
    jac[index0, index1] = 1.
    return jac


def flash(x_guess, y_guess, equilibrium, Z, T, P, model, v0=[None, None],
          Xass0=[None, None], K_tol=1e-8, nacc=3, accelerate_every=5,
          not_in_x_list=[], not_in_y_list=[],
          maxiter_radfordrice=10, tol_rachfordrice=1e-8,
          good_initial=False,
          minimization_approach='gibbs',
          minimization_method='BFGS',
          minimization_tol=1e-12,
          minimization_options={},
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

    # setting up temperature and auxiliary variables
    temp_aux = model.temperature_aux(T)
    # setting up initial guesses
    X = x_guess
    Y = y_guess
    vx0, vy0 = v0
    Xass_x0, Xass_y0 = Xass0

    # setting up initial values for volume roots
    global vx, vy, Xass_x, Xass_y

    with np.errstate(all='ignore'):
        fugx, vx, Xass_x = model.logfugef_aux(X, temp_aux, P, equilibrium[0],
                                              vx0, Xass_x0)
        fugy, vy, Xass_y = model.logfugef_aux(Y, temp_aux, P, equilibrium[1],
                                              vy0, Xass_y0)
    lnK = fugx - fugy
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], Z[not_in_x], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta_new = (bmin + bmax)/2

    singlephase = False

    ##########################################
    # 1. Accelerated Successive Substitution #
    ##########################################
    method = "ASS"

    # dummy variables to track iterarions and errors
    e1 = 100.
    # itacc = 0
    # it = 0
    # it2 = 0
    # n = accelerate_every

    it_acc = 0
    it_total = 0
    it_max = nacc * accelerate_every

    while e1 > K_tol and it_total < it_max and not good_initial:
        it_total += 1
        it_acc += 1
        lnK_old = 1. * lnK
        beta, D, singlephase = rachfordrice(beta_new, K, Z,
                                            tol=tol_rachfordrice,
                                            maxiter=maxiter_radfordrice,
                                            not_in_y=not_in_y,
                                            not_in_x=not_in_x)

        # updating phase compositions
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
        # updating fugacity coefficients
        with np.errstate(all='ignore'):
            fugx, vx, Xass_x = model.logfugef_aux(X, temp_aux, P, equilibrium[0],
                                                  vx, Xass_x)
            fugy, vy, Xass_y = model.logfugef_aux(Y, temp_aux, P, equilibrium[1],
                                                  vy, Xass_y)
        lnK = fugx - fugy

        # acceleration
        if it_acc == (accelerate_every-3):
            lnK3 = lnK
        elif it_acc == (accelerate_every-2):
            lnK2 = lnK
        elif it_acc == (accelerate_every-1):
            lnK1 = lnK
        elif it_acc == accelerate_every:
            dacc = gdem(lnK, lnK1, lnK2, lnK3)
            lnK += dacc
            # reset the acceleration counter
            it_acc = 0

        K = np.exp(lnK)
        e1 = np.sum((lnK-lnK_old)**2, where=in_equilibria)

        # getting a feasible beta (if not in the feasible range)
        bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], Z[not_in_x], 0.]))
        bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
        if beta < bmin or beta > bmax:
            beta_new = (bmin + bmax) / 2.



    """
    while e1 > K_tol and itacc < nacc and not good_initial:
        it += 1
        it2 += 1
        lnK_old = lnK
        beta, D, singlephase = rachfordrice(beta_new, K, Z,
                                            tol=tol_rachfordrice,
                                            maxiter=maxiter_radfordrice,
                                            not_in_y=not_in_y,
                                            not_in_x=not_in_x)
        # updating phase compositions
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
        # updating fugacity coefficients
        with np.errstate(all='ignore'):
            fugx, vx, Xass_x = model.logfugef_aux(X, temp_aux, P, equilibrium[0],
                                                  vx, Xass_x)
            fugy, vy, Xass_y = model.logfugef_aux(Y, temp_aux, P, equilibrium[1],
                                                  vy, Xass_y)
        lnK = fugx - fugy

        # acceleration
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

        # getting a feasible beta (if not in the feasible range)
        bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], Z[not_in_x], 0.]))
        bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
        if beta < bmin or beta > bmax:
            beta_new = (bmin + bmax) / 2.

    """
    ###################################
    # 2. Minimization of Gibbs energy #
    ###################################

    # if e1 > K_tol and it_total == it_max and not singlephase:
    if good_initial:
        beta = beta_new

    if e1 > K_tol and not singlephase:
        # in case the minimization does not converge, we save the previous values
        vx_copy = 1. * vx
        vy_copy = 1. * vy
        Xass_x_copy = 1. * Xass_x
        Xass_y_copy = 1. * Xass_y

        # PT flash by minimizing Gibbs energy
        if minimization_approach == 'gibbs':
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

            where_x = np.logical_and(z_notzero, in_x)
            where_y = np.logical_and(z_notzero, in_y)
            where_equilibria = np.logical_and(z_notzero, in_equilibria)
            ny_var = beta * Y[where_equilibria]

            # bounds = [(0., ny_max) for ny_max in Z[where_equilibria]]

            args = (equilibrium, Z, nx, ny, where_x, where_y, where_equilibria, temp_aux, P, model)
            ny_sol = minimize(fobj, ny_var, jac=jac, method=minimization_method, hess=hess,
                              tol=minimization_tol, args=args, options=minimization_options)

            if ny_sol.success:
                method = "Gibbs_minimization"
                ny_var = ny_sol.x
                ny[where_equilibria] = ny_var
                nx[where_equilibria] = Z[where_equilibria] - ny_var
                beta = np.sum(ny)
                X = nx / np.sum(nx)
                Y = ny / beta

                # updating volume roots for founded equilibria compositions and volumes
                rho_x, Xass_x = model.density_aux(X, temp_aux, P, equilibrium[0],
                                                  rho0=1./vx, Xass0=Xass_x)
                rho_y, Xass_y = model.density_aux(Y, temp_aux, P, equilibrium[1],
                                                  rho0=1./vy, Xass0=Xass_y)
                vx = 1./rho_x
                vy = 1./rho_y

                it_total += ny_sol.nit
                e1 = np.linalg.norm(ny_sol.jac)
            else:
                vx = vx_copy
                vy = vy_copy
                Xass_x = Xass_x_copy
                Xass_y = Xass_y_copy

        #  PT flash by minimizing Gibbs energy (using a helmholtz approach)
        elif minimization_approach == 'helmholtz':
            fobj = Qfun_obj
            jac = True
            hess = None

            # setting up initial guesses
            z_notzero = Z != 0

            ny = beta * Y
            nx = Z - ny

            ny_total = np.sum(ny)
            nx_total = np.sum(nx)
            vy_total = vy*ny_total
            vx_total = vx*nx_total

            lnvy_total = np.log(vy_total)
            lnvx_total = np.log(vx_total)

            inc = np.hstack([ny, nx, lnvy_total, lnvx_total])

            # bounds for mole phase fractions
            bounds = [(0, x) for x in Z] + [(0, x) for x in Z] + [(None, None), (None, None)]
            # constraints for mass balance
            cons = [{'type': 'eq', 'fun': Qfun_mass_balance, 'jac': Qfun_jac_mass_balance, 'args': (Z, nc)}]
            # This minimization requires to check the mass balance constraints,
            # I did some test and SLQP is the method that worked "the best"
            Q_sol = minimize(fobj, inc, jac=jac, method='SLSQP', hess=hess,
                             tol=minimization_tol, args=(temp_aux, P, model),
                             bounds=bounds, constraints=cons, options=minimization_options)
            print(Q_sol)
            if Q_sol.success:
                method = "Helmholtz_minimization"
                inc = Q_sol.x
                ny = inc[:nc]
                nx = inc[nc:(nc+nc)]
                lnvy_total = inc[(nc+nc)]
                lnvx_total = inc[(nc+nc+1)]

                vy_total = np.exp(lnvy_total)
                vx_total = np.exp(lnvx_total)

                ny_total = np.sum(ny)
                nx_total = np.sum(nx)
                vy = vy_total / ny_total
                vx = vx_total / nx_total

                beta = ny_total
                Y = ny / ny_total
                X = nx / nx_total

                jac = Q_sol.jac
                qerror = np.hstack([jac[:nc] - jac[nc:(nc+nc)], jac[(nc+nc):]])
                e1 = np.linalg.norm(qerror)

                it_total += Q_sol.nit
            else:
                vx = vx_copy
                vy = vy_copy
                Xass_x = Xass_x_copy
                Xass_y = Xass_y_copy

        else:
            raise Exception('Minimization approach not recognized. Use "gibbs" or "helmholtz"')

    if beta == 1.0:
        X = Y.copy()
    elif beta == 0.:
        Y = X.copy()

    if full_output:
        sol = {'T': T, 'P': P, 'beta': beta, 'error': e1, 'iter': it_total,
               'X': X, 'v1': vx, 'Xass1': Xass_x, 'state1': equilibrium[0],
               'Y': Y, 'v2': vy, 'Xass2': Xass_y, 'state2': equilibrium[1],
               'method': method}
        out = EquilibriumResult(sol)
        return out

    return X, Y, beta
