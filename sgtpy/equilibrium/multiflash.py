from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize
from ..math import gdem
from .equilibriumresult import EquilibriumResult


def multiflash_obj(inc, Z, K):

    inc[inc <= 1e-10] = 1e-10
    beta, tetha = np.array_split(inc, 2)
    ninc = len(inc)
    nbeta = len(beta)

    betar = beta[1:]
    Kexp = K.T*np.exp(tetha)
    K1 = Kexp - 1.  # nc x P - 1
    sum1 = 1 + K1@betar  # P - 1
    Xref = Z/sum1

    f1 = beta.sum() - 1.
    f2 = (K1.T*Xref).sum(axis=1)
    betatetha = (betar + tetha)
    f3 = (betar*tetha) / betatetha
    f = np.hstack([f1, f2, f3])

    jac = np.zeros([ninc, ninc])
    Id = np.eye(nbeta-1)

    # f1 derivative
    f1db = np.ones(nbeta)
    jac[0, :nbeta] = f1db

    # f2 derivative
    ter1 = Xref/sum1 * K1.T
    dfb = - ter1@K1
    jac[1:nbeta, 1:nbeta] = dfb
    dft = Id * (Xref*Kexp.T).sum(axis=1)
    ter2 = Kexp*betar
    dft -= ter1 @ ter2
    jac[1:nbeta, nbeta:] = dft

    # f3 derivative
    f3db = (tetha/betatetha)**2
    f3dt = (betar/betatetha)**2
    jac[nbeta:, 1:nbeta] = Id*f3db
    jac[nbeta:, nbeta:] = Id*f3dt

    return f, jac, Kexp, Xref


def gibbs_obj(ind, phases, Z, z_notzero, n_nonzero, temp_aux, P, model):

    nfase = len(phases)

    nc = model.nc
    ind = 1.*ind.reshape(nfase-1, n_nonzero)
    dep = Z[z_notzero] - ind.sum(axis=0)

    X = np.zeros((nfase, nc))
    X[1:, z_notzero] = ind
    X[0, z_notzero] = dep

    n = X.copy()
    X = (X.T/X.sum(axis=1)).T

    lnphi = np.zeros_like(X)
    global vg, Xassg

    for i, state in enumerate(phases):
        lnphi[i], vg[i], Xassg[i] = model.logfugef_aux(X[i], temp_aux, P,
                                                       state, vg[i], Xassg[i])
    with np.errstate(invalid='ignore'):
        fug = np.nan_to_num(np.log(X[:, z_notzero]) + lnphi[:, z_notzero])
    G = np.sum(n[:, z_notzero] * fug)
    dG = (fug[1:] - fug[0]).flatten()
    return G, dG


def multiflash(X0, betatetha, equilibrium, z, T, P, model, v0=[None],
               Xass0=[None], K_tol=1e-10, nacc=5, full_output=False):
    """
    multiflash (z,T,P) -> (x, w, y, ..., beta)

    Parameters
    ----------
    X0 : array_like
        (n_phases, nc) initial guess
    betatheta : array_like
        phase fractions and stability array
    equilibrium : list
        'L' for liquid, 'V' for vaopur phase. ['L', 'L', 'V'] for LLVE
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
        Desired accuracy of K (= X/Xr) vector
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        composition values matrix
    beta : array_like
        phase fraction array
    theta : array_like
        stability variables arrays

    """

    nfase = len(equilibrium)

    nc = model.nc
    if np.all(X0.shape != (nfase, nc)) or len(z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    error = 1
    it = 0
    itacc = 0
    ittotal = 0
    n = 5

    temp_aux = model.temperature_aux(T)

    X = X0.copy()
    lnphi = np.zeros_like(X)

    if len(v0) != 1 and len(v0) != nfase:
        v0 *= nfase
    v = v0.copy()
    if len(Xass0) != 1 and len(Xass0) != nfase:
        Xass0 *= nfase
    Xass = Xass0.copy()

    for i, state in enumerate(equilibrium):
        lnphi[i], v[i], Xass[i] = model.logfugef_aux(X[i], temp_aux, P, state,
                                                     v[i], Xass[i])

    lnK = lnphi[0] - lnphi[1:]
    K = np.exp(lnK)

    x = betatetha

    while error > K_tol and itacc < nacc:
        ittotal += 1
        it += 1
        lnK_old = lnK.copy()

        ef = 1.
        ex = 1.
        itin = 0
        while ef > 1e-8 and ex > 1e-8 and itin < 30:
            itin += 1
            f, jac, Kexp, Xref = multiflash_obj(x, z, K)
            dx = np.linalg.solve(jac, -f)
            x += dx
            ef = np.linalg.norm(f)
            ex = np.linalg.norm(dx)

        x[x <= 1e-10] = 0.
        beta, tetha = np.array_split(x, 2)
        beta /= beta.sum()

        # Update compositions
        X[0] = Xref
        X[1:] = Xref*Kexp.T
        X = np.abs(X)
        X = (X.T/X.sum(axis=1)).T

        # Update fugacity coefficient
        for i, state in enumerate(equilibrium):
            lnphi[i], v[i], Xass[i] = model.logfugef_aux(X[i], temp_aux, P,
                                                         state, v[i], Xass[i])
        lnK = lnphi[0] - lnphi[1:]
        error = np.sum((lnK - lnK_old)**2)

        # Accelerated succesive sustitution
        if it == (n-3):
            lnK3 = lnK.flatten()
        elif it == (n-2):
            lnK2 = lnK.flatten()
        elif it == (n-1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0
            itacc += 1
            lnKf = lnK.flatten()
            dacc = gdem(lnKf, lnK1, lnK2, lnK3).reshape(lnK.shape)
            lnK += dacc

        K = np.exp(lnK)

    if error > K_tol and itacc == nacc and ef < 1e-8 and np.all(tetha == 0):
        fobj = gibbs_obj
        jac = True
        hess = None
        method = 'BFGS'
        global vg, Xassg
        vg = v.copy()
        Xassg = Xass.copy()
        z_notzero = np.nonzero(z)[0]
        n_nonzero = len(z_notzero)

        ind0 = (X.copy()[:, z_notzero].T*beta).T[1:].flatten()

        ind1 = minimize(fobj, ind0, args=(equilibrium, z, z_notzero, n_nonzero,
                        temp_aux, P, model), jac=jac, method=method,
                        hess=hess)
        v = vg.copy()
        Xass = Xassg.copy()

        ittotal += ind1.nit
        error = np.linalg.norm(ind1.jac)
        nc = model.nc

        ind = 1.*ind0.reshape(nfase-1, n_nonzero)
        dep = z[z_notzero] - ind.sum(axis=0)

        X = np.zeros((nfase, nc))
        X[1:, z_notzero] = ind
        X[0, z_notzero] = dep
        beta = X.sum(axis=1)
        X = (X.T/beta).T
        # updating volume roots for founded equilibria compositions
        for i, state in enumerate(equilibrium):
            rho, Xass[i] = model.density_aux(X[i], temp_aux, P, state,
                                             rho0=1/v[i], Xass0=Xass[i])
            v[i] = 1./rho

    if full_output:
        sol = {'T': T, 'P': P, 'error_outer': error, 'error_inner': ef,
               'iter': ittotal, 'beta': beta, 'tetha': tetha, 'X': X, 'v': v,
               'Xass': Xass, 'states': equilibrium}
        out = EquilibriumResult(sol)
        return out

    return X, beta, tetha
