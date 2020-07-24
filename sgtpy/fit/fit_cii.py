import numpy as np
from ..math import gauss


def fit_cii(tension, Tsat, Psat, rhol, rhov, eos, n=100):
    n = 100
    roots, weigths = gauss(n)
    tena = np.zeros_like(Tsat)
    ndata = len(tension)
    for i in range(ndata):
        Tfactor, Pfactor, rofactor, tenfactor = eos.sgt_adim_fit(Tsat[i])
        rola = rhol[i] * rofactor
        rova = rhov[i] * rofactor
        Tad = Tsat[i] * Tfactor
        Pad = Psat[i] * Pfactor
        # Equilibrium chemical potential
        mu0 = eos.muad(rova, Tad)

        roi = (rola-rova) * roots + rova
        wreal = (rola-rova) * weigths
        dOm = np.zeros(n)
        for j in range(n):
            dOm[j] = eos.dOm(roi[j], Tad, mu0, Pad)
        tenint = np.nan_to_num(np.sqrt(2*dOm))
        tena[i] = np.dot(wreal, tenint)
        tena[i] *= tenfactor

    cii0 = (tension/tena)**2
    return np.mean(cii0)
