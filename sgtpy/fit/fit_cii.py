import numpy as np
from ..math import gauss


def fit_cii(tension, Tsat, Psat, rhol, rhov, eos, n=100):
    """
    fit_cii
    Optimize influence parameter for SGT for pure componentes


    Parameters
    ----------
    tension : array
        Experimental interfacial tension array
    Tsat : array
        Experimental saturation temperature [K]
    Psat : array
        Computed saturation pressure [Pa]
    rhol : array
        Computed liquid density [mol/m3]
    rhov : array
        Computed vapor density [mol/m3]
    eos : object
        created from pure fluid and saftvrmie function

    Returns
    -------
    cii : float
        Influence Parameter of the fluid
    """
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
