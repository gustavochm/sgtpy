from __future__ import division, print_function, absolute_import
import numpy as np
from ..math import gauss
from .tensionresult import TensionResult


def sgt_pure(rhov, rhol, Tsat, Psat, model, n=100, full_output=False,
             check_eq=True):
    """
    SGT for pure component (rhol, rhov, T, P) -> interfacial tension

    Parameters
    ----------
    rhov : float
        liquid phase density
    rhol : float
        vapor phase density
    Tsat : float
        saturation temperature
    Psat : float
        saturation pressure
    model : object
        created with an EoS
    n : int, optional
        number of collocation points for IFT integration
    full_output : bool, optional
        wheter to outputs all calculation info
    check_eq : bool, optional
        whether to check if given density vectors are in phase equilibria

    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    # roots and weights of Gauss quadrature
    roots, w = gauss(n)

    temp_aux = model.temperature_aux(Tsat)
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)

    rola = rhol * rofactor
    rova = rhov * rofactor
    # Tad = Tsat * Tfactor
    Pad = Psat * Pfactor

    # Equilibrium chemical potential
    mu0, Xass0 = model.muad_aux(rova, temp_aux)
    mu02, Xass1 = model.muad_aux(rola, temp_aux)
    if check_eq:
        if not np.allclose(mu0, mu02, rtol=1e-3):
            raise Exception('Not equilibria compositions, mu1 != mu2')

    roi = (rola-rova)*roots+rova
    wreal = np.abs((rola-rova)*w)

    dOm = np.zeros(n)
    i = 0
    dOm[i], Xass = model.dOm_aux(roi[i], temp_aux, mu0, Pad, Xass0)
    for i in range(1, n):
        dOm[i], Xass = model.dOm_aux(roi[i], temp_aux, mu0, Pad, Xass)

    tenint = np.nan_to_num(np.sqrt(2*dOm))
    ten = np.dot(wreal, tenint)
    ten *= tenfactor

    if full_output:
        zint = np.sqrt(1/(2*dOm))
        z = np.cumsum(wreal*zint)
        z *= zfactor
        roi /= rofactor
        dictresult = {'tension': ten, 'rho': roi, 'z': z,
                      'GPT': dOm}
        out = TensionResult(dictresult)
        return out

    return ten
