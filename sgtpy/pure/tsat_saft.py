from scipy.optimize import brentq
from .EquilibriumResult import EquilibriumResult


def fobj_tsat(T, P, saft):
    global vl, vv
    global Xassl, Xassv
    temp_aux = saft.temperature_aux(T)
    lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
    lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
    FO = lnphiv - lnphil
    return FO


def tsat(saft, P, Tbounds, v0=[None, None], Xass0=[None, None],
         full_output=False):

    global Xassl, Xassv
    Xassl, Xassv = Xass0

    global vl, vv
    vl, vv = v0

    sol = brentq(fobj_tsat, Tbounds[0], Tbounds[1], args=(P, saft),
                 full_output=True)
    Tsat = sol[0]
    if full_output:
        dict = {'T': Tsat, 'P': P, 'vl': vl, 'vv': vv, 'Xassl': Xassl,
                'Xassv': Xassv, 'sucess': sol[1].converged,
                'iterations': sol[1].iterations}
        out = EquilibriumResult(dict)
    else:
        out = (Tsat, vl, vv)

    return out
