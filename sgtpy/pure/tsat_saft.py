from scipy.optimize import brentq, newton
from .EquilibriumResult import EquilibriumResult


def fobj_tsat(T, P, saft):
    global vl, vv
    global Xassl, Xassv
    temp_aux = saft.temperature_aux(T)
    lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
    lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
    FO = lnphiv - lnphil
    return FO


def tsat(saft, P, T0=None, Tbounds=None, v0=[None, None], Xass0=[None, None],
         full_output=False):

    bool1 = T0 is None
    bool2 = Tbounds is None

    if bool1 and bool2:
        raise Exception('You must provide either Tbounds or T0')

    global Xassl, Xassv
    Xassl, Xassv = Xass0

    global vl, vv
    vl, vv = v0
    if not bool1:
        sol = newton(fobj_tsat, x0=T0, args=(P, saft),
                     full_output=True)
    elif not bool2:
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
