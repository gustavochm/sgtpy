import numpy as np
from scipy.optimize import root
from ..constants import kb, Na
from .EquilibriumResult import EquilibriumResult


R = Na * kb


def mu_obj(rho, temp_aux, saft):
    rhol, rhov = Na * rho

    global Xassl, Xassv
    dal, Xassl = saft.d2afcn_aux(rhol, temp_aux, Xassl)
    afcnl, dafcnl, d2afcnl = dal
    Pl = rhol**2 * dafcnl / Na
    dPl = (2 * rhol * dafcnl + rhol**2 * d2afcnl)

    mul = afcnl + rhol*dafcnl
    dmul = Na * (rhol*d2afcnl + 2*dafcnl)

    dav, Xassv = saft.d2afcn_aux(rhov, temp_aux)
    afcnv, dafcnv, d2afcnv = dav
    Pv = rhov**2 * dafcnv / Na
    dPv = (2 * rhov * dafcnv + rhov**2 * d2afcnv)

    muv = afcnv + rhov * dafcnv
    dmuv = Na * (rhol*d2afcnv + 2*dafcnv)

    FO = np.array([mul-muv, Pl - Pv])
    dFO = np.array([[dmul, -dmuv],
                   [dPl, - dPv]])
    return FO, dFO


def psat(saft, T, P0=None, v0=[None, None], Xass0=[None, None],
         full_output=True):

    P0input = P0 is None
    v0input = v0 == [None, None]

    if P0input and v0input:
        print('You need to provide either initial pressure or volumes')
    elif not P0input:
        good_initial = False
        P = P0
    elif not v0input:
        good_initial = True

    temp_aux = saft.temperature_aux(T)
    beta = temp_aux[0]
    RT = Na/beta

    global Xassl, Xassv
    Xassl, Xassv = Xass0

    vl, vv = v0
    if not good_initial:
        lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
        lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
        FO = lnphiv - lnphil
        dFO = (vv - vl)/RT
        dP = FO/dFO
        if dP > P:
            dP /= 2
        P -= dP
        for i in range(10):
            lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
            lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
            FO = lnphiv - lnphil
            dFO = (vv - vl)/RT
            P -= FO/dFO
            sucess = abs(FO) <= 1e-8
            if sucess:
                break
        if not sucess:
            rho0 = 1. / np.array([vl, vv])
            sol = root(mu_obj, rho0, args=(temp_aux, saft), jac=True)
            sucess = sol.success
            i += sol.nfev
            rhol, rhov = sol.x
            vl, vv = 1./sol.x
            rhomolecular = rhol * Na
            dal, Xassl = saft.dafcn_aux(rhomolecular, temp_aux, Xassl)
            afcn, dafcn = dal
            P = rhomolecular**2 * dafcn/Na
    else:
        rho0 = 1. / np.asarray([v0])
        sol = root(mu_obj, rho0, args=(temp_aux, saft), jac=True)
        sucess = sol.success
        i = sol.nfev
        if sol.success:
            rhol, rhov = sol.x
            vl, vv = 1./sol.x
            rhomolecular = rhol * Na
            dal, Xassl = saft.dafcn_aux(rhomolecular, temp_aux, Xassl)
            afcn, dafcn = dal
            P = rhomolecular**2 * dafcn/Na
        else:
            P = None

    if full_output:
        dict = {'T': T, 'P': P, 'vl': vl, 'vv': vv, 'Xassl': Xassl,
                'Xassv': Xassv, 'sucess': sucess, 'iterations': i}
        out = EquilibriumResult(dict)

    else:
        out = P, vl, vv
    return out
