from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root, minimize_scalar
from ..constants import kb, Na
from .EquilibriumResult import EquilibriumResult
from warnings import warn

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


# objective functions for Pmin and Pmax initiation method
def fobj_pmax(rho, temp_aux, self):
    return - self.pressure_aux(rho, temp_aux)[0]


def fobj_pmin(rho, temp_aux, self):
    return self.pressure_aux(rho, temp_aux)[0]


def psat(saft, T, P0=None, v0=[None, None], Xass0=[None, None],
         full_output=True):

    if saft.critical:
        if T >= saft.Tc:
            warn('Temperature is greater than critical temperature, returning critical point')
            if full_output:
                dict = {'T': saft.Tc, 'P': saft.Pc, 'vl': 1./saft.rhoc,
                        'vv': 1./saft.rhoc, 'Xassl': Xass0[0],
                        'Xassv': Xass0[1], 'success': False, 'iterations': 0}
                out = EquilibriumResult(dict)
            else:
                out = saft.Pc, 1./saft.rhoc, 1./saft.rhoc
            return out

    temp_aux = saft.temperature_aux(T)
    beta = temp_aux[0]
    RT = Na/beta

    global Xassl, Xassv
    Xassl, Xassv = Xass0

    vl, vv = v0
    P0input = P0 is None
    vl0input = vl is None
    vv0input = vv is None
    v0input = not vl0input and not vv0input

    init_method = None
    if saft.critical:
        Tr = T/saft.Tc
        if Tr <= 0.8 and P0input:
            if not v0input:
                init_method = 'zero-pressure'

        elif 0.8 < Tr < 1. and P0input:
            if not v0input:
                init_method = 'pmin-pmax'

    if init_method is None:
        if not P0input:
            good_initial = False
            P = P0
        elif v0input:
            good_initial = True
        else:
            raise Exception('You need to provide either initial pressure or both volumes')
    elif init_method == 'zero-pressure':
        rholP0, XassP0 = saft.density_aux(temp_aux, 0., 'L')
        aresP0, XassP0 = saft.ares(rholP0*Na, T, XassP0)
        logfugl0 = aresP0 - 1. + np.log(RT*rholP0)
        fugl0 = np.exp(logfugl0)
        P = fugl0
        vl = 1./rholP0
        Xassl = XassP0
        good_initial = False
    elif init_method == 'pmin-pmax':
        limits_rhov = [saft.rhoc*1e-10, saft.rhoc]
        minpmax = minimize_scalar(fobj_pmax, bounds=limits_rhov,
                                  args=(temp_aux, saft), method='bounded')
        limits_rhol = [saft.rhoc, 5*saft.rhoc]
        minpmin = minimize_scalar(fobj_pmin, bounds=limits_rhol,
                                  args=(temp_aux, saft), method='bounded')
        P = (np.max([0, minpmin.fun]) - minpmax.fun) / 2.
        good_initial = False

    if not good_initial:
        lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
        lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
        FO = lnphiv - lnphil
        dFO = (vv - vl)/RT
        dP = FO/dFO
        if dP > P:
            dP /= 2
        P -= dP
        for i in range(15):
            lnphiv, vv, Xassv = saft.logfug_aux(temp_aux, P, 'V', vv, Xassv)
            lnphil, vl, Xassl = saft.logfug_aux(temp_aux, P, 'L', vl, Xassl)
            FO = lnphiv - lnphil
            dFO = (vv - vl)/RT
            P -= FO/dFO
            success = abs(FO) <= 1e-8
            if success:
                break
        if not success:
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
        success = sol.success
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
                'Xassv': Xassv, 'success': success, 'iterations': i}
        out = EquilibriumResult(dict)
    else:
        out = P, vl, vv
    return out
