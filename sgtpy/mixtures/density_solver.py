from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from ..constants import Na


def dPsaft_fun(rho, x, temp_aux, saft):
    rhomolecular = Na * rho
    global Xass
    da, Xass = saft.d2afcn_drho_aux(x, rhomolecular, temp_aux, Xass)
    afcn, dafcn, d2afcn = da
    dPsaft = 2 * rhomolecular * dafcn + rhomolecular**2 * d2afcn
    return dPsaft


def Psaft_obj(rho, x, temp_aux, saft, Pspec):
    rhomolecular = Na * rho
    global Xass
    da, Xass = saft.dafcn_drho_aux(x, rhomolecular, temp_aux, Xass)
    afcn, dafcn = da
    Psaft = rhomolecular**2 * dafcn / Na
    return Psaft - Pspec


def density_newton_lim(rho_a, rho_b, x, temp_aux, P, Xass0, saft):
    rho = (rho_a + rho_b) / 2
    Psaft, dPsaft, Xass = saft.dP_drho_aux(x, rho, temp_aux, Xass0)
    for i in range(15):
        rho_old = rho
        FO = Psaft - P
        dFO = dPsaft
        drho = FO/dFO
        rho_new = rho - drho

        if FO > 0:
            rho_b = rho
        else:
            rho_a = rho

        if rho_a < rho_new < rho_b:
            rho = rho_new
        else:
            rho = (rho_a + rho_b) / 2

        if np.abs(rho - rho_old) < 1e-6:
            break
        Psaft, dPsaft, Xass = saft.dP_drho_aux(x, rho, temp_aux, Xass)
    return rho, Xass


def density_topliss(state, x, temp_aux, P, Xass0, saft):

    if state != 'L' and state != 'V':
        raise Warning("Not valid state. 'L' for liquid and 'V' for vapor.")

    beta = temp_aux[0]
    # lower boundary a zero density
    rho_lb = 1e-5
    dP_lb = Na / beta

    # Upper boundary limit at infinity pressure
    etamax = 0.7405
    rho_lim = (6 * etamax) / np.dot(x, (saft.ms * np.pi * saft.sigma**3)) / Na
    ub_sucess = False
    rho_ub = 0.4 * rho_lim
    it = 0
    P_ub, dP_ub, Xass_ub = saft.dP_drho_aux(x, rho_ub, temp_aux, Xass0)
    while not ub_sucess and it < 5:
        it += 1
        P_ub, dP_ub, Xass_ub = saft.dP_drho_aux(x, rho_ub, temp_aux, Xass_ub)
        rho_ub += 0.15 * rho_lim
        ub_sucess = P_ub > P and dP_ub > 0

    # Derivative calculation at zero density
    rho_lb1 = 1e-4 * rho_lim
    P_lb1, dP_lb1, Xass_lb = saft.dP_drho_aux(x, rho_lb1, temp_aux, Xass0)
    d2P_lb1 = (dP_lb1 - dP_lb) / rho_lb1
    if d2P_lb1 > 0:
        flag = 3
    else:
        flag = 1

    global Xass
    Xass = Xass0

    # Stage 1
    bracket = [rho_lb, rho_ub]
    if flag == 1:
        # Found inflexion point
        sol_inf = minimize_scalar(dPsaft_fun, args=(x, temp_aux, saft),
                                  bounds=bracket, method='Bounded',
                                  options={'xatol': 1e-1})
        rho_inf = sol_inf.x
        dP_inf = sol_inf.fun
        if dP_inf > 0:
            flag = 3
        else:
            flag = 2

    # Stage 2
    if flag == 2:
        if state == 'L':
            bracket[0] = rho_inf
        elif state == 'V':
            bracket[1] = rho_inf
        rho_ext = brentq(dPsaft_fun, bracket[0], bracket[1],
                         args=(x, temp_aux, saft), xtol=1e-2)
        P_ext, dP_ext, Xass = saft.dP_drho_aux(x, rho_ext, temp_aux, Xass)
        if P_ext > P and state == 'V':
            bracket[1] = rho_ext
        elif P_ext < P and state == 'L':
            bracket[0] = rho_ext
        else:
            flag = -1

    if flag == -1:
        rho = np.nan
    else:
        rho, Xass = density_newton_lim(bracket[0], bracket[1], x, temp_aux,
                                       P, Xass, saft)
        # rho = brentq(Psaft_obj, bracket[0], bracket[1],
        #              args=(x, temp_aux, saft, P))

    return rho, Xass


def density_newton(rho0, x, temp_aux, P, Xass0, saft):

    rho = 1.*rho0
    Psaft, dPsaft, Xass = saft.dP_drho_aux(x, rho, temp_aux, Xass0)
    for i in range(15):
        FO = Psaft - P
        dFO = dPsaft
        drho = FO/dFO
        rho -= drho
        if np.abs(drho) < 1e-6:
            break
        Psaft, dPsaft, Xass = saft.dP_drho_aux(x, rho, temp_aux, Xass)
    return rho, Xass
