from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from .EquilibriumResult import EquilibriumResult
from ..constants import kb, Na


def fobj_crit(inc, eos):

    T, rho = inc

    h = 1e-4

    rho1 = rho + h
    rho2 = rho + 2*h

    P, dP = eos.dP_drho(rho, T)
    P1, dP1 = eos.dP_drho(rho1, T)
    P2, dP2 = eos.dP_drho(rho2, T)

    d2P_drho = (-1.5*dP + 2*dP1 - 0.5*dP2)/h

    # fo = np.array([-rho**2*dP, 2*rho**3*dP + rho**4*d2P_drho])
    # fo /= rho**2

    # fo = np.array([-dP, 2*dP + rho*d2P_drho])
    fo = np.array([-dP, 2*dP/rho + d2P_drho])

    return fo


def initial_guess_criticalpure(eos, n=50):
    """
    This functions computes inicial guesses of the critcal temperature in K
    and the critical density in mol/m^3.

    Function provided by Esteban Cea Klapp.

    Parameters
    ----------

    eos : Object
        saftvrmie pure object
    n: int
        lenght of linspace to study

    Returns
    -------
    Tc: float
        Critical temperature [K]
    rhoc: float
        Critical density [mol/m3]
    """

    romin = 0.7405
    romin *= 6/(Na*eos.xs_m*np.pi*eos.sigma_ii3)

    roc0 = romin/5
    Tc0 = eos.xs_m*eos.eps_ii/kb
    dTc0 = Tc0*0.3

    ro0 = 0.25*roc0
    rof = 2*roc0
    # Step 1. Check if tc0 is subcritical or supercritical
    ro = np.linspace(ro0, rof, n)
    dro = ro[1] - ro[0]
    for i in range(n):
        P, dP = eos.dP_drho(ro[i], Tc0)
        # If dP_drho < 0 in any interval, tc0 is subcritical (tc0 < tc)
        if dP < 0:
            # In this case you need to increase tc0
            dTc0 = Tc0*0.5
            T0 = 'sub'
            ro0 = ro[i] - dro
            break
        else:
            dTc0 = -Tc0*0.1
            T0 = 'sup'

    # Step 2. Find the temperature transition subcritical to supercritical
    loop = True
    ro = np.linspace(ro0, rof, n)
    dro = ro[1] - ro[0]
    k = 0
    T = [Tc0]  # List to store the studied temperatures
    if T0 == 'sub':
        while loop and k < 10:
            Tc0 += dTc0
            loop = False
            k += 1
            T.append(Tc0)
            for i in range(n):
                P, dP = eos.dP_drho(ro[i], Tc0)
                if dP < 0:
                    loop = True
                    ro0 = ro[i]
                    break
                loop = False
                Tsub = T[-2]
                Tsup = T[-1]
    else:
        while loop and k < 10:
            Tc0 += dTc0
            k += 1
            T.append(Tc0)
            for i in range(n):
                P, dP = eos.dP_drho(ro[i], Tc0)
                if dP < 0:
                    loop = False
                    ro0 = ro[i]
                    Tsup = T[-2]
                    Tsub = T[-1]
                    break

    # Step 3. Find rho_min and rho_max at the subcritical temperature

    ro = np.linspace(ro0, rof, n)
    dro = ro[1] - ro[0]
    ro_int = []
    flag = 0
    for i in range(n):
        P, dP = eos.dP_drho(ro[i], Tsub)
        if dP < 0:
            ro_int.append(ro[i])
            flag = 1
        else:
            if flag == 1:
                break

    # computing the initial guesses
    romin = min(ro_int)
    romax = max(ro_int)
    roc0 = (romin + romax)/2
    Tc0 = (Tsub + Tsup)/2

    return Tc0, roc0


def get_critical(eos, Tc0=None, rhoc0=None, method='hybr', full_output=False):
    """
    get_critical(eos, Tc0, rhoc0, method)

    Method that solves the critical coordinate of the fluid.

    Parameters
    ----------

    eos : Object
        saftvrmie pure object
    Tc0 : float
        initial guess for critical temperature [K]
    rhoc : float
        initial guess for critical density [mol/m^3]
    method : string, optional
        SciPy; root method to solve critical coordinate
    full_output: bool, optional
        whether to outputs or not all the calculation info.

    Returns
    -------
    Tc: float
        Critical temperature [K]
    Pc: float
        Critical pressure [Pa]
    rhoc: float
        Critical density [mol/m3]
    """

    if Tc0 is None and rhoc0 is None:
        Tc0, rhoc0 = initial_guess_criticalpure(eos, n=30)

    inc0 = np.array([Tc0, rhoc0])
    sol = root(fobj_crit, inc0, method=method, args=eos)
    Tc, rhoc = sol.x
    Pc = eos.pressure(rhoc, Tc)
    # print(sol)
    if full_output:
        dict = {'Tc': Tc, 'Pc': Pc, 'rhoc': rhoc, 'error': sol.fun,
                'nfev:': sol.nfev, 'message': sol.message,
                'success': sol.success}
        out = EquilibriumResult(dict)
    else:
        out = Tc, Pc, rhoc
    return out
