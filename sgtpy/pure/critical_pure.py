import numpy as np
from scipy.optimize import root


def fobj_crit(inc, eos):

    T, rho = inc

    h = 1e-4

    rho1 = rho + h
    rho2 = rho + 2*h

    P, dP = eos.dP_drho(rho, T)
    P1, dP1 = eos.dP_drho(rho1, T)
    P2, dP2 = eos.dP_drho(rho2, T)

    d2P_drho = (-1.5*dP + 2*dP1 - 0.5*dP2)/h

    fo = np.array([-rho**2*dP, 2*rho**3*dP + rho**4*d2P_drho])
    fo /= rho**2

    return fo


def get_critical(eos, Tc0, rhoc0, method='hybr'):
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

    Returns
    -------
    Tc: float
        Critical temperature [K]
    Pc: float
        Critical pressure [Pa]
    rhoc: float
        Critical density [mol/m3]
    """
    inc0 = np.array([Tc0, rhoc0])
    sol = root(fobj_crit, inc0, method=method, args=eos)
    Tc, rhoc = sol.x
    Pc = eos.pressure(rhoc, Tc)
    return Tc, Pc, rhoc
