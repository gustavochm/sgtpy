import numpy as np
from .fitmulticomponent import fobj_elv, fobj_ell, fobj_hazb
from scipy.optimize import minimize, minimize_scalar
from ..saft import saftvrmie


def fobj_kij(kij, mix, datavle=None, datalle=None, datavlle=None):

    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)
    eos = saftvrmie(mix)

    error = 0.
    if datavle is not None:
        error += fobj_elv(eos, *datavle)
    if datalle is not None:
        error += fobj_ell(eos, *datalle)
    if datavlle is not None:
        error += fobj_hazb(eos, *datavlle)
    return error


def fit_kij(kij_bounds, eos, mix, datavle=None, datalle=None, datavlle=None):
    """
    fit_kij: attemps to fit kij to LVE, LLE, LLVE

    Parameters
    ----------
    kij_bounds : tuple
        bounds for kij correction
    eos : function
        cubic eos to fit kij for qmr mixrule
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize_scalar(fobj_kij, kij_bounds,
                          args=(mix, datavle, datalle, datavlle))
    return fit


def fobj_asso(x, mix, datavle=None, datalle=None, datavlle=None):
    kij, lij = x
    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)
    Lij = np.array([[0, lij], [lij, 0]])
    mix.lij_saft(Lij)
    eos = saftvrmie(mix)

    error = 0.
    if datavle is not None:
        error += fobj_elv(eos, *datavle)
    if datalle is not None:
        error += fobj_ell(eos, *datalle)
    if datavlle is not None:
        error += fobj_hazb(eos, *datavlle)
    return error


def fit_asso(x0, mix, datavle=None, datalle=None, datavlle=None):
    """
    fit_asso: attemps to fit kij to LVE, LLE, LLVE

    Parameters
    ----------
    x0 : array
        initial values for kij and lij
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize(fobj_asso, x0, args=(mix, datavle, datalle, datavlle))
    return fit
