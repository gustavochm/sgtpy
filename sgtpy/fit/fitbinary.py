from __future__ import division, print_function, absolute_import
import numpy as np
from .fitmulticomponent import fobj_vle, fobj_lle, fobj_vlleb
from scipy.optimize import minimize, minimize_scalar
from ..saft import saftvrmie


def fobj_kij(kij, mix, datavle=None, datalle=None, datavlle=None,
             weights_vle=[1., 1.], weights_lle=[1., 1.],
             weights_vlle=[1., 1., 1., 1.]):

    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)
    eos = saftvrmie(mix)

    error = 0.
    if datavle is not None:
        error += fobj_vle(eos, *datavle, weights_vle=weights_vle)
    if datalle is not None:
        error += fobj_lle(eos, *datalle, weights_lle=weights_lle)
    if datavlle is not None:
        error += fobj_vlleb(eos, *datavlle, weights_vlleb=weights_vlle)
    return error


def fit_kij(kij_bounds, mix, datavle=None, datalle=None, datavlle=None,
            weights_vle=[1., 1.], weights_lle=[1., 1.],
            weights_vlle=[1., 1., 1., 1.], minimize_options={}):
    """
    fit_kij attemps to fit kij to VLE, LLE, VLLE

    Parameters
    ----------
    kij_bounds: tuple
        bounds for kij correction
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    weights_vle: list or array_like, optional
        weights_vle[0] = weight for Y composition error, default to 1.
        weights_vle[1] = weight for bubble pressure error, default to 1.
    weights_lle: list or array_like, optional
        weights_lle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_lle[1] = weight for W (liquid 2) composition error, default to 1.
    weights_vlle: list or array_like, optional
        weights_vlle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_vlle[1] = weight for W (liquid 2) composition error, default to 1.
        weights_vlle[2] = weight for Y (vapor) composition error, default to 1.
        weights_vlle[3] = weight for equilibrium pressure error, default to 1.
     minimize_options: dict
        Dictionary of any additional spefication for scipy minimize_scalar

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize_scalar(fobj_kij, kij_bounds, args=(mix, datavle, datalle,
                          datavlle, weights_vle, weights_lle, weights_vlle),
                          **minimize_options)
    return fit


def fobj_asso(x, mix, datavle=None, datalle=None, datavlle=None,
              weights_vle=[1., 1.], weights_lle=[1., 1.],
              weights_vlle=[1., 1., 1., 1.]):
    kij, lij = x
    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)
    Lij = np.array([[0, lij], [lij, 0]])
    mix.lij_saft(Lij)
    eos = saftvrmie(mix)

    error = 0.
    if datavle is not None:
        error += fobj_vle(eos, *datavle, weights_vle=weights_vle)
    if datalle is not None:
        error += fobj_lle(eos, *datalle, weights_lle=weights_lle)
    if datavlle is not None:
        error += fobj_vlleb(eos, *datavlle, weights_vlleb=weights_vlle)
    return error


def fit_asso(x0, mix, datavle=None, datalle=None, datavlle=None,
             weights_vle=[1., 1.], weights_lle=[1., 1.],
             weights_vlle=[1., 1., 1., 1.], minimize_options={}):
    """
    fit_asso attemps to fit kij and lij to VLE, LLE, VLLE

    Parameters
    ----------
    x0: array
        initial values for kij and lij
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    weights_vle: list or array_like, optional
        weights_vle[0] = weight for Y composition error, default to 1.
        weights_vle[1] = weight for bubble pressure error, default to 1.
    weights_lle: list or array_like, optional
        weights_lle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_lle[1] = weight for W (liquid 2) composition error, default to 1.
    weights_vlle: list or array_like, optional
        weights_vlle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_vlle[1] = weight for W (liquid 2) composition error, default to 1.
        weights_vlle[2] = weight for Y (vapor) composition error, default to 1.
        weights_vlle[3] = weight for equilibrium pressure error, default to 1.
     minimize_options: dict
        Dictionary of any additional spefication for scipy minimize

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize(fobj_asso, x0, args=(mix, datavle, datalle, datavlle,
                   weights_vle, weights_lle, weights_vlle),
                   **minimize_options)
    return fit


def fobj_cross(x, mix, assoc, datavle=None, datalle=None, datavlle=None,
               weights_vle=[1., 1.], weights_lle=[1., 1.],
               weights_vlle=[1., 1., 1., 1.]):
    kij, rc = x
    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)

    eos = saftvrmie(mix)
    eos.eABij[0, 1] = mix.eAB[assoc] / 2
    eos.eABij[1, 0] = mix.eAB[assoc] / 2
    eos.rcij[0, 1] = rc * 1e-10
    eos.rcij[1, 0] = rc * 1e-10

    error = 0.
    if datavle is not None:
        error += fobj_vle(eos, *datavle, weights_vle=weights_vle)
    if datalle is not None:
        error += fobj_lle(eos, *datalle, weights_lle=weights_lle)
    if datavlle is not None:
        error += fobj_vlleb(eos, *datavlle, weights_vlleb=weights_vlle)
    return error


def fit_cross(x0, mix, assoc, datavle=None, datalle=None, datavlle=None,
              weights_vle=[1., 1.], weights_lle=[1., 1.],
              weights_vlle=[1., 1., 1., 1.], minimize_options={}):
    """
    fit_cross attemps to fit kij and rcij to VLE, LLE, VLLE

    Parameters
    ----------
    x0 : array
        initial values for kij and rc
    mix: object
        binary mixture
    assoc : int
        index of associating component
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    weights_vle: list or array_like, optional
        weights_vle[0] = weight for Y composition error, default to 1.
        weights_vle[1] = weight for bubble pressure error, default to 1.
    weights_lle: list or array_like, optional
        weights_lle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_lle[1] = weight for W (liquid 2) composition error, default to 1.
    weights_vlle: list or array_like, optional
        weights_vlle[0] = weight for X (liquid 1) composition error, default to 1.
        weights_vlle[1] = weight for W (liquid 2) composition error, default to 1.
        weights_vlle[2] = weight for Y (vapor) composition error, default to 1.
        weights_vlle[3] = weight for equilibrium pressure error, default to 1.
    minimize_options: dict
        Dictionary of any additional spefication for scipy minimize

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize(fobj_cross, x0, args=(mix, assoc, datavle, datalle,
                   datavlle, weights_vle, weights_lle, weights_vlle),
                   **minimize_options)
    return fit
