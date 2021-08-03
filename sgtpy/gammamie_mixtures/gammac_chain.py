from __future__ import division, print_function, absolute_import
import numpy as np


phi70 = 10.
phi71 = 10.
phi72 = 0.57
phi73 = -6.7
phi74 = -8


# Equation (60) Paper 2014
def gammac(xhixm, alpha, tetha):

    gc = phi70*(-np.tanh(phi71 * (phi72-alpha)) + 1)
    gc *= xhixm*tetha*np.exp(phi73*xhixm + phi74*xhixm**2)

    return gc


def dgammac_dxhi00(xhixm, dxhixm_dxhi00, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte*np.exp(phi73*xhixm + phi74 * xhixm**2)

    dg = np.array([g, g])
    dg[0] *= xhixm
    dg[1] *= (1. + xhixm*(phi73+2*phi74*xhixm))
    dg[1] *= dxhixm_dxhi00

    return dg


def d2gammac_dxhi00(xhixm, dxhixm_dxhi00, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha)) + 1) * tetha
    g = cte*np.exp(phi73*xhixm + phi74*xhixm**2)

    dg = np.array([g, g, g])
    dg[0] *= xhixm

    dg[1] *= (1. + xhixm*(phi73 + 2*phi74*xhixm))
    dg[1] *= dxhixm_dxhi00

    aux1 = phi73**2 + 6*phi74+4*phi74*xhixm*(phi73 + phi74*xhixm)
    dg[2] *= (2*phi73 + xhixm * aux1) * dxhixm_dxhi00**2

    return dg


def dgammac_dx(xhixm, dxhixm_dx, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte*np.exp(phi73*xhixm + phi74*xhixm**2)
    gc = g * xhixm

    g *= 1. + xhixm*(phi73 + 2*phi74*xhixm)

    dg = np.multiply.outer(dxhixm_dx, g)
    return gc, dg


def dgammac_dxxhi(xhixm, dxhixm_dxhi00, dxhixm_dx, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte * np.exp(phi73*xhixm + phi74*xhixm**2)
    gc = np.array([g, g])
    gc[0] *= xhixm
    gc[1] *= (1. + xhixm*(phi73 + 2*phi74*xhixm))
    gc[1] *= dxhixm_dxhi00

    g *= 1. + xhixm*(phi73 + 2*phi74*xhixm)

    dg = np.multiply.outer(dxhixm_dx, g)
    return gc, dg
