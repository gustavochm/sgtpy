import numpy as np


phi70 = 10.
phi71 = 10.
phi72 = 0.57
phi73 = -6.7
phi74 = -8


# Equation A37
def gammac(xhixm, alpha, tetha):

    gc = phi70*(-np.tanh(phi71 * (phi72-alpha)) + 1)
    gc *= xhixm*tetha*np.exp(phi73*xhixm + phi74*xhixm**2)

    return gc


def dgammac_dxhi00(xhim, dxhim_dxhi00, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte*np.exp(phi73*xhim + phi74 * xhim**2)

    dg = np.array([g, g])
    dg[0] *= xhim
    dg[1] *= (1. + xhim*(phi73+2*phi74*xhim))
    dg[1] *= dxhim_dxhi00

    return dg


def d2gammac_dxhi00(xhim, dxhim_dxhi00, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha)) + 1) * tetha
    g = cte*np.exp(phi73*xhim + phi74*xhim**2)

    dg = np.array([g, g, g])
    dg[0] *= xhim

    dg[1] *= (1. + xhim*(phi73 + 2*phi74*xhim))
    dg[1] *= dxhim_dxhi00

    aux1 = phi73**2 + 6*phi74+4*phi74*xhim*(phi73 + phi74*xhim)
    dg[2] *= (2*phi73 + xhim * aux1) * dxhim_dxhi00**2

    return dg


def dgammac_dx(xhim, dxhim_dx, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte*np.exp(phi73*xhim + phi74*xhim**2)
    gc = g * xhim

    g *= 1. + xhim*(phi73 + 2*phi74*xhim)

    dg = np.multiply.outer(dxhim_dx, g)
    return gc, dg


def dgammac_dxxhi(xhim, dxhim_dxhi00, dxhim_dx, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte * np.exp(phi73*xhim + phi74*xhim**2)
    gc = np.array([g, g])
    gc[0] *= xhim
    gc[1] *= (1. + xhim*(phi73 + 2*phi74*xhim))
    gc[1] *= dxhim_dxhi00

    g *= 1. + xhim*(phi73 + 2*phi74*xhim)

    dg = np.multiply.outer(dxhim_dx, g)
    return gc, dg
