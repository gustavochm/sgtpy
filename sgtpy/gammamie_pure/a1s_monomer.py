from __future__ import division, print_function, absolute_import
from .monomer_aux import xhi_eff, dxhieff_dxhi00, d2xhieff_dxhi00
from .monomer_aux import d3xhieff_dxhi00


# Eq (19) Paper 2014
def a1s(xhi00, xhix_vec, xm, cictes, a1vdw):
    xhieff = xhi_eff(xhix_vec, cictes)
    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm
    return a1


def da1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw, dxhix_dxhi00):

    xhieff, dxhieff = dxhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00)

    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1

    xhieff2 = xhieff**2
    aux1 = a1vdw * xhi00
    aux2 = (5 - 2 * xhieff)

    ghs = (1 - xhieff/2) / xhieff_13
    a1 = aux1 * ghs * xm

    da1 = 2 - 3*xhieff + xhieff2
    da1 += xhi00 * dxhieff * aux2
    da1 *= a1vdw * xm
    da1 /= 2*xhieff_14

    return a1, da1


def d2a1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw, dxhix_dxhi00):

    out = d2xhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00)
    xhieff, dxhieff, d2xhieff = out

    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1
    xhieff_15 = xhieff_14*xhieff_1

    xhieff2 = xhieff**2
    aux1 = a1vdw * xhi00
    aux2 = (5 - 2 * xhieff)
    aux3 = 2 - 3*xhieff + xhieff2
    aux3 += xhi00 * aux2 * dxhieff
    aux3 *= -xhieff_1
    aux4 = -5 + 7*xhieff - 2*xhieff2
    aux5 = -3 + xhieff

    ghs = (1 - xhieff/2) / xhieff_13
    a1 = aux1 * ghs * xm

    da1 = 2 - 3*xhieff + xhieff2
    da1 += xhi00 * dxhieff * aux2
    da1 *= a1vdw * xm
    da1 /= 2*xhieff_14

    d2a1 = -2 * xhieff_1 * aux2 * dxhieff
    d2a1 += 6 * xhi00 * aux5 * dxhieff**2
    d2a1 += xhi00 * aux4 * d2xhieff
    d2a1 *= -a1vdw * xm
    d2a1 /= 2*xhieff_15

    return a1, da1, d2a1


def d3a1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw, dxhix_dxhi00):

    out = d3xhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00)
    xhieff, dxhieff, d2xhieff, d3xhieff = out

    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1
    xhieff_15 = xhieff_14*xhieff_1
    xhieff_16 = xhieff_15*xhieff_1

    xhieff2 = xhieff**2
    aux1 = a1vdw * xhi00
    aux2 = (5 - 2 * xhieff)
    aux3 = 2 - 3*xhieff + xhieff2
    aux3 += xhi00 * aux2 * dxhieff
    aux3 *= -xhieff_1
    aux4 = -5 + 7*xhieff - 2*xhieff2
    aux5 = -3 + xhieff

    ghs = (1 - xhieff/2) / xhieff_13
    a1 = aux1 * ghs * xm

    da1 = 2 - 3*xhieff + xhieff2
    da1 += xhi00 * dxhieff * aux2
    da1 *= a1vdw * xm
    da1 /= 2*xhieff_14

    d2a1 = -2 * xhieff_1 * aux2 * dxhieff
    d2a1 += 6 * xhi00 * aux5 * dxhieff**2
    d2a1 += xhi00 * aux4 * d2xhieff
    d2a1 *= -a1vdw * xm
    d2a1 /= 2*xhieff_15

    d3a1 = 18 * aux5 * -xhieff_1 * dxhieff**2
    d3a1 += 12 * xhi00 * (7 - 2 * xhieff) * dxhieff**3
    d3a1 += 18 * xhi00 * aux5 * -xhieff_1 * dxhieff * d2xhieff
    d3a1 += xhieff_1**2 * aux2 * (3*d2xhieff+xhi00*d3xhieff)
    d3a1 *= a1vdw * xm
    d3a1 /= 2*xhieff_16

    return a1, da1, d2a1, d3a1
