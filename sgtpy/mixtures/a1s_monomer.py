import numpy as np
from .monomer_aux import xhi_eff, dxhieff_dxhi00, d2xhieff_dxhi00
from .monomer_aux import d3xhieff_dxhi00
from .monomer_aux import dxhieff_dx_dxhi00_dxxhi, dxhieff_dx_d2xhi00_dxxhi


# Equation A16
def a1s(xhi00, xhix_vec, xm, cictes, a1vdw):
    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix_vec, cictes)
    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm
    return a1


def da1s_dxhi00(xhi00, xhix_vec, xm, cctesij, a1vdwij, dxhix_dxhi00):
    # a1s calculation Eq 39
    # xhieff = xhi_eff(xhix, cictes)
    # dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    xhieff, dxhieff = dxhieff_dxhi00(xhix_vec, cctesij, dxhix_dxhi00)
    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1

    xhieff2 = xhieff**2
    aux1 = a1vdwij * xhi00
    aux2 = (5 - 2 * xhieff)

    ghs = (1 - xhieff/2) / xhieff_13
    a1 = aux1 * ghs * xm

    da1 = 2 - 3*xhieff + xhieff2
    da1 += xhi00 * dxhieff * aux2
    da1 *= a1vdwij * xm
    da1 /= 2*xhieff_14

    return a1, da1


def d2a1s_dxhi00(xhi00, xhix_vec, xm, cctesij, a1vdwij, dxhix_dxhi00):
    # a1s calculation Eq 39
    # xhieff = xhi_eff(xhix, cictes)
    # dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    # d2xhieff = d2xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)

    out = d2xhieff_dxhi00(xhix_vec, cctesij, dxhix_dxhi00)
    xhieff, dxhieff, d2xhieff = out

    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1
    xhieff_15 = xhieff_14*xhieff_1

    xhieff2 = xhieff**2
    aux0 = a1vdwij * xm
    aux1 = a1vdwij * xhi00
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
    da1 *= aux0
    da1 /= 2*xhieff_14

    d2a1 = -2 * xhieff_1 * aux2 * dxhieff
    d2a1 += 6 * xhi00 * aux5 * dxhieff**2
    d2a1 += xhi00 * aux4 * d2xhieff
    d2a1 *= -aux0
    d2a1 /= 2*xhieff_15

    return a1, da1, d2a1


def d3a1s_dxhi00(xhi00, xhix_vec, xm, cctesij, a1vdwij, dxhix_dxhi00):

    # xhieff = xhi_eff(xhix, cictes)
    # dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    # d2xhieff = d2xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    # d3xhieff = d3xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)

    out = d3xhieff_dxhi00(xhix_vec, cctesij, dxhix_dxhi00)
    xhieff, dxhieff, d2xhieff, d3xhieff = out

    xhieff_1 = 1 - xhieff
    xhieff_13 = xhieff_1**3
    xhieff_14 = xhieff_13*xhieff_1
    xhieff_15 = xhieff_14*xhieff_1
    xhieff_16 = xhieff_15*xhieff_1

    xhieff2 = xhieff**2
    aux0 = a1vdwij * xm
    aux1 = a1vdwij * xhi00
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
    da1 *= aux0
    da1 /= 2*xhieff_14

    d2a1 = -2 * xhieff_1 * aux2 * dxhieff
    d2a1 += 6 * xhi00 * aux5 * dxhieff**2
    d2a1 += xhi00 * aux4 * d2xhieff
    d2a1 *= -aux0
    d2a1 /= 2*xhieff_15

    d3a1 = 18 * aux5 * -xhieff_1 * dxhieff**2
    d3a1 += 12 * xhi00 * (7 - 2 * xhieff) * dxhieff**3
    d3a1 += 18 * xhi00 * aux5 * -xhieff_1 * dxhieff * d2xhieff
    d3a1 += xhieff_1**2 * aux2 * (3*d2xhieff+xhi00*d3xhieff)
    d3a1 *= aux0
    d3a1 /= 2*xhieff_16

    return a1, da1, d2a1, d3a1


def da1s_dx_dxhi00_dxxhi(xhi00, xhix_vec, xm, ms, cictes, a1vdw,
                         dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00):

    out = dxhieff_dx_dxhi00_dxxhi(xhix_vec, cictes, dxhix_dxhi00, dxhix_dx,
                                  dxhix_dx_dxhi00)
    xhieff, dxhieff, dxhieff_dx, dxhieff_dxxhi = out

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

    ghs = (1 - xhieff/2) / xhieff_13
    a1 = aux1 * ghs * xm

    da1 = 2 - 3*xhieff + xhieff2
    da1 += xhi00 * dxhieff * aux2
    da1 *= a1vdw * xm
    da1 /= 2*xhieff_14

    da1x = np.multiply.outer(ms, (2 - xhieff) * xhieff_1)
    da1x += aux2 * xm * dxhieff_dx
    da1x *= aux1
    da1x /= 2*xhieff_14

    da1xxhi = dxhieff_dx
    da1xxhi *= aux4 + 6*xhi00 * (-3 + xhieff)*dxhieff
    da1xxhi += xhi00 * dxhieff_dxxhi * aux4
    da1xxhi *= xm
    da1xxhi += np.multiply.outer(ms, aux3)
    da1xxhi *= a1vdw
    da1xxhi /= -2*xhieff_15
    return a1, da1, da1x, da1xxhi


def da1s_dx_d2xhi00_dxxhi(xhi00, xhix_vec, xm, ms, cictes, a1vdw,
                          dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00):

    out = dxhieff_dx_d2xhi00_dxxhi(xhix_vec, cictes, dxhix_dxhi00, dxhix_dx,
                                   dxhix_dx_dxhi00)
    xhieff, dxhieff, d2xhieff, dxhieff_dx, dxhieff_dxxhi = out

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

    da1x = np.multiply.outer(ms, (2 - xhieff) * xhieff_1)
    da1x += aux2 * xm * dxhieff_dx
    da1x *= aux1
    da1x /= 2*xhieff_14

    da1xxhi = dxhieff_dx
    da1xxhi *= aux4 + 6*xhi00 * aux5 * dxhieff
    da1xxhi += xhi00 * dxhieff_dxxhi * aux4
    da1xxhi *= xm
    da1xxhi += np.multiply.outer(ms, aux3)
    da1xxhi *= a1vdw
    da1xxhi /= -2*xhieff_15

    return a1, da1, d2a1, da1x, da1xxhi
