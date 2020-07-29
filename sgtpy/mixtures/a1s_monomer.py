import numpy as np
from .monomer_aux import xhi_eff, dxhieff_dxhi00, d2xhieff_dxhi00
from .monomer_aux import d3xhieff_dxhi00, dxhieff_dx, dxhieff_dx_dxhi00


# Equation A16
def a1s(xhi00, xhix, xm, cictes, a1vdw):
    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix, cictes)
    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm
    return a1


def da1s_dxhi00(xhi00, xhix, xm, cictes, a1vdw, dxhix_dxhi00):
    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix, cictes)
    dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)

    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm

    da1 = 2 - 3*xhieff + xhieff**2
    da1 += xhi00 * dxhieff * (5 - 2 * xhieff)
    da1 *= a1vdw * xm
    da1 /= 2*(- 1 + xhieff)**4

    return a1, da1


def d2a1s_dxhi00(xhi00, xhix, xm, cictes, a1vdw, dxhix_dxhi00):
    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix, cictes)
    dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    d2xhieff = d2xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)

    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm

    da1 = 2 - 3*xhieff + xhieff**2
    da1 += xhi00 * dxhieff * (5 - 2 * xhieff)
    da1 *= a1vdw * xm
    da1 /= 2*(- 1 + xhieff)**4

    d2a1 = -2 * (-1 + xhieff) * (-5 + 2*xhieff) * dxhieff
    d2a1 += 6 * xhi00 * (-3 + xhieff) * dxhieff**2
    d2a1 += xhi00 * (-5 + (7 - 2 * xhieff) * xhieff) * d2xhieff
    d2a1 *= a1vdw * xm
    d2a1 /= 2*(- 1 + xhieff)**5

    return a1, da1, d2a1


def d3a1s_dxhi00(xhi00, xhix, xm, cictes, a1vdw, dxhix_dxhi00):

    xhieff = xhi_eff(xhix, cictes)
    dxhieff = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    d2xhieff = d2xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    d3xhieff = d3xhieff_dxhi00(xhix, cictes, dxhix_dxhi00)

    ghs = (1 - xhieff/2) / (1 - xhieff)**3
    a1 = a1vdw * ghs * xhi00 * xm

    da1 = 2 - 3*xhieff + xhieff**2
    da1 += xhi00 * dxhieff * (5 - 2 * xhieff)
    da1 *= a1vdw * xm
    da1 /= 2*(- 1 + xhieff)**4

    d2a1 = -2 * (-1 + xhieff) * (-5 + 2*xhieff) * dxhieff
    d2a1 += 6 * xhi00 * (-3 + xhieff) * dxhieff**2
    d2a1 += xhi00 * (-5 + (7 - 2 * xhieff) * xhieff) * d2xhieff
    d2a1 *= a1vdw * xm
    d2a1 /= 2*(- 1 + xhieff)**5

    d3a1 = 18 * (-3 + xhieff) * (-1 + xhieff) * dxhieff**2
    d3a1 += 12 * xhi00 * (7 - 2 * xhieff) * dxhieff**3
    d3a1 += 18 * xhi00 * (-3 + xhieff) * (-1 + xhieff) * dxhieff * d2xhieff
    d3a1 -= (-1 + xhieff)**2*(-5+2*xhieff)*(3*d2xhieff+xhi00*d3xhieff)

    d3a1 *= a1vdw * xm
    d3a1 /= 2*(- 1 + xhieff)**6

    return a1, da1, d2a1, d3a1


def da1s_dx(xhi00, xhix, xm, ms, cictes, a1vdw, dxhix_dx):
    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix, cictes)
    dxhieff = dxhieff_dx(xhix, cictes, dxhix_dx)

    da1 = np.multiply.outer(ms, (-2 + xhieff) * (-1 + xhieff))
    da1 += (5 - 2 * xhieff) * xm * dxhieff
    da1 *= a1vdw * xhi00
    da1 /= 2*(- 1 + xhieff)**4
    return da1


def da1s_dx_dxhi00(xhi00, xhix, xm, ms, cictes, a1vdw,
                   dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00):

    # a1s calculation Eq 39
    xhieff = xhi_eff(xhix, cictes)
    dxhieffdx = dxhieff_dx(xhix, cictes, dxhix_dx)
    dxhieffdxhi00 = dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00)
    dxhieffdxdxhi00 = dxhieff_dx_dxhi00(xhix, cictes, dxhix_dxhi00,
                                        dxhix_dx, dxhix_dx_dxhi00)

    aux1 = 2 - 3*xhieff + xhieff**2
    aux1 += xhi00 * (5 - 2 * xhieff) * dxhieffdxhi00
    aux1 *= (-1 + xhieff)

    da1 = dxhieffdx
    da1 *= -5 + 7*xhieff - 2*xhieff**2 + 6*xhi00 * (-3 + xhieff)*dxhieffdxhi00
    da1 += xhi00 * dxhieffdxdxhi00 * (-5 + xhieff * (7-2*xhieff))
    da1 *= xm
    da1 += np.multiply.outer(ms, aux1)
    da1 *= a1vdw
    da1 /= 2*(- 1 + xhieff)**5
    return da1
