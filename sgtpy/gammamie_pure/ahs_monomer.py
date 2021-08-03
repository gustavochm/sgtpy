from __future__ import division, print_function, absolute_import
import numpy as np


# Hard sphere Eq A6
def ahs(xhi):
    xhi0, xhi1, xhi2, xhi3 = xhi
    xhi3_1 = 1-xhi3
    xhi23 = xhi2**3

    a = (xhi23/xhi3**2 - xhi0) * np.log(xhi3_1)
    a += 3 * xhi1 * xhi2 / xhi3_1
    a += xhi23 / xhi3 / xhi3_1**2
    a /= xhi0
    return a


def dahs_dxhi00(xhi, dxhi_dxhi00):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    xhi3_1 = 1-xhi3
    xhi3_12 = xhi3_1**2
    xhi3_13 = xhi3_12*xhi3_1
    xhi23 = xhi2**3

    a = (xhi23/xhi3**2 - xhi0) * np.log(xhi3_1)
    a += 3 * xhi1 * xhi2 / xhi3_1
    a += xhi23 / xhi3 / xhi3_12
    a /= xhi0

    da = xhi2 * dxhi2**2 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * -xhi3_1
    da -= dxhi0 * dxhi3 * xhi3_12
    da /= -xhi3_13 * dxhi0
    return np.array([a, da])


def d2ahs_dxhi00(xhi, dxhi_dxhi00):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    xhi3_1 = 1-xhi3
    xhi3_12 = xhi3_1**2
    xhi3_13 = xhi3_12*xhi3_1
    xhi3_14 = xhi3_13*xhi3_1
    xhi23 = xhi2**3
    dxhi22 = dxhi2**2
    dxhi23 = dxhi22*dxhi2

    a = (xhi23/xhi3**2 - xhi0) * np.log(xhi3_1)
    a += 3 * xhi1 * xhi2 / xhi3_1
    a += xhi23 / xhi3 / xhi3_12
    a /= xhi0

    da = xhi2 * dxhi22 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * -xhi3_1
    da -= dxhi0 * dxhi3 * xhi3_12
    da /= -xhi3_13 * dxhi0

    d2a = -6 * dxhi1 * dxhi2 * dxhi3 * -xhi3_1
    d2a += dxhi0 * dxhi3**2 * xhi3_12
    d2a += dxhi23 * (3 + xhi3 * (4 - xhi3))
    d2a /= xhi3_14 * dxhi0
    return np.array([a, da, d2a])
