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


def dahs_dx(xhi, dxhi_dx):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0x, dxhi1x, dxhi2x, dxhi3x = dxhi_dx

    xhi3_1 = 1-xhi3
    xhi3_12 = xhi3_1**2
    xhi3_13 = xhi3_12*xhi3_1
    log3 = np.log(xhi3_1)

    xhi02 = xhi0**2
    xhi22 = xhi2**2
    xhi23 = xhi22*xhi2

    xhi32 = xhi3**2
    xhi33 = xhi32*xhi3

    a = (xhi23/xhi32 - xhi0) * log3
    a += 3 * xhi1 * xhi2 / xhi3_1
    a += xhi23 / xhi3 / xhi3_12
    a /= xhi0

    dax1 = -3*xhi0*xhi22 * (log3 + xhi3/xhi3_12)
    dax1 *= dxhi2x / xhi32

    dax2 = (xhi2 * dxhi0x - xhi0 * dxhi2x) * -xhi3_1
    dax2 += xhi0 * xhi2 * dxhi3x
    dax2 *= 3. * xhi1/xhi3_1

    dax2 += 3 * xhi0 * xhi2 * dxhi1x
    dax2 += xhi02 * dxhi3x
    dax2 /= -xhi3_1

    dax3 = log3 * dxhi0x - xhi0 * (-2 + xhi3)*dxhi3x/xhi3_12
    dax3 *= xhi3
    dax3 += 2*log3*xhi0*dxhi3x
    dax3 += xhi32 * (-xhi3_1*dxhi0x + 2*xhi0*dxhi3x)/-xhi3_13
    dax3 *= (xhi23/xhi33)

    dax = dax1 + dax2 + dax3
    dax /= -xhi02

    return a, dax


def dahs_dxxhi(xhi, dxhi_dxhi00, dxhi_dx):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    dxhi0x, dxhi1x, dxhi2x, dxhi3x = dxhi_dx

    xhi3_1 = 1-xhi3
    xhi3_12 = xhi3_1**2
    xhi3_13 = xhi3_12*xhi3_1
    log3 = np.log(xhi3_1)

    xhi02 = xhi0**2
    xhi22 = xhi2**2
    xhi23 = xhi22*xhi2

    xhi32 = xhi3**2
    xhi33 = xhi32*xhi3

    a = (xhi23/xhi32 - xhi0) * log3
    a += 3 * xhi1 * xhi2 / xhi3_1
    a += xhi23 / xhi3 / xhi3_12
    a /= xhi0

    da = xhi2 * dxhi2**2 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * -xhi3_1
    da -= dxhi0 * dxhi3 * xhi3_12
    da /= -xhi3_13 * dxhi0

    dax1 = -3*xhi0*xhi22 * (log3 + xhi3/xhi3_12)
    dax1 *= dxhi2x / xhi32

    dax2 = (xhi2 * dxhi0x - xhi0 * dxhi2x) * -xhi3_1
    dax2 += xhi0 * xhi2 * dxhi3x
    dax2 *= 3. * xhi1/xhi3_1

    dax2 += 3 * xhi0 * xhi2 * dxhi1x
    dax2 += xhi02 * dxhi3x
    dax2 /= -xhi3_1

    dax3 = log3 * dxhi0x - xhi0 * (-2 + xhi3)*dxhi3x/xhi3_12
    dax3 *= xhi3
    dax3 += 2*log3*xhi0*dxhi3x
    dax3 += xhi32 * (-xhi3_1*dxhi0x + 2*xhi0*dxhi3x)/-xhi3_13
    dax3 *= (xhi23/xhi33)

    dax = dax1 + dax2 + dax3
    dax /= -xhi02

    return np.array([a, da]), dax
