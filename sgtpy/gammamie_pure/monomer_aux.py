from __future__ import division, print_function, absolute_import
import numpy as np


# Eq (24) Paper 2014
def J_lam(x0, lam):
    lam3 = 3. - lam
    lam4 = 4. - lam
    J = (lam3*x0**lam4 - lam4*x0**lam3 + 1.)/(lam3 * lam4)
    return J


# Eq (23) Paper 2014
def I_lam(x0, lam):
    lam3 = 3. - lam
    I = (x0**lam3-1.) / lam3
    return I


# Equation (26) Paper 2014
def xhi_eff(xhix_vec, cictes):
    # xhix2 = xhix**2
    # xhix3 = xhix2*xhix
    # xhix4 = xhix3*xhix
    # xhix_vec = np.array([xhix, xhix2, xhix3, xhix4])
    xhieff = np.matmul(cictes, xhix_vec)
    return xhieff


def dxhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00):
    # xhix2 = xhix**2
    # xhix3 = xhix2*xhix
    # xhix4 = xhix3*xhix
    # xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
    #                      [1., 2 * xhix, 3*xhix2, 4*xhix3]])

    xhieff = np.matmul(cictes, xhix_vec[0])
    dxhieff = np.matmul(cictes, xhix_vec[1])
    dxhieff *= dxhix_dxhi00
    return xhieff, dxhieff


def d2xhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00):
    # xhix2 = xhix**2
    # xhix3 = xhix2*xhix
    # xhix4 = xhix3*xhix
    # xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
    #                      [1., 2 * xhix, 3*xhix2, 4*xhix3],
    #                      [0., 2., 6*xhix, 12*xhix2]])

    xhieff = np.matmul(cictes, xhix_vec[0])
    dxhieff = np.matmul(cictes, xhix_vec[1])
    dxhieff *= dxhix_dxhi00
    d2xhieff = np.matmul(cictes, xhix_vec[2])
    d2xhieff *= dxhix_dxhi00**2
    return xhieff, dxhieff, d2xhieff


def d3xhieff_dxhi00(xhix_vec, cictes, dxhix_dxhi00):
    # xhix2 = xhix**2
    # xhix3 = xhix2*xhix
    # xhix4 = xhix3*xhix
    # xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
    #                      [1., 2 * xhix, 3.*xhix2, 4.*xhix3],
    #                      [0., 2., 6*xhix, 12.*xhix2],
    #                     [0., 0., 6., 24.*xhix]])
    dxhix_dxhi00_2 = dxhix_dxhi00**2
    dxhix_dxhi00_3 = dxhix_dxhi00_2*dxhix_dxhi00

    xhieff = np.matmul(cictes, xhix_vec[0])
    dxhieff = np.matmul(cictes, xhix_vec[1])
    dxhieff *= dxhix_dxhi00
    d2xhieff = np.matmul(cictes, xhix_vec[2])
    d2xhieff *= dxhix_dxhi00_2
    d3xhieff = np.matmul(cictes, xhix_vec[3])
    d3xhieff *= dxhix_dxhi00_3
    return xhieff, dxhieff, d2xhieff, d3xhieff


# Equation (32) Paper 2014
def Xi(xhixm, f1, f2, f3):
    x = f1*xhixm + f2*xhixm**5 + f3*xhixm**8
    return x


def dXi_dxhi00(xhixm, dxhixm_dxhi00, f1, f2, f3):
    xhixm2 = xhixm**2
    xhixm4 = xhixm2**2
    xhixm5 = xhixm4*xhixm
    xhixm7 = xhixm5*xhixm2
    xhixm8 = xhixm7*xhixm

    x = f1*xhixm + f2*xhixm5 + f3*xhixm8

    dx = f1 + 5*f2*xhixm4 + 8*f3*xhixm7
    dx *= dxhixm_dxhi00
    return x, dx


def d2Xi_dxhi00(xhixm, dxhixm_dxhi00, f1, f2, f3):
    xhixm2 = xhixm**2
    xhixm3 = xhixm2*xhixm
    xhixm4 = xhixm3*xhixm
    xhixm5 = xhixm4*xhixm
    xhixm6 = xhixm5*xhixm
    xhixm7 = xhixm6*xhixm
    xhixm8 = xhixm7*xhixm

    x = f1*xhixm + f2*xhixm5 + f3*xhixm8

    dx = f1 + 5*f2*xhixm4 + 8*f3*xhixm7
    dx *= dxhixm_dxhi00

    d2x = 20*f2*xhixm3 + 56*f3*xhixm6
    d2x *= dxhixm_dxhi00**2
    return x, dx, d2x


def d3Xi_dxhi00(xhixm, dxhixm_dxhi00, f1, f2, f3):
    x = 60*f2*xhixm**2 + 56*6*f3*xhixm**5
    x *= dxhixm_dxhi00**3
    return x


# Equation (31) Paper 2014
def kHS(xhix):
    k = (1-xhix)**4
    k /= 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4
    return k


def dkHS_dxhi00(xhix, dxhix_dxhi00):
    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix3 * xhix

    den = 1 + 4*xhix + 4*xhix2 - 4*xhix3 + xhix4
    den2 = den**2

    xhix_1 = -1 + xhix
    xhix_12 = xhix_1**2
    xhix_13 = xhix_12*xhix_1
    xhix_14 = xhix_13*xhix_1

    k = xhix_14/den

    dk = -2 - 5*xhix + xhix2
    dk *= -4 * xhix_13 * dxhix_dxhi00
    dk /= den2
    return k, dk


def d2kHS_dxhi00(xhix, dxhix_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix3 * xhix
    xhix5 = xhix4 * xhix
    xhix6 = xhix5 * xhix

    den = 1 + 4*xhix + 4*xhix2 - 4*xhix3 + xhix4
    den2 = den**2
    den3 = den2*den

    xhix_1 = -1 + xhix
    xhix_12 = xhix_1**2
    xhix_13 = xhix_12*xhix_1
    xhix_14 = xhix_13*xhix_1

    k = xhix_14/den

    dk = -2 - 5*xhix + xhix2
    dk *= -4 * xhix_13 * dxhix_dxhi00
    dk /= den2

    d2k = 17 + 82 * xhix + 39 * xhix2 - 80 * xhix3
    d2k += 77 * xhix4 - 30 * xhix5 + 3*xhix6
    d2k *= 4 * xhix_12 * dxhix_dxhi00**2
    d2k /= den3
    return k, dk, d2k


def d3kHS_dxhi00(xhix, dxhix_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix3 * xhix
    xhix5 = xhix4 * xhix
    xhix6 = xhix5 * xhix
    xhix7 = xhix6 * xhix
    xhix8 = xhix7 * xhix
    xhix9 = xhix8 * xhix

    den = 1 + 4*xhix + 4*xhix2 - 4*xhix3 + xhix4
    den2 = den**2
    den3 = den2*den
    den4 = den3*den

    xhix_1 = -1 + xhix
    xhix_12 = xhix_1**2
    xhix_13 = xhix_12*xhix_1
    xhix_14 = xhix_13*xhix_1

    k = xhix_14/den

    dk = -2 - 5*xhix + xhix2
    dk *= -4 * xhix_13 * dxhix_dxhi00
    dk /= den2

    d2k = 17 + 82 * xhix + 39 * xhix2 - 80 * xhix3
    d2k += 77 * xhix4 - 30 * xhix5 + 3*xhix6
    d2k *= 4 * xhix_12 * dxhix_dxhi00**2
    d2k /= den3

    d3k = -97 - 109 * xhix + 238 * xhix2 - 352 * xhix4
    d3k += 372 * xhix5 - 210*xhix6 + 77*xhix7 - 15*xhix8 + xhix9
    d3k *= xhix
    d3k -= 13.
    d3k *= 12.*(-4 * xhix_1 * dxhix_dxhi00**3)
    d3k /= den4

    return k, dk, d2k, d3k
