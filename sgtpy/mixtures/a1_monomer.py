import numpy as np


def a1(xs, a1ij):
    a = np.matmul(np.matmul(a1ij, xs), xs)
    return a


def da1_dxhi00(xs, da1ij):
    da = np.matmul(np.matmul(da1ij, xs), xs)
    return da


def d2a1_dxhi00(xs, d2a1ij):
    d2a = np.matmul(np.matmul(d2a1ij, xs), xs)
    return d2a


def d3a1_dxhi00(xs, d3a1ij):
    d3a = np.matmul(np.matmul(d3a1ij, xs), xs)
    return d3a


def da1_dx(xs, dxs_dx, a1ij, da1ij_dx):

    a = np.matmul(np.matmul(a1ij, xs), xs)

    aux1 = xs * a1ij
    suma1 = 2*np.sum(dxs_dx@aux1, axis=1)

    da1x = np.matmul(np.matmul(da1ij_dx, xs), xs)
    da1x += suma1

    return a, da1x


def da1_dxxhi(xs, dxs_dx, a1ij, da1ij_dx):

    a = np.matmul(np.matmul(a1ij, xs), xs)

    aux1 = xs * a1ij[0]
    suma1 = 2*np.sum(dxs_dx@aux1, axis=1)

    da1x = np.matmul(np.matmul(da1ij_dx, xs), xs)
    da1x += suma1

    return a, da1x


def da1ij_dx_dxhi00(xs, da1ij_dx_dxhi00):
    da = np.matmul(np.matmul(da1ij_dx_dxhi00, xs), xs)
    return da
