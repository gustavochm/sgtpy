from __future__ import division, print_function, absolute_import
import numpy as np


def a1(xs, a1kl):
    a = np.matmul(np.matmul(a1kl, xs), xs)
    return a


def da1_dxhi00(xs, da1kl):
    da = np.matmul(np.matmul(da1kl, xs), xs)
    return da


def d2a1_dxhi00(xs, d2a1kl):
    d2a = np.matmul(np.matmul(d2a1kl, xs), xs)
    return d2a


def d3a1_dxhi00(xs, d3a1kl):
    d3a = np.matmul(np.matmul(d3a1kl, xs), xs)
    return d3a


def da1_dx(xs, dxs_dx, a1kl, da1kl_dx):

    a = np.matmul(np.matmul(a1kl, xs), xs)

    aux1 = xs * a1kl
    suma1 = 2*np.sum(dxs_dx@aux1, axis=1)

    da1x = np.matmul(np.matmul(da1kl_dx, xs), xs)
    da1x += suma1

    return a, da1x


def da1_dxxhi(xs, dxs_dx, da1kl, da1kl_dx):

    a = np.matmul(np.matmul(da1kl, xs), xs)

    aux1 = xs * da1kl[0]
    suma1 = 2*np.sum(dxs_dx@aux1, axis=1)

    da1x = np.matmul(np.matmul(da1kl_dx, xs), xs)
    da1x += suma1

    return a, da1x
