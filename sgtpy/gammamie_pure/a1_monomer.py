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
