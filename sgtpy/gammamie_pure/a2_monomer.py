from __future__ import division, print_function, absolute_import
import numpy as np
from .monomer_aux import Xi, dXi_dxhi00, d2Xi_dxhi00


def a2(xs, khs, xhixm, a2ij, epsij, f1, f2, f3):

    xi = Xi(xhixm, f1, f2, f3)
    a2 = a2ij * khs * (1 + xi) * epsij/2
    a = np.matmul(np.matmul(a2, xs), xs)

    return a


def da2_dxhi00(xs, khs, dkhs, xhixm, dxhixm_dxhi00, da2ij, epsij, f1, f2, f3):

    xi, dxi = dXi_dxhi00(xhixm, dxhixm_dxhi00, f1, f2, f3)
    ctes = epsij / 2
    sum1, dsum1 = da2ij

    a2 = sum1 * khs * (1 + xi) * ctes
    da2 = sum1 * (1+xi) * dkhs + khs * (1 + xi) * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    da = np.array([a2, da2])
    a = np.matmul(np.matmul(da, xs), xs)
    return a


def d2a2_dxhi00(xs, khs, dkhs, d2khs, xhixm, dxhixm_dxhi00, d2a2ij, epsij,
                f1, f2, f3):

    xi, dxi, d2xi = d2Xi_dxhi00(xhixm, dxhixm_dxhi00, f1, f2, f3)

    ctes = epsij / 2
    sum1, dsum1, d2sum1 = d2a2ij

    a2 = sum1 * khs * (1 + xi) * ctes

    da2 = sum1 * (1+xi) * dkhs + khs * (1 + xi) * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    d2a2 = sum1 * (1 + xi) * d2khs
    d2a2 += sum1 * khs * d2xi
    d2a2 += d2sum1 * (1 + xi) * khs
    d2a2 += 2 * dkhs * ((1 + xi) * dsum1 + sum1 * dxi)
    d2a2 += 2 * khs * dsum1 * dxi
    d2a2 *= ctes

    da = np.array([a2, da2, d2a2])
    a = np.matmul(np.matmul(da, xs), xs)
    return a
