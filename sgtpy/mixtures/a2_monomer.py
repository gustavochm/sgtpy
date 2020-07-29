import numpy as np
from .monomer_aux import Xi, dXi_dxhi00,  d2Xi_dxhi00, dXi_dx


# Eq A20
def a2(xs, khs, xhixm, a2ij, epsij, f1, f2, f3):

    xi = Xi(xhixm, f1, f2, f3)
    a2 = a2ij * khs * (1 + xi) * epsij/2
    a = np.dot(xs, np.dot(a2, xs))

    return a


def da2_dxhi00(xs, khs, dkhs, xhixm, dxhim_dxhi00, da2ij, epsij, f1, f2, f3):

    xi, dxi = dXi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3)
    ctes = epsij / 2
    sum1, dsum1 = da2ij

    a2 = sum1 * khs * (1 + xi) * ctes
    da2 = sum1 * (1+xi) * dkhs + khs * (1 + xi) * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    da = np.array([a2, da2])
    a = np.matmul(np.matmul(da, xs), xs)
    return a


def d2a2_dxhi00(xs, khs, dkhs, d2khs, xhixm, dxhim_dxhi00, d2a2ij, epsij,
                f1, f2, f3):

    xi, dxi, d2xi = d2Xi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3)

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


def da2_dx(xs, dxs_dx, khs, dkhs, xhixm, dxhim_dx, a2ij, da2ijx, epsij, f1,
           f2, f3):
    xi = Xi(xhixm, f1, f2, f3)
    dxi = dXi_dx(xhixm, dxhim_dx, f1, f2, f3)
    ctes = epsij/2

    a2 = a2ij * khs * (1 + xi) * ctes
    a = np.dot(xs, np.dot(a2, xs))

    da2 = np.multiply.outer(a2ij * (1+xi), dkhs).T
    da2 += a2ij * khs * dxi
    da2 += khs * (1 + xi) * da2ijx
    da2 *= ctes

    aux1 = xs * a2ij * khs * (1 + xi) * ctes
    suma1 = 2*np.sum(dxs_dx.T@aux1, axis=1)
    dax = np.matmul(np.matmul(da2, xs), xs)
    dax = xs@da2@xs + suma1

    return a, dax


def da2_dxxhi(xs, dxs_dx, khs, dkhs, dkhsx, xhixm, dxhim_dxhi00, dxhim_dx,
              da2ij, da2ijx, epsij, f1, f2, f3):

    ctes = epsij/2

    xi, dxi = dXi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3)
    dxix = dXi_dx(xhixm, dxhim_dx, f1, f2, f3)

    ctes = epsij / 2
    sum1, dsum1 = da2ij

    a2 = sum1 * khs * (1 + xi) * ctes
    da2 = sum1 * (1+xi) * dkhs + khs * (1 + xi) * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    da = np.array([a2, da2])
    a = np.matmul(np.matmul(da, xs), xs)

    da2 = np.multiply.outer(sum1 * (1+xi), dkhsx).T
    da2 += sum1 * khs * dxix
    da2 += khs * (1 + xi) * da2ijx
    da2 *= ctes

    aux1 = xs * sum1 * khs * (1 + xi) * ctes
    suma1 = 2*np.sum(dxs_dx.T@aux1, axis=1)
    dax = np.matmul(np.matmul(da2, xs), xs)
    dax = xs@da2@xs + suma1

    return a, dax
