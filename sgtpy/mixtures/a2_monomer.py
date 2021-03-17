import numpy as np
from .monomer_aux import Xi, dXi_dxhi00,  d2Xi_dxhi00, dXi_dx, dXi_dx_dxhi00


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
    xi1 = 1 + xi

    a2 = sum1 * khs * xi1 * ctes
    da2 = sum1 * xi1 * dkhs + khs * xi1 * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    da = np.array([a2, da2])
    a = np.matmul(np.matmul(da, xs), xs)
    return a


def d2a2_dxhi00(xs, khs, dkhs, d2khs, xhixm, dxhim_dxhi00, d2a2ij, epsij,
                f1, f2, f3):

    xi, dxi, d2xi = d2Xi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3)
    xi1 = 1 + xi

    ctes = epsij / 2
    sum1, dsum1, d2sum1 = d2a2ij

    a2 = sum1 * khs * xi1 * ctes

    da2 = sum1 * xi1 * dkhs + khs * xi1 * dsum1 + khs * sum1 * dxi
    da2 *= ctes

    d2a2 = sum1 * xi1 * d2khs
    d2a2 += sum1 * khs * d2xi
    d2a2 += d2sum1 * xi1 * khs
    d2a2 += 2 * dkhs * (xi1 * dsum1 + sum1 * dxi)
    d2a2 += 2 * khs * dsum1 * dxi
    d2a2 *= ctes

    da = np.array([a2, da2, d2a2])
    a = np.matmul(np.matmul(da, xs), xs)
    return a


def da2_dx(xs, dxs_dx, khs, dkhsx, xhixm, dxhixm_dx, a2ij, da2ijx, epsij, f1,
           f2, f3):

    xi, dxi = dXi_dx(xhixm, dxhixm_dx, f1, f2, f3)
    xi1 = 1 + xi
    ctes = epsij/2
    aux0 = xi1 * a2ij
    aux1 = aux0 * khs
    aux2 = aux1 * ctes

    a = np.dot(xs, np.dot(aux2, xs))

    da2 = np.multiply.outer(dkhsx, aux0)
    da2 += a2ij * khs * dxi
    da2 += khs * xi1 * da2ijx
    da2 *= ctes

    aux3 = aux2 * xs
    suma1 = 2*np.sum(dxs_dx@aux3, axis=1)
    dax = np.matmul(np.matmul(xs, da2), xs) + suma1

    return a, dax


def da2_dxxhi(xs, dxs_dx, khs, dkhs, dkhsx, xhixm, dxhim_dxhi00, dxhim_dx,
              da2ij, da2ijx, epsij, f1, f2, f3):

    ctes = epsij/2

    xi, dxi, dxix = dXi_dx_dxhi00(xhixm, dxhim_dxhi00, dxhim_dx, f1, f2, f3)
    xi1 = 1 + xi

    sum1, dsum1 = da2ij

    aux0 = xi1 * sum1
    aux1 = xi1 * khs
    aux2 = sum1 * khs

    aux3 = aux0 * khs
    aux4 = aux3 * ctes

    a2 = aux4
    da2 = aux0 * dkhs + aux1 * dsum1 + aux2 * dxi
    da2 *= ctes

    da = np.array([a2, da2])
    a = np.matmul(np.matmul(xs, da), xs)

    da2 = np.multiply.outer(dkhsx, aux0)
    da2 += aux1 * da2ijx
    da2 += aux2 * dxix
    da2 *= ctes

    aux5 = xs * aux4
    suma1 = 2*np.sum(dxs_dx@aux5, axis=1)
    dax = np.matmul(np.matmul(xs, da2), xs) + suma1
    return a, dax
