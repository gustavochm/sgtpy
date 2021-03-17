import numpy as np


# Third pertubation
def a3(xs, xhim, epsij, f4, f5, f6):
    a3ij = np.exp(f5 * xhim + f6 * xhim**2)
    a3ij *= -epsij**3 * f4 * xhim
    a = np.dot(xs, np.dot(a3ij, xs))
    return a


def da3_dxhi00(xs, xhim, dxhim_dxhi00, epsij, f4, f5, f6):

    cte = -epsij**3 * f4 * np.exp(f5 * xhim + f6 * xhim**2)

    a3ij = cte * xhim

    da3ij = cte * (1 + f5 * xhim + 2 * f6 * xhim**2) * dxhim_dxhi00

    a3 = np.array([a3ij, da3ij])
    a = np.matmul(np.matmul(a3, xs), xs)

    return a


def d2a3_dxhi00(xs, xhim, dxhim_dxhi00, epsij, f4, f5, f6):

    cte = -epsij**3 * f4 * np.exp(f5 * xhim + f6 * xhim**2)

    a3ij = cte * xhim

    da3ij = cte * (1 + f5 * xhim + 2 * f6 * xhim**2) * dxhim_dxhi00

    d2a3ij = cte * dxhim_dxhi00**2
    aux3 = 2 * f5 + f5**2 * xhim + 6 * f6 * xhim
    aux3 += 4 * f5 * f6 * xhim**2 + 4 * f6**2*xhim**3
    d2a3ij *= aux3
    a3 = np.array([a3ij, da3ij, d2a3ij])
    a = np.matmul(np.matmul(a3, xs), xs)

    return a


def da3_dx(xs, dxs_dx, xhim, dxhim_dx, epsij, f4, f5, f6):
    aux1 = -epsij**3 * f4 * np.exp(f5 * xhim + f6 * xhim**2)
    aux2 = (1 + f5 * xhim + 2 * f6 * xhim**2)
    aux3 = aux1 * aux2

    a3ij = aux1 * xhim

    da3ijx = np.multiply.outer(dxhim_dx, aux3)

    aux4 = xs * a3ij
    a3m = np.sum(aux4.T*xs)
    suma1 = 2*np.sum(dxs_dx@aux4, axis=1)

    dax = xs@da3ijx@xs + suma1
    return a3m, dax


def da3_dxxhi(xs, dxs_dx, xhim, dxhim_dxhi00, dxhim_dx, epsij, f4, f5, f6):

    aux1 = -epsij**3 * f4 * np.exp(f5 * xhim + f6 * xhim**2)
    aux2 = (1 + f5 * xhim + 2 * f6 * xhim**2)
    aux3 = aux1 * aux2

    a3ij = aux1 * xhim
    da3ij = aux3 * dxhim_dxhi00
    a3 = np.array([a3ij, da3ij])
    a3m = np.matmul(np.matmul(a3, xs), xs)

    da3ijx = np.multiply.outer(dxhim_dx, aux3)

    aux1 = xs * a3ij
    suma1 = 2*np.sum(dxs_dx@aux1, axis=1)

    dax = xs@da3ijx@xs + suma1

    return a3m, dax
