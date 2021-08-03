from __future__ import division, print_function, absolute_import
import numpy as np


# Third pertubation
def a3(xs, xhixm, eps_kl, f4, f5, f6):
    a3kl = np.exp(f5 * xhixm + f6 * xhixm**2)
    a3kl *= -eps_kl**3 * f4 * xhixm
    a = np.dot(xs, np.dot(a3kl, xs))
    return a


def da3_dxhi00(xs, xhixm, dxhixm_dxhi00, eps_kl, f4, f5, f6):

    cte = -eps_kl**3 * f4 * np.exp(f5 * xhixm + f6 * xhixm**2)

    a3kl = cte * xhixm

    da3kl = cte * (1 + f5 * xhixm + 2 * f6 * xhixm**2) * dxhixm_dxhi00

    a3 = np.array([a3kl, da3kl])
    a = np.matmul(np.matmul(a3, xs), xs)

    return a


def d2a3_dxhi00(xs, xhixm, dxhim_dxhi00, eps_kl, f4, f5, f6):

    cte = -eps_kl**3 * f4 * np.exp(f5 * xhixm + f6 * xhixm**2)

    a3kl = cte * xhixm

    da3kl = cte * (1 + f5 * xhixm + 2 * f6 * xhixm**2) * dxhim_dxhi00

    d2a3kl = cte * dxhim_dxhi00**2

    aux3 = 2 * f5 + f5**2 * xhixm + 6 * f6 * xhixm
    aux3 += 4 * f5 * f6 * xhixm**2 + 4 * f6**2*xhixm**3
    d2a3kl *= aux3
    a3 = np.array([a3kl, da3kl, d2a3kl])
    a = np.matmul(np.matmul(a3, xs), xs)

    return a
