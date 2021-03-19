import numpy as np


# Second pertubation Eq 36
def a2m(suma_a2, khs, xi, cte_a2m):
    a2 = khs*(1+xi)*suma_a2*cte_a2m
    return a2


def da2m_deta(suma_a2, dKhs, dXi, cte_a2m):

    khs, dkhs = dKhs
    xi, dx1 = dXi
    x1 = 1. + xi

    #sum1, dsum1 = np.matmul(da1sb, x0lambda)
    sum1, dsum1 = suma_a2
    # a2 = khs*x1*eps*c2*sum1/2.
    a2 = khs*x1*sum1*cte_a2m

    da2 = sum1*x1*dkhs + khs * x1 * dsum1 + khs * sum1 * dx1
    da2 *= cte_a2m
    return np.hstack([a2, da2])


def d2a2m_deta(suma_a2,  d2Khs, d2Xi, cte_a2m):

    khs, dkhs, d2khs = d2Khs
    xi, dx1, d2x1 = d2Xi
    x1 = 1. + xi

    sum1, dsum1, d2sum1 = suma_a2

    a2 = khs*x1*sum1*cte_a2m

    da2 = sum1*x1*dkhs + khs * x1 * dsum1 + khs * sum1 * dx1
    da2 *= cte_a2m

    d2a2 = d2khs * sum1 * x1 + d2x1 * sum1 * khs + d2sum1 * khs * x1
    d2a2 += 2 * dkhs * dsum1 * x1
    d2a2 += 2 * sum1 * dkhs * dx1
    d2a2 += 2 * khs * dsum1 * dx1
    d2a2 *= cte_a2m
    return np.hstack([a2, da2, d2a2])


def da2m_new_deta(suma_a2, dKhs, cte_a2m):

    khs, dkhs = dKhs
    # sum1, dsum1 = np.matmul(da1sb, x0lambda)
    sum1, dsum1 = suma_a2
    da2 = cte_a2m*(dkhs*sum1 + khs * dsum1)

    return da2


def d2a2m_new_deta(suma_a2, d2Khs, cte_a2m):

    khs, dkhs, d2khs = d2Khs
    # sum1, dsum1, d2sum1 = np.matmul(d2a1sb, x0lambda)
    sum1, dsum1, d2sum1 = suma_a2
    # aux = cte_a2m
    da2 = cte_a2m*(dkhs*sum1 + khs * dsum1)

    d2a2 = 2 * dkhs * dsum1 + sum1 * d2khs + d2sum1 * khs
    d2a2 *= cte_a2m

    return np.hstack([da2, d2a2])


def d3a2m_new_deta(suma_a2, d3Khs, cte_a2m):

    khs, dkhs, d2khs, d3khs = d3Khs
    # sum1, dsum1, d2sum1, d3sum1 = np.matmul(d3a1sb, x0lambda)
    sum1, dsum1, d2sum1, d3sum1 = suma_a2
    # aux = cte_a2m
    da2 = cte_a2m*(dkhs*sum1 + khs * dsum1)

    d2a2 = 2 * dkhs * dsum1 + sum1 * d2khs + d2sum1 * khs
    d2a2 *= cte_a2m

    d3a2 = 3 * dsum1 * d2khs + 3 * dkhs * d2sum1
    d3a2 += khs * d3sum1 + d3khs * sum1
    d3a2 *= cte_a2m
    return np.hstack([da2, d2a2, d3a2])
