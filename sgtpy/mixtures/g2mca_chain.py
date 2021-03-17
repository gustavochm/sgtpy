import numpy as np


def g2mca(xhi00, khs, xm, da2new, suma_g2, eps, a1vdw_cte):

    g2 = 3 * da2new / eps
    g2 -= khs * suma_g2 / xhi00
    g2 /= xm
    g2 /= - a1vdw_cte

    return g2


def dg2mca_dxhi00(xhi00, khs, dkhs, xm, d2a2new, dsuma_g2, eps, a1vdw_cte):

    sum1, dsum1 = dsuma_g2
    g2 = 3. * np.asarray(d2a2new) / eps

    g2[0] -= khs * sum1 / xhi00
    g2[1] += khs * sum1 / xhi00**2
    g2[1] -= (sum1 * dkhs + khs * dsum1)/xhi00
    g2 /= xm
    g2 /= - a1vdw_cte
    return g2


def d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xm, d3a2new,
                   d2suma_g2, eps, a1vdw_cte):

    sum1, dsum1, d2sum1 = d2suma_g2

    g2 = 3. * np.asarray(d3a2new) / eps

    aux1 = khs * sum1 / xhi00
    aux2 = aux1 / xhi00
    aux3 = (sum1 * dkhs + khs * dsum1)/xhi00

    g2[0] -= aux1
    g2[1] += aux2
    g2[1] -= aux3
    g2[2] += 2*aux3/xhi00
    g2[2] -= 2*aux2/xhi00
    g2[2] -= (2*dkhs*dsum1 + sum1*d2khs + khs*d2sum1)/xhi00
    g2 /= xm
    g2 /= - a1vdw_cte

    return g2


def dg2mca_dx(xhi00, khs, dkhsx, xm, ms, da2new, da2newx, suma_g2, suma_g2x,
              eps, a1vdw_cte):

    g2 = 3 * da2new / eps
    g2 -= khs * suma_g2 / xhi00
    g2 /= xm
    g2 /= - a1vdw_cte

    dg2x = -np.multiply.outer(dkhsx,  suma_g2)
    dg2x -= khs * suma_g2x
    dg2x += khs * np.multiply.outer(ms, suma_g2) / xm
    dg2x /= xhi00
    dg2x -= 3 * np.multiply.outer(ms, da2new / eps) / xm
    dg2x += 3. * da2newx / eps
    dg2x /= xm
    dg2x /= -a1vdw_cte

    return g2, dg2x


def dg2mca_dxxhi(xhi00, khs, dkhs, dkhsx, xm, ms, d2a2new, da2newx,
                 dsuma_g2, suma_g2x, eps, a1vdw_cte):

    sum1, dsum1 = dsuma_g2
    g2 = 3. * np.asarray(d2a2new) / eps

    g2[0] -= khs * sum1 / xhi00
    g2[1] += khs * sum1 / xhi00**2
    g2[1] -= (sum1 * dkhs + khs * dsum1)/xhi00
    g2 /= xm
    g2 /= - a1vdw_cte

    dg2x = -np.multiply.outer(dkhsx,  sum1)
    dg2x -= khs * suma_g2x
    dg2x += khs * np.multiply.outer(ms, sum1) / xm
    dg2x /= xhi00
    dg2x -= 3 * np.multiply.outer(ms, d2a2new[0] / eps) / xm
    dg2x += 3. * da2newx / eps
    dg2x /= xm
    dg2x /= -a1vdw_cte
    return g2, dg2x
