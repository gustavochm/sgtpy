from __future__ import division, print_function, absolute_import
import numpy as np


# Equation (62) Paper 2014
def g2mca(xhi00, khs, xs_m, da2new, suma_g2, eps_ii, a1vdw_cteii):

    g2 = 3 * da2new / eps_ii
    g2 -= khs * suma_g2 / xhi00
    g2 /= xs_m
    g2 /= - a1vdw_cteii
    return g2


def dg2mca_dxhi00(xhi00, khs, dkhs, xs_m, d2a2new, dsuma_g2, eps_ii,
                  a1vdw_cteii):

    sum1, dsum1 = dsuma_g2
    g2 = 3. * np.asarray(d2a2new) / eps_ii

    g2[0] -= khs * sum1 / xhi00
    g2[1] += khs * sum1 / xhi00**2
    g2[1] -= (sum1 * dkhs + khs * dsum1)/xhi00
    g2 /= xs_m
    g2 /= - a1vdw_cteii
    return g2


def d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xs_m, d3a2new, d2suma_g2,
                   eps_ii, a1vdw_cteii):

    sum1, dsum1, d2sum1 = d2suma_g2

    g2 = 3. * np.asarray(d3a2new) / eps_ii

    aux1 = khs * sum1 / xhi00
    aux2 = aux1 / xhi00
    aux3 = (sum1 * dkhs + khs * dsum1)/xhi00

    g2[0] -= aux1
    g2[1] += aux2
    g2[1] -= aux3
    g2[2] += 2*aux3/xhi00
    g2[2] -= 2*aux2/xhi00
    g2[2] -= (2*dkhs*dsum1 + sum1*d2khs + khs*d2sum1)/xhi00
    g2 /= xs_m
    g2 /= - a1vdw_cteii

    return g2
