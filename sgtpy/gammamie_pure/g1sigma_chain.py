from __future__ import division, print_function, absolute_import
import numpy as np


# Equation (53) Paper 2014
def g1sigma(xhi00, xs_m, da1_chain, suma_g1, a1vdw_cteii):
    g1 = 3 * da1_chain - suma_g1/xhi00
    g1 /= xs_m
    g1 /= -a1vdw_cteii
    return g1


def dg1sigma_dxhi00(xhi00, xs_m, d2a1_chain, dsuma_g1, a1vdw_cteii):

    sum1, dsum1 = dsuma_g1
    g1 = 3 * d2a1_chain
    g1[0] -= sum1/xhi00
    g1[1] += (sum1/xhi00 - dsum1)/xhi00
    g1 /= xs_m
    g1 /= -a1vdw_cteii

    return g1


def d2g1sigma_dxhi00(xhi00, xs_m, d3a1_chain, d2suma_g1, a1vdw_cteii):
    sum1, dsum1, d2sum1 = d2suma_g1
    g1 = 3. * d3a1_chain - d2suma_g1/xhi00
    g1[1] += sum1/xhi00**2
    g1[2] += -2*sum1/xhi00**3 + 2*dsum1/xhi00**2
    g1 /= xs_m
    g1 /= -a1vdw_cteii
    return g1
