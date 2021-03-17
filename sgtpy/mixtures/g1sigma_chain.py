import numpy as np


def g1sigma(xhi00, xm, da1_chain, suma_g1, a1vdw_cteii):
    g1 = 3 * da1_chain - suma_g1/xhi00
    g1 /= xm
    g1 /= -a1vdw_cteii
    return g1


def dg1sigma_dxhi00(xhi00, xm, d2a1_chain, dsuma_g1, a1vdw_cteii):

    sum1, dsum1 = dsuma_g1
    g1 = 3 * d2a1_chain
    g1[0] -= sum1/xhi00
    g1[1] += (sum1/xhi00 - dsum1)/xhi00
    g1 /= xm
    g1 /= -a1vdw_cteii

    return g1


def d2g1sigma_dxhi00(xhi00, xm, d3a1_chain, d2suma_g1, a1vdw_cteii):

    sum1, dsum1, d2sum1 = d2suma_g1
    g1 = 3. * d3a1_chain - d2suma_g1/xhi00
    g1[1] += sum1/xhi00**2
    g1[2] += -2*sum1/xhi00**3 + 2*dsum1/xhi00**2
    g1 /= xm
    g1 /= -a1vdw_cteii
    return g1


def dg1sigma_dx(xhi00, xm, ms, da1, da1x, suma_g1, suma_g1x, a1vdw_cte):

    g1 = 3. * da1 - suma_g1/xhi00
    g1 /= xm
    g1 /= -a1vdw_cte

    g1x = np.multiply.outer(ms, (suma_g1/xhi00 - 3. * da1)/xm)
    g1x += -suma_g1x/xhi00
    g1x += 3*da1x
    g1x /= xm
    g1x /= -a1vdw_cte

    return g1, g1x


def dg1sigma_dxxhi(xhi00, xm, ms, da1, da1x, suma_g1, suma_g1x, a1vdw_cte):

    sum1, dsum1 = suma_g1

    g1 = 3 * da1
    g1[0] -= sum1/xhi00
    g1[1] += (sum1/xhi00 - dsum1)/xhi00
    g1 /= xm
    g1 /= -a1vdw_cte

    g1x = np.multiply.outer(ms, (sum1/xhi00 - 3. * da1[0])/xm)
    g1x += -suma_g1x/xhi00
    g1x += 3*da1x
    g1x /= xm
    g1x /= -a1vdw_cte
    return g1, g1x
