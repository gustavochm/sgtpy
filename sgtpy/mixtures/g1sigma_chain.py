import numpy as np


def g1sigma(xhi00, xm, da1, suma_g1, eps, dii):
    cte = 1 / (12 * eps * dii**3)
    g1 = 3 * da1 - suma_g1/xhi00
    g1 /= xm
    g1 *= cte
    return g1


def dg1sigma_dxhi00(xhi00, xm, da1, suma_g1, eps, dii):

    cte = 1 / (12 * eps * dii**3)

    sum1, dsum1 = suma_g1

    g1 = 3 * da1
    g1[0] -= sum1/xhi00
    g1[1] += (sum1/xhi00 - dsum1)/xhi00

    g1 /= xm
    g1 *= cte
    return g1


def d2g1sigma_dxhi00(xhi00, xm, da1, suma_g1, eps, dii):

    cte = 1 / (12 * eps * dii**3)

    sum1, dsum1, d2sum1 = suma_g1
    g1 = 3. * da1 - suma_g1/xhi00
    g1[1] += sum1/xhi00**2
    g1[2] += -2*sum1/xhi00**3 + 2*dsum1/xhi00**2

    g1 /= xm
    g1 *= cte
    return g1


def dg1sigma_dx(xhi00, xm, ms, da1, da1x, suma_g1, suma_g1x, eps, dii):

    cte = 1 / (12 * eps * dii**3)

    g1 = 3. * da1 - suma_g1/xhi00
    g1 /= xm
    g1 *= cte

    g1x = np.multiply.outer(ms, (suma_g1/xhi00 - 3. * da1)/xm)
    g1x += -suma_g1x/xhi00
    g1x += 3*da1x
    g1x /= xm
    g1x *= cte
    return g1, g1x


def dg1sigma_dxxhi(xhi00, xm, ms, da1, da1x, suma_g1, suma_g1x, eps, dii):

    cte = 1 / (12 * eps * dii**3)

    sum1, dsum1 = suma_g1

    g1 = 3 * da1
    g1[0] -= sum1/xhi00
    g1[1] += (sum1/xhi00 - dsum1)/xhi00

    g1 /= xm
    g1 *= cte

    g1x = np.multiply.outer(ms, (sum1/xhi00 - 3. * da1[0])/xm)
    g1x += -suma_g1x/xhi00
    g1x += 3*da1x
    g1x /= xm
    g1x *= cte
    return g1, g1x
