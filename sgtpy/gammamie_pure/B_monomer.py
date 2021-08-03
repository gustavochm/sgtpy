from __future__ import division, print_function, absolute_import


# Eq (20) Paper 2014
def B(xhi00, xhix, xm, Ikl, Jkl, a1vdw_cte):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*Ikl/xhix13 - 9.*xhix*(1+xhix)*Jkl/(2.*xhix13)
    b *= -a1vdw_cte * xm * xhi00
    return b


def dB_dxhi00(xhi00, xhix, xm, Ikl, Jkl, a1vdw_cte, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1

    b = (1-xhix/2)*Ikl/xhix_13 - 9.*xhix*(1+xhix)*Jkl/(2.*xhix_13)
    b *= -a1vdw_cte * xm * xhi00

    aux1 = xhix_1 * (-2*Ikl + xhix * (Ikl + 9*Jkl + 9*Jkl*xhix))
    aux2 = -5*Ikl+9*Jkl+xhix*(2*Ikl+36*Jkl+9*Jkl*xhix)

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= a1vdw_cte * xm
    db /= 2 * xhix_14
    return b, db


def d2B_dxhi00(xhi00, xhix, xm, Ikl, Jkl, a1vdw_cte, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1

    b = (1-xhix/2)*Ikl/xhix_13 - 9.*xhix*(1+xhix)*Jkl/(2.*xhix_13)
    b *= -a1vdw_cte * xm * xhi00

    aux1 = xhix_1 * (-2*Ikl + xhix * (Ikl + 9*Jkl + 9*Jkl*xhix))
    aux2 = -5*Ikl+9*Jkl+xhix*(2*Ikl+36*Jkl+9*Jkl*xhix)
    aux3 = aux2 * xhix_1
    aux4 = -3*(Ikl-4*Jkl)+xhix*(Ikl+21*Jkl+3*Jkl*xhix)

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= a1vdw_cte * xm
    db /= 2 * xhix_14

    d2b = aux3 * dxhix_dxhi00
    d2b += 3 * xhi00 * aux4 * dxhix_dxhi00**2
    d2b *= a1vdw_cte * xm
    d2b /= xhix_15
    return b, db, d2b


def d3B_dxhi00(xhi00, xhix, xm, Ikl, Jkl, a1vdw_cte, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1
    xhix_16 = xhix_15*xhix_1

    dxhix_dxhi00_2 = dxhix_dxhi00**2
    dxhix_dxhi00_3 = dxhix_dxhi00_2 * dxhix_dxhi00

    b = (1-xhix/2)*Ikl/xhix_13 - 9.*xhix*(1+xhix)*Jkl/(2.*xhix_13)
    b *= -a1vdw_cte * xm * xhi00

    aux1 = xhix_1 * (-2*Ikl + xhix * (Ikl + 9*Jkl + 9*Jkl*xhix))
    aux2 = -5*Ikl+9*Jkl+xhix*(2*Ikl+36*Jkl+9*Jkl*xhix)
    aux3 = aux2 * xhix_1
    aux4 = -3*(Ikl-4*Jkl)+xhix*(Ikl+21*Jkl+3*Jkl*xhix)
    aux5 = (-14*Ikl+81*Jkl+xhix*(4*Ikl+90*Jkl+9*Jkl*xhix))

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= a1vdw_cte * xm
    db /= 2 * xhix_14

    d2b = aux3 * dxhix_dxhi00
    d2b += 3 * xhi00 * aux4 * dxhix_dxhi00_2
    d2b *= a1vdw_cte * xm
    d2b /= xhix_15

    d3b = 9 * xhix_1 * aux4 * dxhix_dxhi00_2
    d3b += 3 * xhi00 * aux5 * dxhix_dxhi00_3
    d3b *= a1vdw_cte * xm
    d3b /= xhix_16

    return b, db, d2b, d3b
