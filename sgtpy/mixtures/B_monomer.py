import numpy as np

# Equation A12
def B(xhi00, xhix, xm, I_ij, J_ij, a1vdw_cteij):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*I_ij/xhix13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix13)
    b *= -a1vdw_cteij * xm * xhi00
    return b


def dB_dxhi00(xhi00, xhix, xm, I_ij, J_ij, a1vdw_cteij, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)
    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1

    aux0 = a1vdw_cteij * xm
    aux1 = xhix_1 * (-2*I_ij + xhix * (I_ij + 9*J_ij + 9*J_ij*xhix))
    aux2 = -5*I_ij+9*J_ij+xhix*(2*I_ij+36*J_ij+9*J_ij*xhix)

    b = (1-xhix/2)*I_ij/xhix_13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix_13)
    b *= - aux0 * xhi00

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= aux0
    db /= 2 * xhix_14

    return b, db


def d2B_dxhi00(xhi00, xhix, xm, I_ij, J_ij, a1vdw_cteij, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1

    aux0 = a1vdw_cteij * xm
    aux1 = xhix_1 * (-2*I_ij + xhix * (I_ij + 9*J_ij + 9*J_ij*xhix))
    aux2 = -5*I_ij+9*J_ij+xhix*(2*I_ij+36*J_ij+9*J_ij*xhix)
    aux3 = xhix_1 * aux2
    aux4 = -3*(I_ij-4*J_ij)+xhix*(I_ij+21*J_ij+3*J_ij*xhix)

    b = (1-xhix/2)*I_ij/xhix_13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix_13)
    b *= - aux0 * xhi00

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= aux0
    db /= 2 * xhix_14

    d2b = aux3 * dxhix_dxhi00
    d2b += 3 * xhi00 * aux4 * dxhix_dxhi00**2
    d2b *= aux0
    d2b /= xhix_15

    return b, db, d2b


def d3B_dxhi00(xhi00, xhix, xm, I_ij, J_ij, a1vdw_cteij, dxhix_dxhi00):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1
    xhix_16 = xhix_15*xhix_1

    dxhix_dxhi00_2 = dxhix_dxhi00**2
    dxhix_dxhi00_3 = dxhix_dxhi00_2 * dxhix_dxhi00

    aux0 = a1vdw_cteij * xm
    aux1 = xhix_1 * (-2*I_ij + xhix * (I_ij + 9*J_ij + 9*J_ij*xhix))
    aux2 = -5*I_ij+9*J_ij+xhix*(2*I_ij+36*J_ij+9*J_ij*xhix)
    aux3 = aux2 * xhix_1
    aux4 = -3*(I_ij-4*J_ij)+xhix*(I_ij+21*J_ij+3*J_ij*xhix)
    aux5 = (-14*I_ij+81*J_ij+xhix*(4*I_ij+90*J_ij+9*J_ij*xhix))

    b = (1-xhix/2)*I_ij/xhix_13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix_13)
    b *= -aux0 * xhi00

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= aux0
    db /= 2 * xhix_14

    d2b = aux3 * dxhix_dxhi00
    d2b += 3 * xhi00 * aux4 * dxhix_dxhi00_2
    d2b *= aux0
    d2b /= xhix_15

    d3b = 9 * xhix_1 * aux4 * dxhix_dxhi00_2
    d3b += 3 * xhi00 * aux5 * dxhix_dxhi00_3
    d3b *= aux0
    d3b /= xhix_16

    return b, db, d2b, d3b


def dB_dx_dxhi00_dxxhi(xhi00, xhix, xm, ms, I_ij, J_ij, a1vdw_cte,
                       dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1

    b = (1-xhix/2)*I_ij/xhix_13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix_13)
    b *= -a1vdw_cte * xm * xhi00

    aux1 = xhix_1 * (-2*I_ij + xhix * (I_ij + 9*J_ij + 9*J_ij*xhix))
    aux2 = -5*I_ij+9*J_ij+xhix*(2*I_ij+36*J_ij+9*J_ij*xhix)
    # aux3 = aux2 * xhix_1

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= a1vdw_cte * xm
    db /= 2 * xhix_14

    bx = np.multiply.outer(ms, aux1)
    bx += np.multiply.outer(dxhix_dx, aux2*xm)
    bx *= a1vdw_cte * xhi00
    bx /= 2 * xhix_14

    aux4 = aux1 + xhi00 * aux2 * dxhix_dxhi00
    aux4 *= xhix_1

    aux5 = (-3*(I_ij-4*J_ij)+xhix*(I_ij+21*J_ij+3*J_ij*xhix))
    aux5 *= 6*xhi00*dxhix_dxhi00
    aux5 += aux2 * xhix_1
    aux5 *= xm

    aux6 = aux2 * xhix_1 * xm * xhi00

    bxxhi = np.multiply.outer(ms, aux4)
    bxxhi += np.multiply.outer(dxhix_dx, aux5)
    bxxhi += np.multiply.outer(dxhix_dx_dxhi00, aux6)
    bxxhi *= a1vdw_cte
    bxxhi /= (2 * xhix_15)
    return b, db, bx, bxxhi


def dB_dx_d2xhi00_dxxhi(xhi00, xhix, xm, ms, I_ij, J_ij, a1vdw_cte,
                        dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):

    xhix_1 = 1-xhix
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1

    b = (1-xhix/2)*I_ij/xhix_13 - 9.*xhix*(1+xhix)*J_ij/(2.*xhix_13)
    b *= -a1vdw_cte * xm * xhi00

    aux1 = xhix_1 * (-2*I_ij + xhix * (I_ij + 9*J_ij + 9*J_ij*xhix))
    aux2 = -5*I_ij+9*J_ij+xhix*(2*I_ij+36*J_ij+9*J_ij*xhix)
    aux3 = aux2 * xhix_1
    aux4 = -3*(I_ij-4*J_ij)+xhix*(I_ij+21*J_ij+3*J_ij*xhix)

    db = aux1 + xhi00 * dxhix_dxhi00 * aux2
    db *= a1vdw_cte * xm
    db /= 2 * xhix_14

    bx = np.multiply.outer(ms, aux1)
    bx += np.multiply.outer(dxhix_dx, aux2*xm)
    bx *= a1vdw_cte * xhi00
    bx /= 2 * xhix_14

    aux5 = aux1 + xhi00 * aux2 * dxhix_dxhi00
    aux5 *= xhix_1

    aux6 = aux2 * xhix_1
    aux6 += 6*xhi00*aux4*dxhix_dxhi00
    aux6 *= xm

    aux7 = aux2 * xhix_1 * xm * xhi00

    bxxhi = np.multiply.outer(ms, aux5)
    bxxhi += np.multiply.outer(dxhix_dx, aux6)
    bxxhi += np.multiply.outer(dxhix_dx_dxhi00, aux7)
    bxxhi *= a1vdw_cte
    bxxhi /= (2 * xhix_15)

    d2b = aux3 * dxhix_dxhi00
    d2b += 3 * xhi00 * aux4 * dxhix_dxhi00**2
    d2b *= a1vdw_cte * xm
    d2b /= xhix_15
    return b, db, d2b, bx, bxxhi
