#Eq A12
import numpy as np
from .monomer_aux import I_lam, J_lam

#a1vdw_cte = -12 * epsij * dij**3

def B(xhi00, xhix, xm, x0, lam, a1vdw_cte):
    #Esto arreglarlo debido a que estas I, J
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*I/xhix13 - 9.*xhix*(1+xhix)*J/(2.*xhix13)
    b *= -a1vdw_cte * xm * xhi00
    #b *= 12.*xhi00* xm * epsij * dij**3 
    return b

#Eq A12
def dB_dxhi00(xhi00, xhix, xm, x0,  lam , a1vdw_cte, dxhix_dxhi00):
    #Esto arreglarlo debido a que estas I, J
    #se deben evaluar una sola vez data una temperatura
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*I/xhix13 - 9.*xhix*(1+xhix)*J/(2.*xhix13)
    b *= -a1vdw_cte * xm * xhi00
    #b *= 12.* xhi00 * xm * epsij * dij**3
    
    db = (-1 + xhix) * ( -2 * I + xhix * (I + 9*J + 9*J*xhix))
    db -= xhi00 * dxhix_dxhi00 * (-5 * I +  9*J + xhix * (2 * I + 36 * J + 9 * J * xhix))
    db *= -a1vdw_cte * xm
    #db *= 12. * epsij * dij**3 * xm
    db /= 2 * (-1 + xhix)**4

    return b, db

#Eq A12
def d2B_dxhi00(xhi00, xhix, xm, x0, lam, a1vdw_cte, dxhix_dxhi00):
    #Esto arreglarlo debido a que estas I, J
    #se deben evaluar una sola vez data una temperatura
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*I/xhix13 - 9.*xhix*(1+xhix)*J/(2.*xhix13)
    b *= -a1vdw_cte * xm * xhi00
    #b *= 12.* xhi00 * xm * epsij * dij**3
    
    db = (-1 + xhix) * ( -2 * I + xhix * (I + 9*J + 9*J*xhix))
    db -= xhi00 * dxhix_dxhi00 * (-5 * I +  9*J + xhix * (2 * I + 36 * J + 9 * J * xhix))
    db *= -a1vdw_cte * xm
    #db *= 12. * epsij * dij**3 * xm
    db /= 2 * (-1 + xhix)**4
    
    d2b = -2* (-1 + xhix) * (-5*I + 9*J + xhix *(2*I + 36*J + 9*J*xhix)) * dxhix_dxhi00
    d2b += 6 * xhi00 * (-3 * (I- 4 *J) + xhix * (I + 21*J + 3* J*xhix)) * dxhix_dxhi00**2
    d2b *= -a1vdw_cte * xm
    #d2b *= 12. * epsij * dij**3 * xm
    d2b /= 2 * (-1 + xhix)**5
    return b, db, d2b

#Eq A12
def d3B_dxhi00(xhi00, xhix, xm, x0, lam, a1vdw_cte, dxhix_dxhi00):
    #Esto arreglarlo debido a que estas I, J
    #se deben evaluar una sola vez data una temperatura
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    xhix13 = (1-xhix)**3
    b = (1-xhix/2)*I/xhix13 - 9.*xhix*(1+xhix)*J/(2.*xhix13)
    #b *= 12.* xhi00 * xm * epsij * dij**3
    b *= -a1vdw_cte * xm * xhi00
    
    db = (-1 + xhix) * ( -2 * I + xhix * (I + 9*J + 9*J*xhix))
    db -= xhi00 * dxhix_dxhi00 * (-5 * I +  9*J + xhix * (2 * I + 36 * J + 9 * J * xhix))
    db *= -a1vdw_cte * xm
    #db *= 12. * epsij * dij**3 * xm
    db /= 2 * (-1 + xhix)**4
    
    d2b = -2* (-1 + xhix) * (-5*I + 9*J + xhix *(2*I + 36*J + 9*J*xhix)) * dxhix_dxhi00
    d2b += 6 * xhi00 * (-3 * (I- 4 *J) + xhix * (I + 21*J + 3* J*xhix)) * dxhix_dxhi00**2
    d2b *= -a1vdw_cte * xm
    #d2b *= 12. * epsij * dij**3 * xm
    d2b /= 2 * (-1 + xhix)**5
    
    d3b = -18 * (-1 + xhix) *(-3 * (I - 4*J) + xhix * (I + 21 * J + 3 * J * xhix))* dxhix_dxhi00**2
    d3b += 6 * xhi00 * (-14 * I + 81 * J + xhix * (4 * I + 90 * J + 9 * J * xhix)) * dxhix_dxhi00**3
    d3b *= a1vdw_cte * xm
    #d3b *= -12. * epsij * dij**3 * xm
    d3b /= 2 * (-1 + xhix)**6
    return b, db, d2b, d3b


#Eq A12
def dB_dx(xhi00, xhix, xm, ms, x0, lam, a1vdw_cte,  dxhix_dx):
    #Esto arreglarlo debido a que estas I, J
    #se deben evaluar una sola vez data una temperatura
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    b = np.multiply.outer((-1+ xhix) * (-2*I + xhix *(I + 9*J + 9*J * xhix) ),  ms).T
    b += np.multiply.outer(- (- 5*I + 9*J + xhix * (2*I+36*J + 9*J*xhix)) * xm, dxhix_dx).T
    b *= - a1vdw_cte * xhi00
    #b *= 12. * epsij * dij**3 * xhi00
    b /= 2 * (-1 + xhix)**4
    return b


#Eq A12
def dB_dx_dxhi00(xhi00, xhix, xm, ms, x0, lam, a1vdw_cte,
                 dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):
    #Esto arreglarlo debido a que estas I, J
    #se deben evaluar una sola vez data una temperatura
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    aux1 = 2*I + xhix * (-3* (I + 3*J) + xhix * (I + 9*J*xhix))
    aux1 += -xhi00 * (-5*I + 9*J + xhix * (2*I+36*J+9*J*xhix)) * dxhix_dxhi00
    aux1 *= (-1 + xhix)
    
    aux2 = - 5 *I + 9 *J + xhix * (2*I + 36*J + 9*J*xhix)
    aux2 *= -(-1+xhix)
    aux2 += 6*xhi00 * (-3*(I-4*J) + xhix * (I + 21 * J + 3 * J * xhix))*dxhix_dxhi00
    aux2 *=  xm
    
    aux3 = - 5 * I + 9*J + xhix * (2*I + 36*J + 9*J*xhix)
    aux3 *= -xhi00 * (-1 + xhix)
    aux3 *= xm
    
    b = np.multiply.outer(ms, aux1)
    b += np.multiply.outer(dxhix_dx, aux2)
    b +=  np.multiply.outer(dxhix_dx_dxhi00, aux3)
    
    b *= -a1vdw_cte
    #b *= 12. * epsij * dij**3
    b /= (2 * (-1 + xhix)**5)
    return b
