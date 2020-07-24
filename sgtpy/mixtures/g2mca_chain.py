from .monomer_aux import kHS, dkHS_dxhi00, d2kHS_dxhi00, dkHS_dx
import numpy as np

def g2mca(xhi00, khs, xm, da2new, suma_g2, eps, dii):
    #khs = kHS(xhix)
    
    cte = 1 / (12* eps**2 * dii**3)
    
    g2 = 3 * da2new 
    g2 -= eps * khs * suma_g2 / xhi00
    g2 /= xm
    g2 *= cte
    return g2

def dg2mca_dxhi00(xhi00, khs, dkhs, xm, da2new, d2a2new, suma_g2, eps, dii):
    
    #khs = kHS(xhix)
    #khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    
    cte = 1 / (12* eps**2 * dii**3)
    cte2 = eps
    
    sum1, dsum1 = suma_g2
    g2 = 3*np.array([da2new, d2a2new])
    #g2 = -cte2 * khs *  suma_a2/ xhi00
    g2[0] -= cte2 * khs * sum1 / xhi00
    g2[1] += cte2 * khs * sum1 / xhi00**2
    g2[1] -= cte2 * (sum1 * dkhs + khs * dsum1)/xhi00
    g2 /= xm
    g2 *= cte
    return g2

def d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xm, da2new, d2a2new, d3a2new, suma_g2, eps, dii):
    
    #khs = kHS(xhix)
    #dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    #khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    
    cte = 1 / (12* eps**2 * dii**3)
    cte2 = eps
    
    sum1, dsum1, d2sum1 = suma_g2
    
    g2 = 3.*np.array([da2new, d2a2new, d3a2new])

    g2[0] -= cte2 * khs * sum1 / xhi00
    g2[1] += cte2 * khs * sum1 / xhi00**2
    g2[1] -= cte2 * (sum1 * dkhs + khs * dsum1)/xhi00
    g2[2] += 2*cte2*(sum1*dkhs + khs *dsum1)/xhi00**2
    g2[2] -= 2*cte2*khs*sum1/xhi00**3
    g2[2] -= (2*dkhs*dsum1+sum1*d2khs + khs*d2sum1)*cte2/xhi00
    g2 /= xm
    g2 *= cte
    return g2


def dg2mca_dx(xhi00, khs, dkhs, xm, ms, da2new, da2newx,
              suma_g2, suma_g2x, eps, dii):
    
    #khs = kHS(xhix)
    #dkhs = dkHS_dx(xhix, dxhix_dx)
    
    cte = 1 / (12* eps**2 * dii**3)
    #cte2 = eps
    
    g2 = 3 * da2new
    g2 -= eps * khs * suma_g2 / xhi00
    g2 /= xm
    g2 *= cte

    
    dg2x = -np.multiply.outer(dkhs,  suma_g2)
    dg2x -= khs * suma_g2x
    dg2x += khs * np.multiply.outer(ms, suma_g2)/ xm
    dg2x /= xhi00
    dg2x *= eps
    dg2x -= 3 * np.multiply.outer(ms, da2new) / xm
    dg2x += 3. * da2newx
    dg2x /= xm
    dg2x *= cte
    
    return g2, dg2x


def dg2mca_dxxhi(xhi00, khs, dkhs, dkhsx, xm, ms, da2new, d2a2new, da2newx,
              suma_g2, suma_g2x, eps, dii):
    
    cte = 1 / (12* eps**2 * dii**3)
    cte2 = eps
    
    sum1, dsum1 = suma_g2
    g2 = 3*np.array([da2new, d2a2new])
    #g2 = -cte2 * khs *  suma_a2/ xhi00
    g2[0] -= cte2 * khs * sum1 / xhi00
    g2[1] += cte2 * khs * sum1 / xhi00**2
    g2[1] -= cte2 * (sum1 * dkhs + khs * dsum1)/xhi00
    g2 /= xm
    g2 *= cte
    
    dg2x = -np.multiply.outer(dkhsx,  suma_g2[0])
    dg2x -= khs * suma_g2x
    dg2x += khs * np.multiply.outer(ms, suma_g2[0])/ xm
    dg2x /= xhi00
    dg2x *= eps
    dg2x -= 3 * np.multiply.outer(ms, da2new) / xm
    dg2x += 3. * da2newx
    dg2x /= xm
    dg2x *= cte
    
    return g2, dg2x
