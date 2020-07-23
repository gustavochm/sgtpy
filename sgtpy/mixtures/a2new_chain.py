import numpy as np
from .monomer_aux import kHS, dkHS_dxhi00, d2kHS_dxhi00, d3kHS_dxhi00
from .monomer_aux import dkHS_dx, dkHS_dx_dxhi00
#Eq A20


def da2new_dxhi00(khs, dkhs, da2, eps):
    #khs = kHS(xhix)
    #khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    
    ctes =  eps / 2
    sum1, dsum1 = da2   
    
    da2 = sum1  * dkhs + khs  * dsum1 
    da2 *= ctes

    return da2


def d2a2new_dxhi00(khs, dkhs, d2khs, d2a2, eps):
    #khs = kHS(xhix)
    #dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    #khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    
    ctes =  eps / 2
    sum1, dsum1, d2sum1 = d2a2
    
    da2 = sum1  * dkhs + khs  * dsum1
    da2 *= ctes
        
    d2a2 = sum1 * d2khs + khs * d2sum1 + 2 * dsum1 * dkhs
    d2a2 *= ctes

    return da2, d2a2


def d3a2new_dxhi00(khs, dkhs, d2khs, d3khs, d3a2, eps):
    #khs = kHS(xhix)
    #dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    #d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    #khs, dkhs, d2khs, d3khs = d3kHS_dxhi00(xhix, dxhix_dxhi00)
    
    ctes =  eps/2
    sum1, dsum1, d2sum1, d3sum1 = d3a2
    
    da2 = sum1  * dkhs + khs  * dsum1
    da2 *= ctes

    d2a2 = sum1 * d2khs + khs * d2sum1 + 2 * dsum1 * dkhs
    d2a2 *= ctes

    
    d3a2 = 3* (dsum1 * d2khs + dkhs * d2sum1)
    d3a2 += sum1 * d3khs + khs * d3sum1
    d3a2 *= ctes

    return da2, d2a2, d3a2


def da2new_dx_dxhi00(khs, dkhs, dkhsx, dkhsxxhi,
                     da2, da2x, da2xxhi, eps):

    #khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    #dkhsx = dkHS_dx(xhix, dxhix_dx)
    #k,dkhsxxhi = dkHS_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    #khs, dkhs, dkhsx, dkhsxxhi = dkHS_dx_dxhi00(xhix, dxhix_dxhi00,
    #                            dxhix_dx, dxhix_dx_dxhi00)
    ctes =  eps / 2
    sum1, dsum1 = da2
    
    da2 = sum1  * dkhs + khs  * dsum1
    da2 *= ctes
    
    da2x = np.multiply.outer(dkhsx, dsum1) + dkhs*da2x
    da2x += np.multiply.outer(dkhsxxhi, sum1) + khs*da2xxhi
    da2x *= ctes

    return da2, da2x


def da2new_dxxhi_dxhi00(khs, dkhs, d2khs, dkhsx, dkhsxxhi,
                     d2a2, da2x, da2xxhi, eps):
    
    ctes =  eps / 2
    sum1, dsum1, d2sum1 = d2a2
    
    da2 = sum1  * dkhs + khs  * dsum1
    da2 *= ctes
    
    d2a2 = sum1 * d2khs + khs * d2sum1 + 2 * dsum1 * dkhs
    d2a2 *= ctes
    
    da2x = np.multiply.outer(dkhsx, dsum1) + dkhs*da2x
    da2x += np.multiply.outer(dkhsxxhi, sum1) + khs*da2xxhi
    da2x *= ctes
    
    return da2, d2a2, da2x
