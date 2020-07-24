import numpy as np

#Hard sphere Eq A6
def ahs(xhi):
    xhi0, xhi1, xhi2, xhi3 = xhi
    a = (xhi2**3/xhi3**2 - xhi0) * np.log(1 - xhi3)
    a += 3 * xhi1 * xhi2 / (1 - xhi3)
    a += xhi2**3 / xhi3 / (1 - xhi3)**2
    a /= xhi0
    return a

def dahs_dxhi00(xhi, dxhi_dxhi00):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    
    a = (xhi2**3/xhi3**2 - xhi0) * np.log(1 - xhi3)
    a += 3 * xhi1 * xhi2 / (1 - xhi3)
    a += xhi2**3 / xhi3 / (1 - xhi3)**2
    a /= xhi0

    da = xhi2 * dxhi2**2 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * (-1 + xhi3)
    da -= dxhi0 * dxhi3 * (-1 + xhi3)**2
    da /= (-1 + xhi3)**3 * dxhi0
    
    return np.array([a , da])

def d2ahs_dxhi00(xhi, dxhi_dxhi00):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    
    a = (xhi2**3/xhi3**2 - xhi0) * np.log(1 - xhi3)
    a += 3 * xhi1 * xhi2 / (1 - xhi3)
    a += xhi2**3 / xhi3 / (1 - xhi3)**2
    a /= xhi0

    da = xhi2 * dxhi2**2 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * (-1 + xhi3)
    da -= dxhi0 * dxhi3 * (-1 + xhi3)**2
    da /= (-1 + xhi3)**3 * dxhi0

    d2a = -6 * dxhi1 * dxhi2 * dxhi3 * (-1 + xhi3)
    d2a += dxhi0 * dxhi3**2 * (-1 + xhi3)**2
    d2a += dxhi2**3 * (3 + xhi3 * (4 -  xhi3))
    d2a /= (-1 + xhi3)**4 * dxhi0
    
    return np.array([a , da, d2a])

def d3ahs_dxhi00(xhi, dxhi_dxhi00):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00

    d3a = -9*dxhi1*dxhi2*dxhi3*(-1+xhi3)
    d3a += dxhi0*dxhi3**2*(-1+xhi3)**2
    d3a += dxhi2**3 * (8 + xhi3 * (5 -  xhi3))
    d3a *= 2*dxhi3
    d3a /= -(-1 + xhi3)**5 * dxhi0
    
    return d3a

def dahs_dx(xhi, dxhi_dx):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dx
    
    log3 = np.log(1 - xhi3)
    
    a = (xhi2**3/xhi3**2 - xhi0) * log3
    a += 3 * xhi1 * xhi2 / (1 - xhi3)
    a += xhi2**3 / xhi3 / (1 - xhi3)**2
    a /= xhi0
    
    dax1 = -3*xhi0*xhi2**2 * (log3 + xhi3/(-1 + xhi3)**2)
    dax1 *= dxhi2 / xhi3**2
    
    dax2 = (xhi2 * dxhi0 - xhi0 * dxhi2) * (-1 + xhi3)
    dax2 += xhi0 * xhi2 * dxhi3
    dax2 *= -3. * xhi1/(-1 + xhi3)
    
    dax2 += 3 * xhi0 * xhi2 * dxhi1
    dax2 += xhi0**2 *  dxhi3
    dax2 /= (-1 + xhi3)
    
    dax3 = log3 * dxhi0 - xhi0 * (-2 + xhi3) *dxhi3 /(-1 + xhi3)**2
    dax3 *= xhi3
    dax3 += 2*log3*xhi0*dxhi3
    dax3 += xhi3**2 * ((-1+xhi3)*dxhi0 + 2*xhi0*dxhi3)/(-1 + xhi3)**3
    dax3 *= (xhi2/xhi3)**3
    
    dax = dax1 + dax2 + dax3
    dax /= -xhi0**2
    
    return a, dax

def dahs_dxxhi(xhi, dxhi_dxhi00, dxhi_dx):
    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    dxhi0x, dxhi1x, dxhi2x, dxhi3x = dxhi_dx
    
    log3 = np.log(1 - xhi3)
    
    a = (xhi2**3/xhi3**2 - xhi0) * log3
    a += 3 * xhi1 * xhi2 / (1 - xhi3)
    a += xhi2**3 / xhi3 / (1 - xhi3)**2
    a /= xhi0
    
    da = xhi2 * dxhi2**2 * (-3 + xhi3)
    da += 3 * dxhi1 * dxhi2 * (-1 + xhi3)
    da -= dxhi0 * dxhi3 * (-1 + xhi3)**2
    da /= (-1 + xhi3)**3 * dxhi0
    
    dax1 = -3*xhi0*xhi2**2 * (log3 + xhi3/(-1 + xhi3)**2)
    dax1 *= dxhi2x / xhi3**2
    
    dax2 = (xhi2 * dxhi0x - xhi0 * dxhi2x) * (-1 + xhi3)
    dax2 += xhi0 * xhi2 * dxhi3x
    dax2 *= -3. * xhi1/(-1 + xhi3)
    
    dax2 += 3 * xhi0 * xhi2 * dxhi1x
    dax2 += xhi0**2 *  dxhi3x
    dax2 /= (-1 + xhi3)
    
    dax3 = log3 * dxhi0x - xhi0 * (-2 + xhi3) *dxhi3x /(-1 + xhi3)**2
    dax3 *= xhi3
    dax3 += 2*log3*xhi0*dxhi3x
    dax3 += xhi3**2 * ((-1+xhi3)*dxhi0x + 2*xhi0*dxhi3x)/(-1 + xhi3)**3
    dax3 *= (xhi2/xhi3)**3
    
    dax = dax1 + dax2 + dax3
    dax /= -xhi0**2
    
    return np.array([a , da]), dax
