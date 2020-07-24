import numpy as np

#Eq A29 a A33
def kichain(xhix):
    xhix2 = xhix**2
    xhix3 = xhix**3
    xhix4 = xhix2**2
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)
    return np.array([k0, k1, k2, k3])

#Eq A29 a A33
def dkichain_dxhi00(xhix, dxhix_dxhi00):
    xhix2 = xhix**2
    xhix3 = xhix**3
    xhix4 = xhix2**2
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    
    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)
    
    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix**2))
    dk0 *= dxhix_dxhi00 / 3. / (-1 + xhix)**4
    
    dk1 = 12 + xhix * (2+ xhix) * (6 - 6* xhix + xhix**2)
    dk1 *= -dxhix_dxhi00 / 2. / (-1 + xhix)**4
    
    dk2 = 3* xhix 
    dk2 *=  dxhix_dxhi00  / 4. / (-1 + xhix)**3
    
    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix**2)) 
    dk3 *= dxhix_dxhi00 / 6. / (-1 + xhix)**4
    return np.array([[k0, k1, k2, k3], 
                     [dk0, dk1, dk2, dk3]])

#Eq A29 a A33
def d2kichain_dxhi00(xhix, dxhix_dxhi00):
    xhix2 = xhix**2
    xhix3 = xhix**3
    xhix4 = xhix2**2
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    
    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)
    
    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix**2))
    dk0 *= dxhix_dxhi00 / 3. / (-1 + xhix)**4
    
    dk1 = 12 + xhix * (2+ xhix) * (6 - 6* xhix + xhix**2)
    dk1 *= -dxhix_dxhi00 / 2. / (-1 + xhix)**4
    
    dk2 = 3* xhix 
    dk2 *=  dxhix_dxhi00  / 4. / (-1 + xhix)**3
    
    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix**2)) 
    dk3 *= dxhix_dxhi00 / 6. / (-1 + xhix)**4
    
    d2k0 = 3 * (-30 + xhix * (1+ xhix) * (4 + xhix))
    d2k0 *=   dxhix_dxhi00**2 / 3. / (-1 + xhix)**5
    
    d2k1 = 12 * (5 - 2 * (-1 + xhix) * xhix)
    d2k1 *= dxhix_dxhi00**2 / 2. / (-1 + xhix)**5
    
    d2k2 = -3* (1+ 2 *xhix) 
    d2k2 *=  dxhix_dxhi00**2  / 4. / (-1 + xhix)**4
    
    d2k3 = 6 *(-4 + xhix * (-7 + xhix))
    d2k3 *= dxhix_dxhi00**2 / 6. / (-1 + xhix)**5
    return np.array([[k0, k1, k2, k3], 
                     [dk0, dk1, dk2, dk3],
                     [d2k0, d2k1, d2k2, d2k3]])

#Eq A29 a A33
def dkichain_dx(xhix, dxhix_dx):
    xhix2 = xhix**2
    #xhix3 = xhix**3
    #xhix4 = xhix2**2
    xhix_1 = (1 - xhix)
    #xhix_13 = xhix_1**3
    
    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix**2))
    dk0 *= dxhix_dx / 3. / (-1 + xhix)**4
    
    dk1 = 12 + xhix * (2+ xhix) * (6 - 6* xhix + xhix**2)
    dk1 *= -dxhix_dx / 2. / (-1 + xhix)**4

    
    dk2 = 3* xhix 
    dk2 *=  dxhix_dx  / 4. / (-1 + xhix)**3
    
    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix**2)) 
    dk3 *= dxhix_dx / 6. / (-1 + xhix)**4
    return np.array([dk0, dk1, dk2, dk3]).T


def gdHS(x0i, xhix):
    ks = kichain(xhix)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    g = np.exp(np.dot(ks, xs))
    return g


def dgdHS_dxhi00(x0i, xhix, dxhix_dxhi00):
    dks = dkichain_dxhi00(xhix , dxhix_dxhi00)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dg = np.dot(dks, xs)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
    return dg

def d2gdHS_dxhi00(x0i, xhix, dxhix_dxhi00):
    d2ks = d2kichain_dxhi00(xhix , dxhix_dxhi00)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    d2g = np.matmul(d2ks, xs)
    d2g[0] = np.exp(d2g[0])
    d2g[2] += d2g[1]**2
    d2g[2] *= d2g[0]
    d2g[1] *= d2g[0]
    return d2g

def dgdHS_dx(x0i, xhix, dxhix_dx):
    g = gdHS(x0i, xhix)
    dks = dkichain_dx(xhix, dxhix_dx)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dg = g * np.matmul(dks, xs)
    return g, dg

def dgdHS_dxxhi(x0i, xhix, dxhix_dxhi00, dxhix_dx):
    g = dgdHS_dxhi00(x0i, xhix, dxhix_dxhi00)
    dks = dkichain_dx(xhix, dxhix_dx)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dg = g[0] * np.matmul(dks, xs)
    return g, dg

    dks = dkichain_dxhi00(xhix , dxhix_dxhi00)
    xs = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dg = np.dot(dks, xs)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
