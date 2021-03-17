import numpy as np


# Equation A29 to A33
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


def dkichain_dxhi00(xhix, dxhix_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1

    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)

    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix2))
    dk0 *= dxhix_dxhi00 / 3. / xhix_14

    dk1 = 12 + xhix * (2+xhix) * (6 - 6 * xhix + xhix2)
    dk1 *= -dxhix_dxhi00 / 2. / xhix_14

    dk2 = 3*xhix
    dk2 *= dxhix_dxhi00 / 4. / -xhix_13

    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix2))
    dk3 *= dxhix_dxhi00 / 6. / xhix_14

    dks = np.array([[k0, k1, k2, k3], [dk0, dk1, dk2, dk3]])
    return dks


def d2kichain_dxhi00(xhix, dxhix_dxhi00):

    dxhix_dxhi00_2 = dxhix_dxhi00**2

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1
    xhix_15 = xhix_14*xhix_1

    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)

    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix2))
    dk0 *= dxhix_dxhi00 / 3. / xhix_14

    dk1 = 12 + xhix * (2+xhix) * (6 - 6 * xhix + xhix2)
    dk1 *= -dxhix_dxhi00 / 2. / xhix_14

    dk2 = 3*xhix
    dk2 *= dxhix_dxhi00 / 4. / -xhix_13

    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix2))
    dk3 *= dxhix_dxhi00 / 6. / xhix_14

    d2k0 = 3 * (-30 + xhix * (1+xhix) * (4 + xhix))
    d2k0 *= dxhix_dxhi00_2 / 3. / -xhix_15

    d2k1 = 12 * (5 - 2 * (-1 + xhix) * xhix)
    d2k1 *= dxhix_dxhi00_2 / 2. / -xhix_15

    d2k2 = -3*(1+2*xhix)
    d2k2 *= dxhix_dxhi00_2 / 4. / xhix_14

    d2k3 = 6*(-4 + xhix * (-7 + xhix))
    d2k3 *= dxhix_dxhi00_2 / 6. / -xhix_15

    d2ks = np.array([[k0, k1, k2, k3],
                     [dk0, dk1, dk2, dk3],
                     [d2k0, d2k1, d2k2, d2k3]])
    return d2ks


def dkichain_dx(xhix, dxhix_dx):

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1

    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)

    dk0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix2))
    dk0 *= dxhix_dx / 3. / xhix_14

    dk1 = 12 + xhix * (2+xhix) * (6 - 6 * xhix + xhix2)
    dk1 *= -dxhix_dx / 2. / xhix_14

    dk2 = 3*xhix
    dk2 *= dxhix_dx / 4. / -xhix_13

    dk3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix2))
    dk3 *= dxhix_dx / 6. / xhix_14

    ks = np.array([k0, k1, k2, k3])
    dksx = np.array([dk0, dk1, dk2, dk3]).T
    return ks, dksx


def dkichain_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx):
    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_1 = (1 - xhix)
    xhix_13 = xhix_1**3
    xhix_14 = xhix_13*xhix_1

    k0 = -np.log(xhix_1) + (42*xhix - 39*xhix2 + 9*xhix3 - 2*xhix4)/(6*xhix_13)
    k1 = (xhix4 + 6*xhix2 - 12*xhix)/(2*xhix_13)
    k2 = -3*xhix2/(8*xhix_1**2)
    k3 = (-xhix4 + 3*xhix2 + 3*xhix)/(6*xhix_13)

    aux_k0 = 24 + xhix * (-6 + xhix * (3 - 7 * xhix + xhix2))
    aux_k0 /= 3.*xhix_14
    aux_k1 = 12 + xhix * (2+xhix) * (6 - 6 * xhix + xhix2)
    aux_k1 /= -2.*xhix_14
    aux_k2 = 3*xhix / 4. / -xhix_13
    aux_k3 = 3 + xhix * (12 + (-3 + xhix) * (-xhix + xhix2))
    aux_k3 /= 6.*xhix_14

    dks = np.array([[k0, k1, k2, k3], [aux_k0, aux_k1, aux_k2, aux_k3]])
    dksx = np.multiply.outer(dxhix_dx, dks[1])

    dks[1] *= dxhix_dxhi00

    return dks, dksx


def gdHS(x0i_matrix, xhix):
    ks = kichain(xhix)
    # x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])
    g = np.exp(np.dot(ks, x0i_matrix))
    return g


def dgdHS_dxhi00(x0i_matrix, xhix, dxhix_dxhi00):
    dks = dkichain_dxhi00(xhix, dxhix_dxhi00)
    # x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dg = np.dot(dks, x0i_matrix)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
    return dg


def d2gdHS_dxhi00(x0i_matrix, xhix, dxhix_dxhi00):
    d2ks = d2kichain_dxhi00(xhix, dxhix_dxhi00)
    # x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])
    d2g = np.matmul(d2ks, x0i_matrix)
    d2g[0] = np.exp(d2g[0])
    d2g[2] += d2g[1]**2
    d2g[2] *= d2g[0]
    d2g[1] *= d2g[0]
    return d2g


def dgdHS_dx(x0i_matrix, xhix, dxhix_dx):
    ks, dksx = dkichain_dx(xhix, dxhix_dx)
    # x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])
    g = g = np.exp(np.dot(ks, x0i_matrix))
    dgx = g * np.matmul(dksx, x0i_matrix)
    return g, dgx


def dgdHS_dxxhi(x0i_matrix, xhix, dxhix_dxhi00, dxhix_dx):
    dks, dksx = dkichain_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx)
    dg = np.dot(dks, x0i_matrix)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
    # x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])
    dgx = dg[0] * np.matmul(dksx, x0i_matrix)
    return dg, dgx
