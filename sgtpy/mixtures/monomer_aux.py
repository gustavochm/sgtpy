import numpy as np


# Equation A17
def xhi_eff(xhix, cictes):
    xhix_vec = np.array([xhix, xhix**2, xhix**3, xhix**4])
    xhieff = np.matmul(cictes, xhix_vec)
    return xhieff


def dxhieff_dxhi00(xhix, cictes, dxhix_dxhi00):
    xhix_vec = np.array([1., 2 * xhix, 3*xhix**2, 4*xhix**3])
    dxhieff = dxhix_dxhi00 * np.matmul(cictes, xhix_vec)
    return dxhieff


def d2xhieff_dxhi00(xhix, cictes, dxhix_dxhi00):
    xhix_vec = np.array([0, 2, 6*xhix, 12*xhix**2])
    dxhieff = (dxhix_dxhi00)**2 * np.matmul(cictes, xhix_vec)
    return dxhieff


def d3xhieff_dxhi00(xhix, cictes, dxhix_dxhi00):
    xhix_vec = np.array([0, 0., 6., 24*xhix])
    dxhieff = (dxhix_dxhi00)**3 * np.matmul(cictes, xhix_vec)
    return dxhieff


def dxhieff_dx(xhix, cictes, dxhix_dx):
    xhix_vec = np.array([1., 2 * xhix, 3*xhix**2, 4*xhix**3])
    dxhieff = np.multiply.outer(dxhix_dx, np.matmul(cictes, xhix_vec))
    return dxhieff


def dxhieff_dx_dxhi00(xhix, cictes,  dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):
    xhix_vec = np.array([1., 2 * xhix, 3*xhix**2, 4*xhix**3])
    aux1 = np.matmul(cictes, xhix_vec)

    ter1 = np.multiply.outer(dxhix_dx_dxhi00, aux1)

    xhix_vec2 = np.array([0., 2, 6*xhix, 12*xhix**2])
    ter2 = np.multiply.outer(dxhix_dx, np.matmul(cictes, xhix_vec2))*dxhix_dxhi00
    return ter2 + ter1


# Equation A16
def J_lam(x0, lam):
    lam3 = 3. - lam
    lam4 = 4. - lam
    J = (lam3*x0**lam4 - lam4*x0**lam3 + 1.)/(lam3 * lam4)
    return J


# Equation A14
def I_lam(x0, lam):
    lam3 = 3. - lam
    I = (x0**lam3-1.) / lam3
    return I


# Equation A21
def kHS(xhix):
    k = (1-xhix)**4
    k /= 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4
    return k


def dkHS_dxhi00(xhix, dxhix_dxhi00):
    den = 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4

    k = (1-xhix)**4
    k /= den

    dk = -2 - 5*xhix + xhix**2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dxhi00
    dk /= den**2
    return k, dk


def d2kHS_dxhi00(xhix, dxhix_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix2**2
    xhix5 = xhix3 * xhix2
    xhix6 = xhix5 * xhix

    den = 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4

    k = (1-xhix)**4
    k /= den

    dk = -2 - 5*xhix + xhix2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dxhi00
    dk /= den**2

    d2k = 17 + 82 * xhix + 39 * xhix2 - 80 * xhix3
    d2k += 77 * xhix4 - 30 * xhix5 + 3*xhix6
    d2k *= 4 * (- 1 + xhix)**2 * dxhix_dxhi00**2
    d2k /= den**3
    return k, dk, d2k


def d3kHS_dxhi00(xhix, dxhix_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix2**2
    xhix5 = xhix3 * xhix2
    xhix6 = xhix5 * xhix

    den = 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4

    k = (1-xhix)**4
    k /= den

    dk = -2 - 5*xhix + xhix**2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dxhi00
    dk /= den**2

    d2k = 17 + 82 * xhix + 39 * xhix2 - 80 * xhix3
    d2k += 77 * xhix4 - 30 * xhix5 + 3*xhix6
    d2k *= 4 * (- 1 + xhix)**2 * dxhix_dxhi00**2
    d2k /= den**3

    d3k = -97 - 109 * xhix + 238 * xhix2 - 352 * xhix4
    d3k += 372 * xhix5 - 210*xhix6 + 77*xhix**7 - 15*xhix**8 + xhix**9
    d3k *= xhix
    d3k -= 13.
    d3k *= 12.
    d3k *= -4 * (- 1 + xhix) * dxhix_dxhi00**3
    d3k /= den**4

    return k, dk, d2k, d3k


def dkHS_dx(xhix, dxhix_dx):
    dk = -2 - 5*xhix + xhix**2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dx
    dk /= (1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4)**2
    return dk


def dkHS_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00):

    den = 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4

    k = (1-xhix)**4
    k /= den

    dk = -2 - 5*xhix + xhix**2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dxhi00
    dk /= den**2

    dkx = -2 - 5*xhix + xhix**2
    dkx *= -4 * (- 1 + xhix)**3 * dxhix_dx
    dkx /= (1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4)**2

    aux1 = 17 + 82 * xhix + 39 * xhix**2 - 80 * xhix**3
    aux1 += 77 * xhix**4 - 30 * xhix**5 + 3*xhix**6

    aux2 = -2 - 13 * xhix - 27*xhix**2 - 8 * xhix**3
    aux2 += 22 * xhix**4 - 9 * xhix**5 + xhix**6

    dkxxhi = dxhix_dxhi00 * dxhix_dx * aux1
    dkxxhi += -(-1+xhix)*aux2*dxhix_dx_dxhi00

    dkxxhi *= 4 * (- 1 + xhix)**2
    dkxxhi /= den**3
    return k, dk, dkx, dkxxhi


def dkHS_dx_dxhi002(xhix, dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00):

    xhix2 = xhix**2
    xhix3 = xhix2 * xhix
    xhix4 = xhix2**2
    xhix5 = xhix3 * xhix2
    xhix6 = xhix5 * xhix

    den = 1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4

    k = (1-xhix)**4
    k /= den

    dk = -2 - 5*xhix + xhix**2
    dk *= -4 * (- 1 + xhix)**3 * dxhix_dxhi00
    dk /= den**2

    d2k = 17 + 82 * xhix + 39 * xhix2 - 80 * xhix3
    d2k += 77 * xhix4 - 30 * xhix5 + 3*xhix6
    d2k *= 4 * (- 1 + xhix)**2 * dxhix_dxhi00**2
    d2k /= den**3

    dkx = -2 - 5*xhix + xhix**2
    dkx *= -4 * (- 1 + xhix)**3 * dxhix_dx
    dkx /= (1 + 4*xhix + 4*xhix**2 - 4*xhix**3 + xhix**4)**2

    aux1 = 17 + 82 * xhix + 39 * xhix**2 - 80 * xhix**3
    aux1 += 77 * xhix**4 - 30 * xhix**5 + 3*xhix**6

    aux2 = -2 - 13 * xhix - 27*xhix**2 - 8 * xhix**3
    aux2 += 22 * xhix**4 - 9 * xhix**5 + xhix**6

    dkxxhi = dxhix_dxhi00 * dxhix_dx * aux1
    dkxxhi += -(-1+xhix)*aux2*dxhix_dx_dxhi00

    dkxxhi *= 4 * (- 1 + xhix)**2
    dkxxhi /= den**3
    return k, dk, d2k, dkx, dkxxhi


# Equation A22
def Xi(xhixm, f1, f2, f3):
    x = f1*xhixm + f2*xhixm**5 + f3*xhixm**8
    return x


# Equation A22
def dXi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3):
    xhixm2 = xhixm**2
    xhixm4 = xhixm2**2
    xhixm5 = xhixm4*xhixm
    xhixm7 = xhixm5*xhixm2
    xhixm8 = xhixm4**2

    x = f1*xhixm + f2*xhixm5 + f3*xhixm8

    dx = f1 + 5*f2*xhixm4 + 8*f3*xhixm7
    dx *= dxhim_dxhi00
    return x, dx


def d2Xi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3):
    xhixm2 = xhixm**2
    xhixm3 = xhixm2*xhixm
    xhixm4 = xhixm2**2
    xhixm5 = xhixm4*xhixm
    xhixm6 = xhixm5*xhixm
    xhixm7 = xhixm5*xhixm2
    xhixm8 = xhixm4**2

    x = f1*xhixm + f2*xhixm5 + f3*xhixm8

    dx = f1 + 5*f2*xhixm4 + 8*f3*xhixm7
    dx *= dxhim_dxhi00

    d2x = 20*f2*xhixm3 + 56*f3*xhixm6
    d2x *= dxhim_dxhi00**2
    return x, dx, d2x


def d3Xi_dxhi00(xhixm, dxhim_dxhi00, f1, f2, f3):
    x = 60*f2*xhixm**2 + 56*6*f3*xhixm**5
    x *= dxhim_dxhi00**3
    return x


# Equation A22
def dXi_dx(xhixm, dxhim_dx, f1, f2, f3):
    x = f1 + 5*f2*xhixm**4 + 8*f3*xhixm**7
    x = np.multiply.outer(dxhim_dx, x)
    return x
