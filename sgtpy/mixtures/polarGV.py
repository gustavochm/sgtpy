import numpy as np


# Expansion of polar term
aij = np.array([[0.3043504, 0.9534641, -1.1610080],
               [-0.1358588, -1.8396383, 4.5258607],
               [1.4493329, 2.0131180, 0.9751222],
               [0.3556977, -7.3724958, -12.281038],
               [-2.0653308, 8.2374135, 5.9397575]])

bij = np.array([[0.2187939, -0.5873164, 3.4869576],
               [-1.1896431, 1.2489132, -14.915974],
               [1.1626889, -0.508528, 15.372022],
               [0., 0., 0.],
               [0., 0., 0.]])

cij = np.array([[-0.0646774, -0.9520876, -0.6260979],
               [0.1975882, 2.9924258, 1.2924686],
               [-0.8087562, -2.3802636, 1.652783],
               [0.6902849, -0.2701261, -3.4396744],
               [0.,  0., 0.]])


def Apolar(rho, x, anij, bnij, cnij,
           eta, epsa, epsija, sigma3, sigmaij3, sigmaijk3,  npol, muad2):

    etavec = np.power(eta, [0, 1, 2, 3, 4])

    J2DDij = np.tensordot(anij + bnij * epsija, etavec, axes=((0), (0)))

    ter1 = x * sigma3 * epsa * npol * muad2
    outer_ter1 = np.outer(ter1, ter1)
    ter2 = - np.pi * outer_ter1 / sigmaij3
    A2 = rho * np.sum(ter2 * J2DDij)

    J3DDijk = np.tensordot(cnij, etavec, axes=((0), (0)))

    ter3 = np.multiply.outer(ter1, outer_ter1) / sigmaijk3

    A3 = rho**2 * np.sum(ter3 * J3DDijk)
    A3 *= -4 * np.pi**2 / 3

    with np.errstate(divide='ignore', invalid='ignore'):
        Apol = np.nan_to_num(A2/(1 - A3/A2))

    return Apol


def dApolar_drho(rho, x, anij, bnij, cnij, eta, deta, epsa, epsija, sigma3,
                 sigmaij3, sigmaijk3,  npol, muad2):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3]])

    J2DDij, dJ2DDij = np.tensordot(etavec, anij+bnij*epsija, axes=((1), (0)))
    dJ2DDij *= deta

    J3DDijk, dJ3DDijk = np.tensordot(etavec, cnij, axes=((1), (0)))
    dJ3DDijk *= deta

    ter1 = x * sigma3 * epsa * npol * muad2
    outer_ter1 = np.outer(ter1, ter1)
    ter2 = - np.pi * outer_ter1 / sigmaij3
    A2 = rho * np.sum(ter2 * J2DDij)
    dA2 = np.sum(ter2 * (J2DDij + rho * dJ2DDij))

    ter2 = np.multiply.outer(ter1, outer_ter1) / sigmaijk3

    A3 = rho**2 * np.sum(ter2 * J3DDijk)
    A3 *= -4 * np.pi**2 / 3
    dA3 = np.sum(ter2 * rho * (2 * J3DDijk + rho * dJ3DDijk))
    dA3 *= -4 * np.pi**2 / 3

    with np.errstate(divide='ignore', invalid='ignore'):
        Apol = np.nan_to_num(A2/(1 - A3/A2))
        dApolar = A2 * ((A2 - 2 * A3) * dA2 + A2 * dA3)
        dApolar /= ((A2 - A3)**2)

    A = np.nan_to_num(np.array([Apol, dApolar]))

    return A


def d2Apolar_drho(rho, x, anij, bnij, cnij, eta, deta, epsa, epsija, sigma3,
                  sigmaij3, sigmaijk3,  npol, muad2):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3],
                      [0., 0., 2., 6*eta, 12*eta**2]])

    J2DDij, dJ2DDij, d2J2DDij = np.tensordot(etavec, anij+bnij*epsija,
                                             axes=((1), (0)))
    dJ2DDij *= deta
    d2J2DDij *= deta**2

    J3DDijk, dJ3DDijk, d2J3DDijk = np.tensordot(etavec, cnij, axes=((1), (0)))
    dJ3DDijk *= deta
    d2J3DDijk *= deta**2

    ter1 = x * sigma3 * epsa * npol * muad2
    outer_ter1 = np.outer(ter1, ter1)
    ter2 = - np.pi * outer_ter1 / sigmaij3
    A2 = rho * np.sum(ter2 * J2DDij)
    dA2 = np.sum(ter2 * (J2DDij + rho * dJ2DDij))
    d2A2 = np.sum(ter2 * (2 * dJ2DDij + rho * d2J2DDij))

    ter2 = np.multiply.outer(ter1, outer_ter1) / sigmaijk3

    A3 = rho**2 * np.sum(ter2 * J3DDijk)
    A3 *= -4 * np.pi**2 / 3
    dA3 = np.sum(ter2 * rho * (2 * J3DDijk + rho * dJ3DDijk))
    dA3 *= -4 * np.pi**2 / 3
    d2A3 = np.sum(ter2 * (2 * J3DDijk + rho * (4*dJ3DDijk+rho*d2J3DDijk)))
    d2A3 *= -4 * np.pi**2 / 3

    with np.errstate(divide='ignore', invalid='ignore'):
        Apol = (A2/(1 - A3/A2))

        dApolar = A2 * ((A2 - 2 * A3) * dA2 + A2 * dA3)
        dApolar /= ((A2 - A3)**2)

        d2Apolar = 2 * A3**2 * (dA2**2 + A2 * d2A2)
        d2Apolar += A2**2 * (2 * dA3**2 + A2 * (d2A2 + d2A3))
        d2Apolar -= A2 * A3 * (4 * dA2 * dA3 + A2 * (3*d2A2 + d2A3))
        d2Apolar /= ((A2 - A3)**3)

    A = np.nan_to_num(np.array([Apol, dApolar, d2Apolar]))

    return A


def dApolar_dx(rho, x, anij, bnij, cnij, eta, deta_dx, epsa, epsija, sigma3,
               sigmaij3, sigmaijk3,  npol, muad2):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3]])

    J2DDij, dJ2DDij = np.tensordot(etavec, anij+bnij*epsija, axes=((1), (0)))
    dJ2DDij_dx = np.multiply.outer(deta_dx, dJ2DDij)

    J3DDijk, dJ3DDijk = np.tensordot(etavec, cnij, axes=((1), (0)))
    dJ3DDijk_dx = np.multiply.outer(deta_dx, dJ3DDijk)

    ter1 = x * sigma3 * epsa * npol * muad2
    outer_ter1 = np.outer(ter1, ter1)
    ter2 = - np.pi * outer_ter1 / sigmaij3
    A2 = rho * np.sum(ter2 * J2DDij)

    ter1x = sigma3 * epsa * npol * muad2
    outer_ter1x = np.outer(ter1x, ter1x)
    ter2x = - np.pi * outer_ter1x * J2DDij/sigmaij3
    dA2x = rho * (2*ter2x@x + np.sum(ter2 * dJ2DDij_dx, axis=(1, 2)))

    ter2 = np.multiply.outer(ter1, outer_ter1) / sigmaijk3

    A3 = rho**2 * np.sum(ter2 * J3DDijk)
    A3 *= -4 * np.pi**2 / 3

    ter2x = np.multiply.outer(ter1x, outer_ter1x) / sigmaijk3
    dA3x = 3*x@(ter2x * J3DDijk)@x + np.sum(ter2 * dJ3DDijk_dx, axis=(1, 2, 3))
    dA3x *= -4 * np.pi**2 / 3 * rho**2

    with np.errstate(divide='ignore', invalid='ignore'):
        Apol = np.nan_to_num(A2/(1 - A3/A2))
        dApolarx = A2 * ((A2 - 2 * A3) * dA2x + A2 * dA3x)
        dApolarx /= ((A2 - A3)**2)

    return Apol, np.nan_to_num(dApolarx)


def dApolar_dxrho(rho, x, anij, bnij, cnij, eta, deta, deta_dx, epsa, epsija,
                  sigma3, sigmaij3, sigmaijk3,  npol, muad2):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3]])

    J2DDij, dJ2DDij = np.tensordot(etavec, anij+bnij*epsija, axes=((1), (0)))
    dJ2DDij_dx = np.multiply.outer(deta_dx, dJ2DDij)
    dJ2DDij *= deta

    J3DDijk, dJ3DDijk = np.tensordot(etavec, cnij, axes=((1), (0)))
    dJ3DDijk_dx = np.multiply.outer(deta_dx, dJ3DDijk)
    dJ3DDijk *= deta

    ter1 = x * sigma3 * epsa * npol * muad2
    outer_ter1 = np.outer(ter1, ter1)
    ter2 = - np.pi * outer_ter1 / sigmaij3
    A2 = rho * np.sum(ter2 * J2DDij)
    dA2 = np.sum(ter2 * (J2DDij + rho * dJ2DDij))

    ter1x = sigma3 * epsa * npol * muad2
    outer_ter1x = np.outer(ter1x, ter1x)
    ter2x = - np.pi * outer_ter1x * J2DDij/sigmaij3
    dA2x = rho * (2*ter2x@x + np.sum(ter2 * dJ2DDij_dx, axis=(1, 2)))

    ter2 = np.multiply.outer(ter1, outer_ter1) / sigmaijk3

    A3 = rho**2 * np.sum(ter2 * J3DDijk)
    A3 *= -4 * np.pi**2 / 3

    dA3 = np.sum(ter2 * rho * (2 * J3DDijk + rho * dJ3DDijk))
    dA3 *= -4 * np.pi**2 / 3

    ter2x = np.multiply.outer(ter1x, outer_ter1x) / sigmaijk3
    dA3x = 3*x@(ter2x * J3DDijk)@x + np.sum(ter2 * dJ3DDijk_dx, axis=(1, 2, 3))
    dA3x *= -4 * np.pi**2 / 3 * rho**2

    with np.errstate(divide='ignore', invalid='ignore'):
        Apol = A2/(1 - A3/A2)

        dApolar = A2 * ((A2 - 2 * A3) * dA2 + A2 * dA3)
        dApolar /= ((A2 - A3)**2)

        dApolarx = A2 * ((A2 - 2 * A3) * dA2x + A2 * dA3x)
        dApolarx /= ((A2 - A3)**2)

    A = np.nan_to_num(np.array([Apol, dApolar]))

    return A, np.nan_to_num(dApolarx)
