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


def Apol(rho, eta, epsa, anij, bnij, cnijk, muad2, npol, sigma3):

    etav = np.power(eta, [0, 1, 2, 3, 4])

    J2Dij = np.dot(anij + bnij * epsa, etav)
    aux2 = -np.pi * epsa**2 * sigma3 * npol**2 * muad2**2
    A2 = aux2 * rho * J2Dij

    J3Dijk = np.dot(cnijk, etav)
    aux3 = - 4 * np.pi**2 / 3 * epsa**3 * sigma3**2 * npol**3 * muad2**3
    A3 = aux3 * rho**2 * J3Dijk

    Apolar = A2 / (1 - A3/A2)
    return Apolar


def dApol_drho(rho, eta, deta, epsa, anij, bnij, cnijk, muad2, npol, sigma3):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3]])

    J2Dij, dJ2Dij = np.tensordot(etavec, anij + bnij * epsa, axes=((1), (0)))
    aux2 = -np.pi * epsa**2 * sigma3 * npol**2 * muad2**2
    A2 = aux2 * rho * J2Dij
    dA2 = aux2 * (J2Dij + rho * dJ2Dij * deta)

    J3Dijk, dJ3Dijk = np.tensordot(etavec, cnijk, axes=((1), (0)))
    aux3 = - 4 * np.pi**2 / 3 * epsa**3 * sigma3**2 * npol**3 * muad2**3
    A3 = aux3 * rho**2 * J3Dijk
    dA3 = aux3 * rho * (2 * J3Dijk + rho * dJ3Dijk * deta)

    Apolar = A2 / (1 - A3/A2)

    dApolar = A2 * ((A2 - 2 * A3) * dA2 + A2 * dA3)
    dApolar /= ((A2 - A3)**2)
    A = np.array([Apolar, dApolar])

    return A


def d2Apol_drho(rho, eta, deta, epsa, anij, bnij, cnijk, muad2, npol, sigma3):

    etavec = np.array([[1., eta, eta**2, eta**3, eta**4],
                      [0., 1., 2*eta, 3*eta**2, 4*eta**3],
                      [0., 0., 2., 6*eta, 12*eta**2]])

    J2Dij, dJ2Dij, d2J2Dij = np.tensordot(etavec, anij + bnij * epsa,
                                          axes=((1), (0)))

    aux2 = -np.pi * epsa**2 * sigma3 * npol**2 * muad2**2
    A2 = aux2 * rho * J2Dij
    dA2 = aux2 * (J2Dij + rho * dJ2Dij * deta)
    d2A2 = aux2 * (2 * dJ2Dij * deta + rho * d2J2Dij * deta**2)

    J3Dijk, dJ3Dijk, d2J3Dijk = np.tensordot(etavec, cnijk, axes=((1), (0)))

    aux3 = - 4 * np.pi**2 / 3 * epsa**3 * sigma3**2 * npol**3 * muad2**3
    A3 = aux3 * rho**2 * J3Dijk
    dA3 = aux3 * rho * (2 * J3Dijk + rho * dJ3Dijk * deta)
    d2A3 = aux3 * (2 * J3Dijk + rho * (4*dJ3Dijk*deta+rho*d2J3Dijk*deta**2))

    Apolar = A2 / (1 - A3/A2)

    dApolar = A2 * ((A2 - 2 * A3) * dA2 + A2 * dA3)
    dApolar /= ((A2 - A3)**2)

    d2Apolar = 2 * A3**2 * (dA2**2 + A2 * d2A2)
    d2Apolar += A2**2 * (2 * dA3**2 + A2 * (d2A2 + d2A3))
    d2Apolar -= A2 * A3 * (4 * dA2 * dA3 + A2 * (3*d2A2 + d2A3))
    d2Apolar /= ((A2 - A3)**3)

    A = np.array([Apolar, dApolar, d2Apolar])
    return A
