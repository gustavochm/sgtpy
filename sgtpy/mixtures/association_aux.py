import numpy as np
from scipy.optimize import fsolve


def association_config(sitesmix, eos):

    compindex = []
    for i, j in enumerate(sitesmix):
        compindex += np.count_nonzero(j) * [i]
    compindex = np.asarray(compindex)

    types = np.array([['B', 'P', 'N']])
    n = len(sitesmix)
    n = eos.nc
    types = np.repeat(types, n, axis=0)
    nozero = np.nonzero(sitesmix)
    types = types[nozero]
    ntypes = np.asarray(sitesmix)[nozero]
    nsites = len(types)

    S = np.array(sitesmix)
    S = S[S != 0]

    DIJ = np.zeros([nsites, nsites])
    int_i = []
    int_j = []
    for i in range(nsites):
        for j in range(nsites):
            bool1 = types[i] == 'B'
            bool2 = types[i] == 'P' and (types[j] == 'N' or types[j] == 'B')
            bool3 = types[i] == 'N' and (types[j] == 'P' or types[j] == 'B')
            if bool1 or bool2 or bool3:
                DIJ[i, j] = ntypes[j]
                int_i.append(i)
                int_j.append(j)

    indexabij = (int_i, int_j)
    indexab = (compindex[int_i], compindex[int_j])
    diagasso = np.diag_indices(nsites)

    dxjdx = np.zeros([eos.nc, nsites])
    for i in range(eos.nc):
        dxjdx[i] = compindex == i

    return S, DIJ, compindex, indexabij, indexab, nsites, dxjdx, diagasso


def Iab(xhi, dii, dij, rcij, rdij, sigmaij3):

    xhi0, xhi1, xhi2, xhi3 = xhi
    aux = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)

    xhi13 = 1 - xhi3

    gdhs = 1 / xhi13
    gdhs += 3 * aux * xhi2 / xhi13**2
    gdhs += 2 * aux**2 * xhi2**2 / xhi13**3

    Kab = np.log((rcij + 2*rdij)/dij)
    Kab *= 6*rcij**3 + 18 * rcij**2*rdij - 24 * rdij**3
    aux1 = (rcij + 2 * rdij - dij)
    aux2 = (22*rdij**2 - 5*rcij*rdij - 7*rdij*dij - 8*rcij**2+rcij*dij+dij**2)
    Kab += aux1 * aux2
    Kab /= (72*rdij**2 * sigmaij3)
    Kab *= 4 * np.pi * dij**2

    iab = gdhs * Kab
    return iab


def dIab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii, dij, rcij, rdij, sigmaij3):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    aux = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)

    xhi13 = 1 - xhi3

    gdhs = 1 / xhi13
    gdhs += 3 * aux * xhi2 / xhi13**2
    gdhs += 2 * aux**2 * xhi2**2 / xhi13**3

    dgdhs = 4. * aux**2 * xhi2 * dxhi2 / xhi13**3
    dgdhs += 3. * aux * dxhi2 / xhi13**2
    dgdhs += 6. * aux**2 * xhi2**2 * dxhi3 / xhi13**4
    dgdhs += 6. * aux * xhi2 * dxhi3 / xhi13**3
    dgdhs += dxhi3 / xhi13**2
    dgdhs *= dxhi00_drho

    Kab = np.log((rcij + 2*rdij)/dij)
    Kab *= 6*rcij**3 + 18 * rcij**2*rdij - 24 * rdij**3
    aux1 = (rcij + 2 * rdij - dij)
    aux2 = (22*rdij**2 - 5*rcij*rdij - 7*rdij*dij - 8*rcij**2+rcij*dij+dij**2)
    Kab += aux1 * aux2
    Kab /= (72*rdij**2 * sigmaij3)
    Kab *= 4 * np.pi * dij**2
    iab = gdhs * Kab
    diab = dgdhs * Kab
    return iab, diab


def d2Iab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii, dij, rcij, rdij, sigmaij3):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    aux = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)

    xhi13 = 1 - xhi3

    gdhs = 1 / xhi13
    gdhs += 3 * aux * xhi2 / xhi13**2
    gdhs += 2 * aux**2 * xhi2**2 / xhi13**3

    dgdhs = 4. * aux**2 * xhi2 * dxhi2 / xhi13**3
    dgdhs += 3. * aux * dxhi2 / xhi13**2
    dgdhs += 6. * aux**2 * xhi2**2 * dxhi3 / xhi13**4
    dgdhs += 6. * aux * xhi2 * dxhi3 / xhi13**3
    dgdhs += dxhi3 / xhi13**2
    dgdhs *= dxhi00_drho

    d2gdhs = 2 * dxhi3**2 / xhi13**3
    d2gdhs += 2 * aux**2 * 2 * dxhi2**2 / xhi13**3
    d2gdhs += 2 * aux**2 * 12 * xhi2 * dxhi2 * dxhi3 / xhi13**4
    d2gdhs += 2 * aux**2 * 12 * xhi2**2 * dxhi3**2 / xhi13**5
    d2gdhs += 3 * aux * 4 * dxhi2 * dxhi3 / xhi13**3
    d2gdhs += 3 * aux * 6 * xhi2 * dxhi3**2 / xhi13**4
    d2gdhs *= dxhi00_drho**2

    Kab = np.log((rcij + 2*rdij)/dij)
    Kab *= 6*rcij**3 + 18 * rcij**2*rdij - 24 * rdij**3
    aux1 = (rcij + 2 * rdij - dij)
    aux2 = (22*rdij**2 - 5*rcij*rdij - 7*rdij*dij-8*rcij**2+rcij*dij+dij**2)
    Kab += aux1 * aux2
    Kab /= (72*rdij**2 * sigmaij3)
    Kab *= 4 * np.pi * dij**2

    iab = gdhs * Kab
    diab = dgdhs * Kab
    d2iab = d2gdhs * Kab

    return iab, diab, d2iab


def dIab_dx(xhi, dxhi_dx, dii, dij, rcij, rdij, sigmaij3):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0x, dxhi1x, dxhi2x, dxhi3x = dxhi_dx

    aux = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)

    xhi13 = 1 - xhi3

    gdhs = 1 / xhi13
    gdhs += 3 * aux * xhi2 / xhi13**2
    gdhs += 2 * aux**2 * xhi2**2 / xhi13**3

    dgdhsx = 4 * np.multiply.outer(aux**2, dxhi2x) * xhi2 / (1-xhi3)**3
    dgdhsx += 3 * np.multiply.outer(aux, dxhi2x) / (1-xhi3)**2
    dgdhsx += 6 * np.multiply.outer(aux**2, dxhi3x) * xhi2**2 / (1-xhi3)**4
    dgdhsx += 6 * np.multiply.outer(aux, dxhi3x) * xhi2 / (1-xhi3)**3
    dgdhsx += dxhi3x / (1-xhi3)**2
    dgdhsx = dgdhsx.T

    Kab = np.log((rcij + 2*rdij)/dij)
    Kab *= 6*rcij**3 + 18 * rcij**2*rdij - 24 * rdij**3
    aux1 = (rcij + 2 * rdij - dij)
    aux2 = (22*rdij**2 - 5*rcij*rdij - 7*rdij*dij - 8*rcij**2+rcij*dij+dij**2)
    Kab += aux1 * aux2
    Kab /= (72*rdij**2 * sigmaij3)
    Kab *= 4 * np.pi * dij**2

    iab = gdhs * Kab
    diabx = dgdhsx * Kab
    return iab, diabx


def dIab_dxrho(xhi, dxhi_dxhi00, dxhi00_drho, dxhi_dx, dii, dij, rcij, rdij,
               sigmaij3):

    xhi0, xhi1, xhi2, xhi3 = xhi
    dxhi0, dxhi1, dxhi2, dxhi3 = dxhi_dxhi00
    dxhi0x, dxhi1x, dxhi2x, dxhi3x = dxhi_dx
    aux = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)

    xhi13 = 1 - xhi3

    gdhs = 1 / xhi13
    gdhs += 3 * aux * xhi2 / xhi13**2
    gdhs += 2 * aux**2 * xhi2**2 / xhi13**3

    dgdhs = 4. * aux**2 * xhi2 * dxhi2 / xhi13**3
    dgdhs += 3. * aux * dxhi2 / xhi13**2
    dgdhs += 6. * aux**2 * xhi2**2 * dxhi3 / xhi13**4
    dgdhs += 6. * aux * xhi2 * dxhi3 / xhi13**3
    dgdhs += dxhi3 / xhi13**2
    dgdhs *= dxhi00_drho

    dgdhsx = 4 * np.multiply.outer(aux**2, dxhi2x) * xhi2 / (1-xhi3)**3
    dgdhsx += 3 * np.multiply.outer(aux, dxhi2x) / (1-xhi3)**2
    dgdhsx += 6 * np.multiply.outer(aux**2, dxhi3x) * xhi2**2 / (1-xhi3)**4
    dgdhsx += 6 * np.multiply.outer(aux, dxhi3x) * xhi2 / (1-xhi3)**3
    dgdhsx += dxhi3x / (1-xhi3)**2
    dgdhsx = dgdhsx.T

    Kab = np.log((rcij + 2*rdij)/dij)
    Kab *= 6*rcij**3 + 18 * rcij**2*rdij - 24 * rdij**3
    aux1 = (rcij + 2 * rdij - dij)
    aux2 = (22*rdij**2 - 5*rcij*rdij - 7*rdij*dij - 8*rcij**2+rcij*dij+dij**2)
    Kab += aux1 * aux2
    Kab /= (72*rdij**2 * sigmaij3)
    Kab *= 4 * np.pi * dij**2

    iab = gdhs * Kab
    diab = dgdhs * Kab
    diabx = dgdhsx * Kab
    return iab, diab, diabx


def fobj_xass(Xass, xj, aux_asso, diagasso):
    fo = Xass - 1 / (1 + aux_asso@(xj*Xass))
    return fo


def fobj_xass_jac(Xass, xj, aux_asso, diagasso):
    den = 1 + aux_asso@(xj*Xass)
    dfo = ((aux_asso*xj).T/den**2).T
    dfo[diagasso] += 1.
    return dfo


def Xass_solver(nsites, xj, rho, DIJ, Dabij, diagasso, Xass0):

    Xass = 1. * Xass0

    aux_asso = rho * DIJ * Dabij

    omega = 0.2
    for i in range(5):
        den = 1. + aux_asso@(xj*Xass)
        fo = 1. / den
        dXass = (1 - omega) * (fo - Xass)
        Xass += dXass

    bool_method = not np.any(xj == 0.)

    if bool_method:
        KIJ = np.outer(xj, xj) * aux_asso

        KIJXass = KIJ@Xass
        dQ = xj * (1/Xass - 1) - KIJXass
        HIJ = -1 * KIJ
        HIJ[diagasso] -= (xj + KIJXass)/Xass
        for i in range(15):
            dXass = np.linalg.solve(HIJ, -dQ)
            Xnew = Xass + dXass
            Xnew_neg = Xnew < 0
            Xnew[Xnew_neg] = 0.2*Xass[Xnew_neg]
            Xass = Xnew
            KIJXass = KIJ@Xass
            dQ = xj * (1/Xass - 1) - KIJXass
            sucess = np.linalg.norm(dQ) < 1e-8
            if sucess:
                break
            HIJ = -1 * KIJ
            HIJ[diagasso] -= (xj + KIJXass)/Xass
        '''
        for i in range(15):
            KIJXass = KIJ@Xass
            dQ = xj * (1/Xass - 1) - KIJXass
            HIJ = -1 * KIJ
            HIJ[diagasso] -= (xj + KIJXass)/Xass
            dXass = np.linalg.solve(HIJ, -dQ)
            Xass += dXass
            sucess = np.linalg.norm(dXass) < 1e-8
            if sucess:
                break
         '''
    else:
        Xass = fsolve(fobj_xass, x0=Xass, args=(xj, aux_asso, diagasso),
                      fprime=fobj_xass_jac)

    return Xass


def CIJ_matrix(rhom, xj, Xass, DIJ, Dabij, diagasso):
    CIJ = rhom * np.outer(Xass**2, xj) * Dabij * DIJ
    CIJ[diagasso] += 1.
    return CIJ


def dXass_drho(rhom, xj, Xass, DIJ, Dabij, dDabij_drho, CIJ):
    brho = -(DIJ*(Dabij + rhom * dDabij_drho))@(xj * Xass)
    brho *= Xass**2
    dXass = np.linalg.solve(CIJ, brho)
    return dXass


def d2Xass_drho(rhom, xj, Xass, dXass_drho, DIJ, Dabij, dDabij_drho,
                d2Dabij_drho, CIJ):

    b2rho = -rhom * (DIJ * (Xass * d2Dabij_drho+2*dXass_drho*dDabij_drho))@xj
    b2rho += 2 * (1/Xass - 1) / (rhom**2)
    b2rho *= Xass**2
    b2rho += 2 * dXass_drho / (rhom)
    b2rho += 2 * dXass_drho**2 / (Xass)

    d2Xass = np.linalg.solve(CIJ, b2rho)
    return d2Xass


def dXass_dx(rhom, xj, Xass, DIJ, Dabij, dDabij_dx, dxjdx, CIJ):

    bx = (dxjdx * Xass)@(DIJ*Dabij).T + (DIJ * dDabij_dx)@(xj * Xass)
    bx *= Xass**2
    bx *= -rhom
    dXass = np.linalg.solve(CIJ, bx.T)

    return dXass.T
