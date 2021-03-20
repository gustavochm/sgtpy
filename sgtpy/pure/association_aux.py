import numpy as np


def association_config(eos):

    types = np.array(['B', 'P', 'N'])
    nozero = np.nonzero(eos.sites)
    types = types[nozero]
    ntypes = np.asarray(eos.sites)[nozero]
    nsites = len(types)
    S = np.array(eos.sites)
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
    diagasso = np.diag_indices(nsites)

    return S, DIJ, indexabij, nsites, diagasso


def Xass_solver(nsites, KIJ, diagasso, Xass0=None):

    if Xass0 is None:
        Xass = 0.2 * np.ones(nsites)
    else:
        Xass = 1. * Xass0
    omega = 0.2

    for i in range(5):
        fo = 1. / (1. + KIJ@Xass)
        dXass = (1 - omega) * (fo - Xass)
        Xass += dXass

    for i in range(15):
        KIJXass = KIJ@Xass
        dQ = (1./Xass - 1.) - KIJXass
        HIJ = - 1. * KIJ
        HIJ[diagasso] -= (1. + KIJXass)/Xass
        dXass = np.linalg.solve(HIJ, -dQ)
        Xass += dXass
        sucess = np.linalg.norm(dXass) < 1e-9
        if sucess:
            break

    return Xass


def Iab(Kab, eta):

    gdhs = (1 - eta/2) / (1 - eta)**3
    Iab = Kab * gdhs

    return Iab


def dIab_drho(Kab, eta, deta_drho):

    eta_1 = 1-eta
    eta_13 = eta_1**3
    eta_14 = eta_13*eta_1

    gdhs = (1 - eta/2) / eta_13
    dgdhs = (2.5 - eta) * deta_drho / eta_14

    Iab = Kab * gdhs
    dIab = Kab * dgdhs
    return Iab, dIab


def d2Iab_drho(Kab, eta, deta_drho):
    eta_1 = 1-eta
    eta_13 = eta_1**3
    eta_14 = eta_13*eta_1
    eta_15 = eta_14*eta_1

    gdhs = (1 - eta/2) / eta_13
    dgdhs = (2.5 - eta) * deta_drho / eta_14
    d2gdhs = 3 * (3 - eta) * deta_drho**2 / eta_15

    Iab = Kab * gdhs
    dIab = Kab * dgdhs
    d2Iab = Kab * d2gdhs
    return Iab, dIab, d2Iab


def dXass_drho(rho, Xass, DIJ, Dabij, dDabij_drho, CIJ):
    brho = -(DIJ*(Dabij + rho * dDabij_drho))@Xass
    brho *= Xass**2
    dXass = np.linalg.solve(CIJ, brho)
    return dXass


def d2Xass_drho(rho, Xass, dXass_drho, DIJ, Dabij, dDabij_drho, d2Dabij_drho,
                CIJ):

    b2rho = np.sum(DIJ*(Xass*d2Dabij_drho + 2*dXass_drho*dDabij_drho), axis=1)
    b2rho *= - rho
    b2rho += 2 * (1/Xass - 1) / (rho**2)
    b2rho *= Xass**2
    b2rho += 2 * dXass_drho / (rho)
    b2rho += 2 * dXass_drho**2 / (Xass)

    d2Xass_drho = np.linalg.solve(CIJ, b2rho)
    return d2Xass_drho
