from __future__ import division, print_function, absolute_import
import numpy as np
from numba import jit
from scipy.optimize import fsolve


cpq = np.zeros([11, 11])
cpq[0, 0] = 7.56425183020431E-02
cpq[0, 1] = -1.28667137050961E-01
cpq[0, 2] = 1.28350632316055E-01
cpq[0, 3] = -7.25321780970292E-02
cpq[0, 4] = 2.57782547511452E-02
cpq[0, 5] = -6.01170055221687E-03
cpq[0, 6] = 9.33363147191978E-04
cpq[0, 7] = -9.55607377143667E-05
cpq[0, 8] = 6.19576039900837E-06
cpq[0, 9] = -2.30466608213628E-07
cpq[0, 10] = 3.74605718435540E-09
cpq[1, 0] = 1.34228218276565E-01
cpq[1, 1] = -1.82682168504886E-01
cpq[1, 2] = 7.71662412959262E-02
cpq[1, 3] = -7.17458641164565E-04
cpq[1, 4] = -8.72427344283170E-03
cpq[1, 5] = 2.97971836051287E-03
cpq[1, 6] = -4.84863997651451E-04
cpq[1, 7] = 4.35262491516424E-05
cpq[1, 8] = -2.07789181640066E-06
cpq[1, 9] = 4.13749349344802E-08
cpq[2, 0] = -5.65116428942893E-01
cpq[2, 1] = 1.00930692226792E+00
cpq[2, 2] = -6.60166945915607E-01
cpq[2, 3] = 2.14492212294301E-01
cpq[2, 4] = -3.88462990166792E-02
cpq[2, 5] = 4.06016982985030E-03
cpq[2, 6] = -2.39515566373142E-04
cpq[2, 7] = 7.25488368831468E-06
cpq[2, 8] = -8.58904640281928E-08
cpq[3, 0] = -3.87336382687019E-01
cpq[3, 1] = -2.11614570109503E-01
cpq[3, 2] = 4.50442894490509E-01
cpq[3, 3] = -1.76931752538907E-01
cpq[3, 4] = 3.17171522104923E-02
cpq[3, 5] = -2.91368915845693E-03
cpq[3, 6] = 1.30193710011706E-04
cpq[3, 7] = -2.14505500786531E-06
cpq[4, 0] = 2.13713180911797E+00
cpq[4, 1] = -2.02798460133021E+00
cpq[4, 2] = 3.36709255682693E-01
cpq[4, 3] = 1.18106507393722E-03
cpq[4, 4] = -6.00058423301506E-03
cpq[4, 5] = 6.26343952584415E-04
cpq[4, 6] = -2.03636395699819E-05
cpq[5, 0] = -3.00527494795524E-01
cpq[5, 1] = 2.89920714512243E+00
cpq[5, 2] = -5.67134839686498E-01
cpq[5, 3] = 5.18085125423494E-02
cpq[5, 4] = -2.39326776760414E-03
cpq[5, 5] = 4.15107362643844E-05
cpq[6, 0] = -6.21028065719194E+00
cpq[6, 1] = -1.92883360342573E+00
cpq[6, 2] = 2.84109761066570E-01
cpq[6, 3] = -1.57606767372364E-02
cpq[6, 4] = 3.68599073256615E-04
cpq[7, 0] = 1.16083532818029E+01
cpq[7, 1] = 7.42215544511197E-01
cpq[7, 2] = -8.23976531246117E-02
cpq[7, 3] = 1.86167650098254E-03
cpq[8, 0] = -1.02632535542427E+01
cpq[8, 1] = -1.25035689035085E-01
cpq[8, 2] = 1.14299144831867E-02
cpq[9, 0] = 4.65297446837297E+00
cpq[9, 1] = -1.92518067137033E-03
cpq[10, 0] = -8.67296219639940E-01


@jit(cache=True)
def Iab(rho_ad, T_ad, Iijklab):
    for p in range(11):
        for q in range(0, 11-p):
            Iijklab += cpq[p, q] * rho_ad**p * T_ad**q
            pass
    return Iijklab


@jit(cache=True)
def dIab_drho(rho_ad, T_ad, drho_ad, Iijklab, dIijklab_drho):
    for p in range(11):
        for q in range(0, 11-p):
            Iijklab += cpq[p, q] * rho_ad**p * T_ad**q
            dIijklab_drho += cpq[p, q] * p * rho_ad**(p-1) * T_ad**q
    dIijklab_drho *= drho_ad
    return Iijklab, dIijklab_drho


@jit(cache=True)
def d2Iab_drho(rho_ad, T_ad, drho_ad, Iijklab, dIijklab_drho, d2Iijklab_drho):
    for p in range(11):
        for q in range(0, 11-p):
            Iijklab += cpq[p, q] * rho_ad**p * T_ad**q
            dIijklab_drho += cpq[p, q] * p * rho_ad**(p-1) * T_ad**q
            d2Iijklab_drho += cpq[p, q] * p * (p-1) * rho_ad**(p-2) * T_ad**q
    dIijklab_drho *= drho_ad
    d2Iijklab_drho *= drho_ad**2
    return Iijklab, dIijklab_drho, d2Iijklab_drho


@jit(cache=True)
def dIab(rho_ad, T_ad, Iijklab, dIijklab):
    for p in range(11):
        for q in range(0, 11-p):
            Iijklab += cpq[p, q] * rho_ad**p * T_ad**q
            dIijklab += cpq[p, q] * p * rho_ad**(p-1) * T_ad**q
    return Iijklab


def fobj_xass(Xass, xjvk, aux_asso, diagasso):
    fo = Xass - 1 / (1 + aux_asso@(xjvk*Xass))
    return fo


def fobj_xass_jac(Xass, xjvk, aux_asso, diagasso):
    den = 1 + aux_asso@(xjvk*Xass)
    dfo = ((aux_asso*xjvk).T/den**2).T
    dfo[diagasso] += 1.
    return dfo


def Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass0):

    omega = 0.2
    Xass = 1.*Xass0

    aux_asso = rho * DIJ * Dijklab

    for i in range(5):
        den = 1. + aux_asso@(xjvk*Xass)
        fo = 1. / den
        dXass = (1 - omega) * (fo - Xass)
        Xass += dXass

    bool_method = not np.any(xjvk == 0.)

    if bool_method:

        KIJ = np.outer(xjvk, xjvk) * aux_asso

        KIJXass = KIJ@Xass
        dQ = xjvk * (1/Xass - 1) - KIJXass
        HIJ = -1 * KIJ
        HIJ[diagasso] -= (xjvk + KIJXass)/Xass
        for i in range(15):
            dXass = np.linalg.solve(HIJ, -dQ)
            Xnew = Xass + dXass

            is_nan = np.isnan(Xnew)
            Xnew[is_nan] = 0.2

            Xnew_neg = Xnew < 0
            Xnew[Xnew_neg] = 0.2*Xass[Xnew_neg]
            Xass = Xnew
            KIJXass = KIJ@Xass
            dQ = xjvk * (1/Xass - 1) - KIJXass
            sucess = np.linalg.norm(dQ) < 1e-9
            if sucess:
                break
            HIJ = -1 * KIJ
            HIJ[diagasso] -= (xjvk + KIJXass)/Xass
    else:
        Xass = fsolve(fobj_xass, x0=Xass, args=(xjvk, aux_asso, diagasso),
                      fprime=fobj_xass_jac)

    return Xass


def CIJ_matrix(rhom, xjvk, Xass, DIJ, Dabij, diagasso):
    CIJ = rhom * np.outer(Xass**2, xjvk) * Dabij * DIJ
    CIJ[diagasso] += 1.
    return CIJ


def dXass_drho(rhom, xjvk, Xass, DIJ, Dabij, dDabij_drho, CIJ):
    brho = -(DIJ*(Dabij + rhom * dDabij_drho))@(xjvk * Xass)
    brho *= Xass**2
    dXass = np.linalg.solve(CIJ, brho)
    return dXass


def d2Xass_drho(rhom, xjvk, Xass, dXass_drho, DIJ, Dabij, dDabij_drho,
                d2Dabij_drho, CIJ):

    b2rho = -rhom * (DIJ * (Xass * d2Dabij_drho+2*dXass_drho*dDabij_drho))@xjvk
    b2rho += 2 * (1/Xass - 1) / (rhom**2)
    b2rho *= Xass**2
    b2rho += 2 * dXass_drho / (rhom)
    b2rho += 2 * dXass_drho**2 / (Xass)

    d2Xass = np.linalg.solve(CIJ, b2rho)
    return d2Xass


def dXass_dx(rhom, xjvk, Xass, DIJ, Dabij, dDabij_dx, dxjdx, CIJ):

    bx = (dxjdx * Xass)@(DIJ*Dabij).T + (DIJ * dDabij_dx)@(xjvk * Xass)
    bx *= Xass**2
    bx *= -rhom
    dXass = np.linalg.solve(CIJ, bx.T)

    return dXass.T


def association_solver(self, x, rhom, temp_aux, Xass0=None):
    if Xass0 is None:
        Xass = 0.2 * np.ones(self.nsites)
    else:
        Xass = 1. * Xass0

    # setting up component
    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    x_k = x[self.groups_index]
    diagasso = self.diagasso
    vki_asso = self.vki_asso
    DIJ = self.DIJ

    xs_ki = x_k*Sk*vki*vk
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    T_ad = temp_aux[29]
    sigma_kl3 = self.sigma_kl3
    sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
    rho_ad = rhom * xs_m * sigma_x3

    Iijklab = np.zeros([self.nc, self.nc])
    Iab(rho_ad, T_ad, Iijklab)

    # vki_asso = self.vki[self.group_asso_index]
    vki_asso = self.vki_asso
    DIJ = self.DIJ
    xj_asso = x[self.molecule_id_index_sites]
    xjvk = xj_asso*vki_asso

    # Fklab = np.exp(self.epsAB_kl * beta) - 1
    Fklab = temp_aux[30]
    Dijklab = self.kAB_kl * Fklab
    Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

    Xass = Xass_solver(rhom, xjvk, DIJ, Dijklab, diagasso, Xass)
    return Xass


def assocation_check(self, x, rhom, temp_aux, Xass):
    # setting up component
    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    x_k = x[self.groups_index]
    vki_asso = self.vki_asso
    DIJ = self.DIJ

    xs_ki = x_k*Sk*vki*vk
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    T_ad = temp_aux[29]
    sigma_kl3 = self.sigma_kl3
    sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
    rho_ad = rhom * xs_m * sigma_x3

    Iijklab = np.zeros([self.nc, self.nc])
    Iab(rho_ad, T_ad, Iijklab)

    # vki_asso = self.vki[self.group_asso_index]
    vki_asso = self.vki_asso
    DIJ = self.DIJ
    xj_asso = x[self.molecule_id_index_sites]
    xjvk = xj_asso*vki_asso

    # Fklab = np.exp(self.epsAB_kl * beta) - 1
    Fklab = temp_aux[30]
    Dijklab = self.kAB_kl * Fklab
    Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

    aux_asso = rhom * DIJ * Dijklab
    fo = Xass - 1 / (1 + aux_asso@(xjvk*Xass))
    return fo
