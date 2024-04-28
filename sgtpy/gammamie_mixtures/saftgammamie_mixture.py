from __future__ import division, print_function, absolute_import
import numpy as np
from ..constants import kb, Na
from ..math import gauss
from .a1sB_monomer import x0lambda_evalm, x0lambda_evalc
from .monomer_aux import I_lam, J_lam
from .ares import ares, dares_drho, d2ares_drho
from .ares import dares_dx, dares_dx_drho
from .ideal import aideal, daideal_drho, d2aideal_drho
from .ideal import daideal_dx, daideal_dx_drho
from .density_solver import density_topliss, density_newton
from .association_aux import assocation_check, association_solver

from ..gammamie_pure import saftgammamie_pure

R = kb * Na


def U_mie(r, c, eps, lambda_r, lambda_a):
    u = c * eps * (np.power.outer(r, lambda_r) - np.power.outer(r, lambda_a))
    return u


# Second perturbation
phi16 = np.array([[7.5365557, -37.60463, 71.745953, -46.83552, -2.467982,
                 -0.50272, 8.0956883],
                 [-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.7090],
                 [1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0],
                 [-1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377,
                 40.53683],
                 [-1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61,
                 2556.181],
                 [9236.9, -129430., 357230., -315530., 1390.2, -4518.2,
                 4241.6]])


nfi = np.arange(0, 7)
nfi_num = nfi[:4]
nfi_den = nfi[4:]


# Equation A26
def fi(alphaij, i):
    phi = phi16[i-1]
    num = np.tensordot(np.power.outer(alphaij, nfi_num), phi[nfi_num],
                       axes=(-1, -1))
    den = 1 + np.tensordot(np.power.outer(alphaij, nfi_den - 3), phi[nfi_den],
                           axes=(-1, -1))
    return num/den


class saftgammamie_mix():
    '''
    Fluid mixtures SAFT-Gamma-Mie EoS Object

    This object have implemeted methods for phase equilibrium
    as for interfacial properties calculations.

    Parameters
    ----------
    mixture : object
        fluid mixture created with mixture class

    Attributes
    ----------
    vki: array, number of groups i
    subgroups: array, groups present in the mixture
    ngroups: int, number of groups in the fluid

    vk: array, number of sites of each group
    Sk: array, shape factor o of each group

    lr_kl: array, repulsive exponent used in monomer term
    la_kl: array, attractive exponent used in monomer term
    lar_kl: array, sum of repulsive and attractive exponents
    eps_kl: array, energy interactions used in monomer term [J]
    sigma_kl: array, lenght scale used in monomer term [m]
    Ckl: array, prefactor of mie potential used in monomer term
    alphakl: array, van der waals constant used in monomer term

    lr_kk: array, repulsive exponent used in monomer term
    la_kk: array, attractive exponent used in monomer term
    lar_kk: array, sum of repulsive and attractive exponents
    eps_kk: array, energy interactions used in monomer term [J]
    sigma_kk: array, lenght scale used in monomer term [m]
    Ckk: array, prefactor of mie potential used in monomer term
    alphakk: array, van der waals constant used in monomer term


    zs_k: array, fraction of group k in each molecule

    lr_ii: array, repulsive exponent used in chain term
    la_ii: array, attractive exponent used in chain term
    lar_ii: array, sum of repulsive and attractive exponents
    eps_ii: array, energy interactions used in chain term [J]
    sigma_ii: array, lenght scale used in chain term [m]
    Cii: array, prefactor of mie potential used in chain term
    alphaii: array, van der waals constant used in chain term

    eAB_kl: array, association energy [J]
    kAB_kl: array, association volume [m]

    cii : array_like
        influence factor for SGT [J m^5 / mol^2]
    cij : array_like
        cross influence parameter matrix for SGT [J m^5 / mol^2]
    beta: array_like
        correction to cross influence parameter matrix

    secondorder: bool
        bool to indicate if composition derivatives of fugacity coefficient
        are available
    secondordersgt: bool
        bool to indicate if composition derivatives of chemical potential
        are available


    Methods
    -------
    cii_correlation : correlates the influence parameter of the fluid
    diameter : computes the diameter at given temperature
    temperature_aux : computes temperature depedent parameters of the fluid
    association_solver: computes the fraction of non-bonded sites
    association_check: checks if the association sites are consistent
    ares: computes the residual dimentionless Helmholtz free energy
    dares_drho: computes the residual dimentionless Helmholtz free energy
                and its density first density derivative
    d2ares_drho: computes the residual dimentionless Helmholtz free energy
                 and its density first and second density derivatives
    dares_dx: computes the residual dimentionless Helmholtz free energy and
              its composition derivatives
    dares_dxrho: computes the residual dimentionless Helmholtz free energy
                 and its composition and density derivatives
    afcn: computes total Helmholtz energy
    dafcn_drho: computes total Helmholtz energy and its density derivative
    d2afcn_drho: computes total Helmholtz energy and it density derivatives
    dafcn_dx: computes total Helmholtz energy and its composition derivative
    dafcn_dxrho:computes total Helmholtz energy and its composition and
                density derivative
    density: computes the density of the fluid
    pressure: computes the pressure
    dP_drho: computes pressure and its density derivative
    logfugmix: computes the fugacity coefficient of the mixture
    logfugef: computes the effective fugacity coefficients of the components
              in the mixture
    a0ad: computes adimentional Helmholtz density energy
    muad: computes adimentional chemical potential
    dmuad: computes the adimentional chemical potential and its derivatives
    dOm : computes adimentional Thermodynamic Grand Potential
    ci :  computes influence parameters matrix for SGT
    sgt_adim : computes adimentional factors for SGT
    beta_sgt: method for setting beta correction used in SGT

    association_solver: computes the fraction of non-bonded sites
    association_check: checks the fraction of non-bonded sites solution

    EntropyR : computes the residual entropy of the fluid
    EnthalpyR : computes the residual enthalpy of the fluid
    CvR : computes the residual isochoric heat capacity
    CpR : computes the residual heat capacity
    speed_sound : computes the speed of sound

    Auxiliar methods (computed using temperature_aux output list)
    -------------------------------------------------------------
    density_aux : computes density
    afcn_aux : computes afcn
    dafcn_aux : computes dafcn_drho
    d2afcn_aux : computes d2afcn_drho
    pressure_aux : computes pressure
    dP_drho_aux : computes dP_drho
    logfugmix_aux : computes logfug
    a0ad_aux : compute a0ad
    muad_aux : computes muad
    dmuad_aux : computes dmuad
    dOm_aux : computes dOm
    '''
    def __init__(self, mixture, compute_critical=True):

        self.nc = mixture.nc
        self.mixture = mixture
        self.mw_kk = mixture.mw_kk
        self.Mw = mixture.Mw

        self.vki = mixture.vki
        self.subgroups = mixture.subgroups
        self.ngroups = mixture.ngroups
        self.ngtotal = mixture.ngtotal
        self.groups_index = mixture.groups_index
        self.groups_indexes = mixture.groups_indexes

        self.vk = mixture.vk
        self.Sk = mixture.Sk

        self.lr_kl = mixture.lr_kl
        self.la_kl = mixture.la_kl
        self.lar_kl = self.lr_kl + self.la_kl
        self.eps_kl = mixture.eps_kl * kb
        self.sigma_kl = mixture.sigma_kl * 1e-10
        self.sigma_kl3 = self.sigma_kl**3

        self.lr_kk = mixture.lr_kk
        self.la_kk = mixture.la_kk
        self.lar_kk = self.lr_kk + self.la_kk
        self.eps_kk = mixture.eps_kk * kb
        self.sigma_kk = mixture.sigma_kk * 1e-10
        self.sigma_kk3 = self.sigma_kk**3

        lr_kl, la_kl, lar_kl = self.lr_kl, self.la_kl, self.lar_kl
        dif_ckl = lr_kl - la_kl
        self.Ckl = lr_kl/dif_ckl*(lr_kl/la_kl)**(la_kl/dif_ckl)
        self.alphakl = self.Ckl * (1./(la_kl - 3.) - 1./(lr_kl - 3.))
        self.Ckl2 = self.Ckl**2

        self.diag_index = np.diag_indices(self.ngtotal)

        self.Ckk = self.Ckl[self.diag_index]
        self.alphakk = self.alphakl[self.diag_index]

        c_matrix = np.array([[0.81096, 1.7888, -37.578, 92.284],
                            [1.0205, -19.341, 151.26, -463.5],
                            [-1.9057, 22.845, -228.14, 973.92],
                            [1.0885, -6.1962, 106.98, -677.64]])

        lrkl_power = np.power.outer(lr_kl, [0, -1, -2, -3])
        cctes_lrkl = np.tensordot(lrkl_power, c_matrix, axes=(-1, -1))

        lakl_power = np.power.outer(la_kl, [0, -1, -2, -3])
        cctes_lakl = np.tensordot(lakl_power, c_matrix, axes=(-1, -1))

        lrkl_2power = np.power.outer(2*lr_kl, [0, -1, -2, -3])
        cctes_2lrkl = np.tensordot(lrkl_2power, c_matrix, axes=(-1, -1))

        lakl_2power = np.power.outer(2*la_kl, [0, -1, -2, -3])
        cctes_2lakl = np.tensordot(lakl_2power, c_matrix, axes=(-1, -1))

        larkl_power = np.power.outer(lar_kl, [0, -1, -2, -3])
        cctes_larkl = np.tensordot(larkl_power, c_matrix, axes=(-1, -1))

        # Monomer necessary term
        self.lambdaskl = (la_kl, lr_kl, 2*la_kl, 2*lr_kl, lar_kl)
        self.ccteskl = (cctes_lakl, cctes_lrkl, cctes_2lakl, cctes_2lrkl,
                        cctes_larkl)

        # For second and third order perturbation
        self.f1 = fi(self.alphakl, 1)
        self.f2 = fi(self.alphakl, 2)
        self.f3 = fi(self.alphakl, 3)
        self.f4 = fi(self.alphakl, 4)
        self.f5 = fi(self.alphakl, 5)
        self.f6 = fi(self.alphakl, 6)

        # Chain needed parameters
        nc = self.nc
        Sk = self.Sk
        vki = self.vki
        vk = self.vk

        maxindex = []
        for i in range(nc-1):
            maxindex.append(self.groups_indexes[i][1])

        zs_ki = Sk*vki*vk
        zs_k = np.split(zs_ki, maxindex)
        zs_m = np.zeros(nc)
        for i in range(nc):
            zs_m[i] = np.sum(zs_k[i])
            zs_k[i] /= zs_m[i]

        eps_ii = np.zeros(nc)
        la_ii = np.zeros(nc)
        lr_ii = np.zeros(nc)
        sigma_ii3 = np.zeros(nc)

        eps_kl = self.eps_kl
        sigma_kl3 = self.sigma_kl3

        for i in range(nc):
            i0, i1 = self.groups_indexes[i]
            eps_ii[i] = np.matmul(np.matmul(zs_k[i], eps_kl[i0:i1, i0:i1]), zs_k[i])
            la_ii[i] = np.matmul(np.matmul(zs_k[i], la_kl[i0:i1, i0:i1]), zs_k[i])
            lr_ii[i] = np.matmul(np.matmul(zs_k[i], lr_kl[i0:i1, i0:i1]), zs_k[i])
            sigma_ii3[i] = np.matmul(np.matmul(zs_k[i], sigma_kl3[i0:i1, i0:i1]), zs_k[i])

        lar_ii = la_ii + lr_ii

        self.zs_k = zs_k
        self.zs_m = zs_m
        self.eps_ii = eps_ii
        self.la_ii = la_ii
        self.lr_ii = lr_ii
        self.lar_ii = lar_ii
        self.sigma_ii3 = sigma_ii3
        self.sigma_ii = np.cbrt(self.sigma_ii3)

        dif_cii = lr_ii - la_ii
        self.Cii = lr_ii/dif_cii*(lr_ii/la_ii)**(la_ii/dif_cii)
        self.alphaii = self.Cii * (1./(la_ii - 3.) - 1./(lr_ii - 3.))
        self.Cii2 = self.Cii**2

        lrii_power = np.power.outer(lr_ii, [0, -1, -2, -3])
        cctes_lrii = np.tensordot(lrii_power, c_matrix, axes=(-1, -1))

        laii_power = np.power.outer(la_ii, [0, -1, -2, -3])
        cctes_laii = np.tensordot(laii_power, c_matrix, axes=(-1, -1))

        lrii_2power = np.power.outer(2*lr_ii, [0, -1, -2, -3])
        cctes_2lrii = np.tensordot(lrii_2power, c_matrix, axes=(-1, -1))

        laii_2power = np.power.outer(2*la_ii, [0, -1, -2, -3])
        cctes_2laii = np.tensordot(laii_2power, c_matrix, axes=(-1, -1))

        larii_power = np.power.outer(lar_ii, [0, -1, -2, -3])
        cctes_larii = np.tensordot(larii_power, c_matrix, axes=(-1, -1))

        # Chain necessary term
        self.lambdasii = (la_ii, lr_ii, 2*la_ii, 2*lr_ii, lar_ii)
        self.cctesii = (cctes_laii, cctes_lrii, cctes_2laii, cctes_2lrii,
                        cctes_larii)

        # For diameter calculation
        roots, weights = gauss(30)
        self.roots = roots
        self.weights = weights
        self.umie = U_mie(1./roots, self.Ckk, self.eps_kk, self.lr_kk,
                          self.la_kk)

        self.dxhi00_drho = np.pi / 6
        self.dxhi00_1 = np.array([1., self.dxhi00_drho])
        self.dxhi00_2 = np.array([1., self.dxhi00_drho, self.dxhi00_drho**2])

        self.asso_bool = mixture.asso_bool
        if self.asso_bool:
            self.S = mixture.S
            self.kAB_kl = mixture.kAB_kl * 1e-30
            self.epsAB_kl = mixture.epsAB_kl * kb
            self.DIJ = mixture.DIJ
            self.diagasso = mixture.diagasso
            self.nsites = mixture.nsites
            self.group_asso_index = mixture.group_asso_index
            self.molecule_id_index_sites = mixture.molecule_id_index_sites
            self.indexAB_id = mixture.indexAB_id
            self.indexABij = mixture.indexABij
            self.dxjasso_dx = mixture.dxjasso_dx
            self.vki_asso = mixture.vki_asso
            sigma_ij = np.add.outer(self.sigma_ii, self.sigma_ii) / 2
            eps_ij = np.sqrt(np.outer(eps_ii, eps_ii))
            eps_ij *= np.sqrt(np.outer(sigma_ii3, sigma_ii3))
            eps_ij /= sigma_ij**3
            self.eps_ij = eps_ij

        # for composition derivatives
        dxkdx = np.zeros([self.nc, self.ngtotal])
        for i in range(nc):
            dxkdx[i] = self.groups_index == i
        self.dxkdx = dxkdx

        # for SGT calculation
        self.cii = mixture.cii
        # self.cij = np.sqrt(np.outer(self.cii, self.cii))
        # self.beta = np.zeros([self.nc, self.nc])
        self.beta0 = np.zeros([self.nc, self.nc])
        self.beta1 = np.zeros([self.nc, self.nc])
        self.beta2 = np.zeros([self.nc, self.nc])
        self.beta3 = np.zeros([self.nc, self.nc])

        self.secondorder = False
        self.secondordersgt = True

        # creating pure fluid's eos
        pure_eos = []
        for i, component in enumerate(self.mixture.components):
            component.saftgammamie()
            model = saftgammamie_pure(component, compute_critical=compute_critical)
            pure_eos.append(model)
        self.pure_eos = pure_eos

    def diameter(self, beta):
        integrer = np.exp(-beta * self.umie)
        d = self.sigma_kk * (1. - np.matmul(self.weights, integrer))
        return d

    def temperature_aux(self, T):
        """
        temperature_aux(T)

        Method that computes temperature dependent parameters.
        It returns the following list:

        Parameters
        ----------

        T : float
            Absolute temperature [K]

        Returns
        -------
        temp_aux : list
             list of computed parameters
        """

        beta = 1 / (kb * T)
        beta2 = beta**2
        beta3 = beta2 * beta

        d_kk = self.diameter(beta)
        d_kl = np.add.outer(d_kk, d_kk)/2
        d_kl3 = d_kl**3
        d_kk03 = np.power.outer(d_kk, np.arange(4))

        x0_kl = self.sigma_kl / d_kl

        # for monomer contribution calculation
        a1vdw_ctekl = -12. * self.eps_kl * d_kl3

        la_kl, lr_kl, lar_kl = self.la_kl, self.lr_kl, self.lar_kl
        a1vdw_lakl = a1vdw_ctekl / (la_kl - 3)
        a1vdw_lrkl = a1vdw_ctekl / (lr_kl - 3)
        a1vdw_2lakl = a1vdw_ctekl / (2*la_kl - 3)
        a1vdw_2lrkl = a1vdw_ctekl / (2*lr_kl - 3)
        a1vdw_larkl = a1vdw_ctekl / (lar_kl - 3)
        a1vdwkl = (a1vdw_lakl, a1vdw_lrkl, a1vdw_2lakl, a1vdw_2lrkl,
                   a1vdw_larkl)
        # for a1 and a2 calculation in monomer contribution
        x0_a1, x0_a2 = x0lambda_evalm(x0_kl, la_kl, lr_kl, lar_kl)

        # I and J used B term in monomer contritubion
        I_lakl = I_lam(x0_kl, la_kl)
        I_lrkl = I_lam(x0_kl, lr_kl)
        I_2lakl = I_lam(x0_kl, 2*la_kl)
        I_2lrkl = I_lam(x0_kl, 2*lr_kl)
        I_larkl = I_lam(x0_kl, lar_kl)
        I_lambdaskl = (I_lakl, I_lrkl, I_2lakl, I_2lrkl, I_larkl)

        J_lakl = J_lam(x0_kl, la_kl)
        J_lrkl = J_lam(x0_kl, lr_kl)
        J_2lakl = J_lam(x0_kl, 2*la_kl)
        J_2lrkl = J_lam(x0_kl, 2*lr_kl)
        J_larkl = J_lam(x0_kl, lar_kl)
        J_lambdaskl = (J_lakl, J_lrkl, J_2lakl, J_2lrkl, J_larkl)

        # for chain contribution calculation
        nc = self.nc
        zs_k = self.zs_k
        d_ii3 = np.zeros(nc)
        for i in range(nc):
            i0, i1 = self.groups_indexes[i]
            d_ii3[i] = np.matmul(np.matmul(zs_k[i], d_kl3[i0:i1, i0:i1]), zs_k[i])
        d_ii = np.cbrt(d_ii3)
        x0_ii = self.sigma_ii/d_ii

        a1vdw_cteii = -12. * self.eps_ii * d_ii3
        la_ii, lr_ii, lar_ii = self.la_ii, self.lr_ii, self.lar_ii
        a1vdw_laii = a1vdw_cteii / (la_ii - 3)
        a1vdw_lrii = a1vdw_cteii / (lr_ii - 3)
        a1vdw_2laii = a1vdw_cteii / (2*la_ii - 3)
        a1vdw_2lrii = a1vdw_cteii / (2*lr_ii - 3)
        a1vdw_larii = a1vdw_cteii / (lar_ii - 3)
        a1vdwii = (a1vdw_laii, a1vdw_lrii, a1vdw_2laii, a1vdw_2lrii,
                   a1vdw_larii)
        # for a1, a2, g1 and g2 calculation in chain contribution
        out = x0lambda_evalc(x0_ii, la_ii, lr_ii, lar_ii)
        x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii = out

        # I and J used B term in monomer contritubion
        I_laii = I_lam(x0_ii, la_ii)
        I_lrii = I_lam(x0_ii, lr_ii)
        I_2laii = I_lam(x0_ii, 2*la_ii)
        I_2lrii = I_lam(x0_ii, 2*lr_ii)
        I_larii = I_lam(x0_ii, lar_ii)
        I_lambdasii = (I_laii, I_lrii, I_2laii, I_2lrii, I_larii)

        J_laii = J_lam(x0_ii, la_ii)
        J_lrii = J_lam(x0_ii, lr_ii)
        J_2laii = J_lam(x0_ii, 2*la_ii)
        J_2lrii = J_lam(x0_ii, 2*lr_ii)
        J_larii = J_lam(x0_ii, lar_ii)
        J_lambdasii = (J_laii, J_lrii, J_2laii, J_2lrii, J_larii)

        # used in gdHS in chain contribution
        x0i_matrix = np.array([x0_ii**0, x0_ii, x0_ii**2, x0_ii**3])

        # for gamma_c in chain contribution
        beps_ii = self.eps_ii*beta
        beps_ii2 = beps_ii**2

        # tetha = np.exp(beta * self.eps_ii) - 1.
        tetha = np.exp(beps_ii) - 1.

        temp_aux = [beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl,
                    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl,
                    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii,
                    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii,
                    J_lambdasii, x0i_matrix, beps_ii, beps_ii2]

        if self.asso_bool:
            T_ad = 1/(self.eps_ij*beta)
            Fklab = np.exp(self.epsAB_kl * beta) - 1
            temp_aux.append(T_ad)
            temp_aux.append(Fklab)
        return temp_aux

    def association_solver(self, x, rho, T, Xass0=None):
        """
        association_solver(x, rho, T, Xass0)

        Method that computes the fraction of non-bonded sites.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molar density [mol/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        Returns
        -------
        Xass: array_like
            fraction of non-bonded sites. (None if the mixture is non-associating)
        """
        if self.asso_bool:
            temp_aux = self.temperature_aux(T)
            rhom = rho * Na
            Xass = association_solver(self, x, rhom, temp_aux, Xass0)
        else:
            Xass = None
        return Xass

    def association_check(self, x, rho, T, Xass):
        """
        association_check(x, rho, T, Xass0)

        Method that checks if the association sites are consistent.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molar density [mol/m3]
        T: float
            absolute temperature [K]
        Xass: array
            Initial guess for the calculation of fraction of non-bonded sites
        Returns
        -------
        of: float
            objective function that should be zero if the association sites
            are consistent. (Zero if the mixture is non-associating)
        """
        if self.asso_bool:
            temp_aux = self.temperature_aux(T)
            rhom = rho * Na
            of = assocation_check(self, x, rhom, temp_aux, Xass)
        else:
            of = 0.

        return of

    def ares(self, x, rho, T, Xass0=None):
        """
        ares(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the mixture.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           residual dimentionless Helmholtz free energy [Adim]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = ares(self, x, rho, temp_aux, Xass0)
        return a

    def dares_drho(self, x, rho, T, Xass0=None):
        """
        dares_drho(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the mixture
        and its first density derivative.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           residual dimentionless Helmholtz free energy [Adim, m^3]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = dares_drho(self, x, rho, temp_aux, Xass0)
        return a

    def d2ares_drho(self, x, rho, T, Xass0=None):
        """
        d2ares_drho(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the mixture
        and its first and second density derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           residual dimentionless Helmholtz free energy [Adim, m^3, m^6]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = d2ares_drho(self, x, rho, temp_aux, Xass0)
        return a

    def dares_dx(self, x, rho, T, Xass0=None):
        """
        dares_dx(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the mixture
        and its composition derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           residual dimentionless Helmholtz free energy [Adim]
        ax: array_like
           composition derivatives of residual dimentionless Helmholtz
           free energy [Adim]
        """
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = dares_dx(self, x, rho, temp_aux, Xass0)
        return a, ax

    def dares_dxrho(self, x, rho, T, Xass0=None):
        """
        dares_dx(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the mixture
        and its density and composition derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           residual dimentionless Helmholtz free energy [Adim, m^3]
        ax: array_like
           composition derivatives of residual dimentionless Helmholtz
           free energy [Adim]
        """
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = dares_dx_drho(self, x, rho, temp_aux, Xass0)
        return a, ax

    def afcn_aux(self, x, rho, temp_aux, Xass0=None):
        """
        afcn_aux(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           total Helmholtz free energy [J/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = ares(self, x, rho, temp_aux, Xass0)
        a += aideal(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def dafcn_drho_aux(self, x, rho, temp_aux, Xass0=None):
        """
        dafcn_drho_aux(x, rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its first density derivative.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           toal Helmholtz free energy [J/mol, J m^3/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = dares_drho(self, x, rho, temp_aux, Xass0)
        a += daideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def d2afcn_drho_aux(self, x, rho, temp_aux, Xass0=None):
        """
        d2afcn_drho_aux(x, rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its first and second density derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           toal Helmholtz free energy [J/mol, J m^3/mol, J m^6/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = d2ares_drho(self, x, rho, temp_aux, Xass0)
        a += d2aideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def dafcn_dx_aux(self, x, rho, temp_aux, Xass0=None):
        """
        dafcn_dx_aux(x, rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its composition derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           total dimentionless Helmholtz free energy [J/mol]
        ax: array_like
           composition derivatives of total dimentionless Helmholtz
           free energy [J/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        ar, aresx, Xass = dares_dx(self, x, rho, temp_aux, Xass0)
        aideal, aidealx = daideal_dx(x, rho, beta)
        a = (ar + aideal)
        a *= (Na/beta)
        ax = (aresx + aidealx)
        ax *= (Na/beta)
        return a, ax, Xass

    def dafcn_dxrho_aux(self, x, rho, temp_aux, Xass0=None):
        """
        dafcn_dxrho_aux(x, rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its composition a density derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           total dimentionless Helmholtz free energy [J/mol, J m^3/mol]
        ax: array_like
           composition derivatives of total dimentionless Helmholtz
           free energy [J/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        ar, aresx, Xass = dares_dx_drho(self, x, rho, temp_aux, Xass0)
        aideal, aidealx = daideal_dx_drho(x, rho, beta)
        a = (ar + aideal)
        a *= (Na/beta)
        ax = (aresx + aidealx)
        ax *= (Na/beta)
        return a, ax, Xass

    def afcn(self, x, rho, T, Xass0=None):
        """
        afcn(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           total Helmholtz free energy [J/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.afcn_aux(x, rho, temp_aux, Xass0)
        return a

    def dafcn_drho(self, x, rho, T, Xass0=None):
        """
        dafcn_drho(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its first density derivative.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           toal Helmholtz free energy [J/mol, J m^3/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.dafcn_drho_aux(x, rho, temp_aux, Xass0)
        return a

    def d2afcn_drho(self, x, rho, T, Xass0=None):
        """
        d2afcn_drho(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its first and second density derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           toal Helmholtz free energy [J/mol, J m^3/mol, J m^6/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.d2afcn_drho_aux(x, rho, temp_aux, Xass0)
        return a

    def dafcn_dx(self, x, rho, T, Xass0=None):
        """
        dafcn_dx(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its composition derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           total dimentionless Helmholtz free energy [J/mol]
        ax: array_like
           composition derivatives of total dimentionless Helmholtz
           free energy [J/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = self.dafcn_dx_aux(x, rho, temp_aux, Xass0)
        return a, ax

    def dafcn_dxrho(self, x, rho, T, Xass0=None):
        """
        dafcn_dxrho(x, rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the mixture
        and its composition a density derivatives.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            molecular density [molecules/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array_like
           total dimentionless Helmholtz free energy [J/mol, J m^3/mol]
        ax: array_like
           composition derivatives of total dimentionless Helmholtz
           free energy [J/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = self.dafcn_dxrho_aux(x, rho, temp_aux, Xass0)
        return a, ax

    def density_aux(self, x, temp_aux, P, state, rho0=None, Xass0=None):
        """
        density_aux(x, temp_aux, P, state)
        Method that computes the density of the mixture at given composition,
        temperature, pressure and aggregation state.

        Parameters
        ----------
        x: array_like
            molar fraction array
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapor phase
        rho0 : float, optional
            initial guess to compute density root [mol/m^3]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        density: float
            density [mol/m^3]
        Xass : array
            computed fraction of nonbonded sites
        """
        if rho0 is None:
            rho, Xass = density_topliss(state, x, temp_aux, P, Xass0, self)
        else:
            rho, Xass = density_newton(rho0, x, temp_aux, P, Xass0, self)
        return rho, Xass

    def density(self, x, T, P, state, rho0=None, Xass0=None):
        """
        density(x, T, P, state)
        Method that computes the density of the mixture at given composition,
        temperature, pressure and aggregation state.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapor phase
        rho0 : float, optional
            initial guess to compute density root [mol/m^3]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        density: float
            density [mol/m^3]
        """
        temp_aux = self.temperature_aux(T)
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)
        return rho

    def pressure_aux(self, x, rho, temp_aux, Xass0=None):
        """
        pressure_aux(x, rho, temp_aux, Xass0)

        Method that computes the pressure at given composition,
        density [mol/m3] and temperature [K]

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            density [mol/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        P : float
            pressure [Pa]
        Xass : array
            computed fraction of nonbonded sites
        """
        rhomolecular = Na * rho
        da, Xass = self.dafcn_drho_aux(x, rhomolecular, temp_aux, Xass0)
        afcn, dafcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        return Psaft, Xass

    def dP_drho_aux(self, x, rho, temp_aux, Xass0=None):
        """
        dP_drho_aux(rho, temp_aux, Xass0)

        Method that computes the pressure and its density derivative at given
        composition, density [mol/m3] and temperature [K]

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            density [mol/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        P : float
            pressure [Pa]
        dP: float
            derivate of pressure respect density [Pa m^3 / mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        rhomolecular = Na * rho
        da, Xass = self.d2afcn_drho_aux(x, rhomolecular, temp_aux, Xass0)
        afcn, dafcn, d2afcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        dPsaft = 2 * rhomolecular * dafcn + rhomolecular**2 * d2afcn
        return Psaft, dPsaft, Xass

    def pressure(self, x, rho, T, Xass0=None):
        """
        pressure(x, rho, T, Xass0)

        Method that computes the pressure at given composition,
        density [mol/m3] and temperature [K]

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            density [mol/m3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        P : float
            pressure [Pa]
        """
        temp_aux = self.temperature_aux(T)
        Psaft, Xass = self.pressure_aux(x, rho, temp_aux, Xass0)
        return Psaft

    def dP_drho(self, x, rho, T, Xass0=None):
        """
        dP_drho(rho, T, Xass0)

        Method that computes the pressure and its density derivative at given
        composition, density [mol/m3] and temperature [K]

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho: float
            density [mol/m3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        P : float
            pressure [Pa]
        dP: float
            derivate of pressure respect density [Pa m^3 / mol]
        """
        temp_aux = self.temperature_aux(T)
        Psaft, dPsaft, Xass = self.dP_drho_aux(x, rho, temp_aux, Xass0)
        return Psaft, dPsaft

    def logfugmix_aux(self, x, temp_aux, P, state, v0=None, Xass0=None):
        """
        logfugmix_aux(x, temp_aux, P, state, v0, Xass0)

        Method that computes the fugacity coefficient of the mixture at given
        composition, temperature and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        P: float
            pressure [Pa]
        state: string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        lnphi: float
            fugacity coefficient of the mixture
        v: float
            computed volume of the phase [m^3/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        RT = Na/beta

        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)

        v = 1./rho
        rhomolecular = Na * rho
        ar, Xass = ares(self, x, rhomolecular, temp_aux, Xass)
        Z = P * v / RT
        lnphi = ar + (Z - 1.) - np.log(Z)
        return lnphi, v, Xass

    def logfugmix(self, x, T, P, state, v0=None, Xass0=None):
        """
        logfugmix(x, T, P, state, v0, Xass0)

        Method that computes the fugacity coefficient of the mixture at given
        composition, temperature and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T: float
            absolute temperature [K]
        P: float
            pressure [Pa]
        state: string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        lnphi: float
            fugacity coefficient of the mixture
        v: float
            computed volume of the phase [m^3/mol]
        """
        temp_aux = self.temperature_aux(T)
        lnphi, v, Xass = self.logfugmix_aux(x, temp_aux, P, state, v0, Xass0)
        return lnphi, v

    def logfugef_aux(self, x, temp_aux, P, state, v0=None, Xass0=None):
        """
        logfugef_aux(x, temp_aux, P, state, v0, Xass0)

        Method that computes the effective fugacity coefficient of the
        components in the mixture at given composition, temperature
        and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        P: float
            pressure [Pa]
        state: string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        lnphi: array_like
            effective fugacity coefficient of the components
        v: float
            computed volume of the phase [m^3/mol]
        Xass : array
            computed fraction of nonbonded sites
        """

        beta = temp_aux[0]
        RT = Na/beta

        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0

        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)
        v = 1./rho
        rhomolecular = Na * rho
        ar, daresx, Xass = dares_dx(self, x, rhomolecular, temp_aux, Xass)
        Z = P * v / RT
        mures = ar + (Z - 1.) + daresx - np.dot(x, daresx)
        lnphi = mures - np.log(Z)
        return lnphi, v, Xass

    def logfugef(self, x, T, P, state, v0=None, Xass0=None):
        """
        logfugef(x, T, P, state, v0, Xass0)

        Method that computes the effective fugacity coefficient of the
        components in the mixture at given composition, temperature
        and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T: float
            absolute temperature [K]
        P: float
            pressure [Pa]
        state: string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        lnphi: array_like
            effective fugacity coefficient of the components
        v: float
            computed volume of the phase [m^3/mol]
        """
        temp_aux = self.temperature_aux(T)
        lnphi, v, Xass = self.logfugef_aux(x, temp_aux, P, state, v0, Xass0)
        return lnphi, v

    def a0ad_aux(self, rhoi, temp_aux, Xass0=None):
        """
        a0ad_aux(rhoi, temp_aux, Xass0)

        Method that computes the Helmholtz density energy divided by RT at
        given density vector and temperature.

        Parameters
        ----------

        rhoi : array_like
            density vector [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a0ad: float
            Helmholtz density energy divided by RT [mol/m^3]
        Xass : array
            computed fraction of nonbonded sites
        """

        rho = np.sum(rhoi)
        x = rhoi/rho
        rhomolecular = Na * rho
        beta = temp_aux[0]

        a, Xass = ares(self, x, rhomolecular, temp_aux, Xass0)
        a += aideal(x, rhomolecular, beta)

        a0 = a*rho

        return a0, Xass

    def a0ad(self, rhoi, T, Xass0=None):
        """
        a0ad(rhoi, T, Xass0)

        Method that computes the Helmholtz density energy divided by RT at
        given density vector and temperature.

        Parameters
        ----------

        rhoi : array_like
            density vector [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a0ad: float
            Helmholtz density energy divided by RT [mol/m^3]
        """
        temp_aux = self.temperature_aux(T)
        a0, Xass = self.a0ad_aux(rhoi, temp_aux, Xass0)
        return a0

    def muad_aux(self, rhoi, temp_aux, Xass0=None):
        """
        muad_aux(rhoi, temp_aux, Xass0)

        Method that computes the dimentionless chemical potential at given
        density vector and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: array_like
            chemical potential [Adim]
        Xass : array
            computed fraction of nonbonded sites
        """
        rho = np.sum(rhoi)
        x = rhoi/rho
        rhom = Na * rho

        beta = temp_aux[0]
        ares, aresx, Xass = dares_dx_drho(self, x, rhom, temp_aux, Xass0)
        aideal, aidealx = daideal_dx_drho(x, rhom, beta)
        afcn, dafcn = (ares + aideal)
        ax = (aresx + aidealx)
        Z = dafcn * rhom
        mu = afcn + ax - np.dot(x, ax) + (Z)
        return mu, Xass

    def muad(self, rhoi, T, Xass0=None):
        """
        muad(rhoi, T, Xass0)

        Method that computes the dimentionless chemical potential at given
        density vector and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: array_like
            chemical potential [Adim]
        """
        temp_aux = self.temperature_aux(T)
        mu, Xass = self.muad_aux(rhoi, temp_aux, Xass0)
        return mu

    def dmuad_aux(self, rhoi, temp_aux, Xass0=None):
        """
        dmuad_aux(rhoi, temp_aux, Xass0)

        Method that computes the chemical potential and its numerical
        derivative at given density vector and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: array_like
            chemical potential [J/mol]
        dmuad: array_like
            derivavites of the chemical potential respect to rhoi [J m^3/mol^2]
        Xass : array
            computed fraction of nonbonded sites
        """
        nc = self.nc
        h = 1e-3
        diff = h * np.eye(nc)

        mu, Xass = self.muad_aux(rhoi, temp_aux, Xass0)

        arr = []
        for i in range(nc):
            muad1, _ = self.muad_aux(rhoi + diff[i], temp_aux, Xass)
            muad_1, _ = self.muad_aux(rhoi - diff[i], temp_aux, Xass)
            arr.append((muad1 - muad_1)/(2*h))

        dmu = np.column_stack(arr)
        return mu, dmu, Xass

    def dmuad(self, rhoi, T, Xass0=None):
        """
        dmuad(rhoi, T, Xass0)

        Method that computes the chemical potential and its numerical
        derivative at given density vector and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: array_like
            chemical potential [J/mol]
        dmuad: array_like
            derivavites of the chemical potential respect to rhoi [J m^3/mol^2]
        """
        temp_aux = self.temperature_aux(T)
        mu, dmu, Xass = self.dmuad_aux(rhoi, temp_aux, Xass0)
        return mu, dmu

    def dOm_aux(self, rhoi, temp_aux, mu, Psat, Xass0=None):
        """
        dOm_aux(rhoi, temp_aux, mu, Psat, Xass0)

        Method that computes the Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        mu : float
            adimentional chemical potential at equilibrium
        Psat : float
            equilibrium pressure divided by RT [Pa mol / J]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        dom: float
            Thermodynamic Grand potential [Pa mol / J]
        Xass : array
            computed fraction of nonbonded sites
        """
        a0ad, Xass = self.a0ad_aux(rhoi, temp_aux, Xass0)
        dom = a0ad - np.sum(np.nan_to_num(rhoi*mu)) + Psat
        return dom, Xass

    def dOm(self, rhoi, T, mu, Psat, Xass0=None):
        """
        dOm(rhoi, T, mu, Psat, Xass0)

        Method that computes the Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------
        rhoi : array_like
            density vector [mol/m^3]
        T : float
            absolute temperature [K]
        mu : float
            adimentional chemical potential at equilibrium
        Psat : float
            equilibrium pressure divided by RT [Pa mol / J]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        Out: float
            Thermodynamic Grand potential [Pa]
        """
        temp_aux = self.temperature_aux(T)
        dom, Xass = self.dOm_aux(rhoi, temp_aux, mu, Psat, Xass0)
        return dom

    def sgt_adim(self, T):
        '''
        sgt_adim(T)

        Method that evaluates adimentional factor for temperature, pressure,
        density, tension and distance for interfacial properties computations
        with SGT.

        Parameters
        ----------
        T : float
        absolute temperature [K]

        Returns
        -------
        Tfactor : float
            factor to obtain dimentionless temperature (K -> K)
        Pfactor : float
            factor to obtain dimentionless pressure (Pa -> Pa/RT)
        rofactor : float
            factor to obtain dimentionless density (mol/m3 -> mol/m3)
        tenfactor : float
            factor to obtain dimentionless surface tension (mN/m)
        zfactor : float
            factor to obtain dimentionless distance  (Amstrong -> m)
        '''
        beta = 1 / (kb*T)
        RT = (Na/beta)
        cii0 = np.polyval(self.cii[0], T)  # computing first component cii

        Tfactor = 1.
        Pfactor = 1. / RT
        rofactor = 1.
        tenfactor = np.sqrt(cii0*RT) * 1000  # To give tension in mN/m
        zfactor = 10**-10 * np.sqrt(RT / cii0)

        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def beta_sgt(self, beta0, beta1=None, beta2=None, beta3=None):
        r"""
        beta_sgt(beta)

        Method that adds beta correction for cross influence parameters used
        in SGT. The beta correction is computed as follows:

        .. math::
            \beta_{ij} =  \beta_{ij,0} + \beta_{ij,1} \cdot T +  \beta_{ij,2} \cdot T^2 + \frac{\beta_{ij,3}}{T}

        Parameters
        ----------
        beta0 : array_like
            beta0 matrix (Symmetric, Diagonal==0, shape=(nc, nc)) [Adim]
        beta1 : array_like, optional
            beta1 matrix (Symmetric, Diagonal==0, shape=(nc, nc)) [1/K].
            If None, then a zero matrix is assumed.
        beta2 : array_like, optional
            beta2 matrix (Symmetric, Diagonal==0, shape=(nc, nc)) [1/K^2].
            If None, then a zero matrix is assumed.
        beta3 : array_like, optional
            beta3 matrix (Symmetric, Diagonal==0, shape=(nc, nc)) [K]
            If None, then a zero matrix is assumed.

        """
        nc = self.nc

        Beta0 = np.asarray(beta0)
        shape = Beta0.shape
        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(Beta0, Beta0.T)
        diagZero = np.all(np.diagonal(Beta0) == 0.)
        if isSquare and isSymmetric and diagZero:
            self.beta0 = Beta0
        else:
            raise Exception('beta0 matrix is not square, symmetric or diagonal==0')

        if beta1 is None:
            Beta1 = np.zeros([nc, nc])
            self.beta1 = Beta1
        else:
            Beta1 = np.asarray(beta1)
            shape = Beta1.shape
            isSquare = shape == (nc, nc)
            isSymmetric = np.allclose(Beta1, Beta1.T)
            diagZero = np.all(np.diagonal(Beta1) == 0.)
            if isSquare and isSymmetric and diagZero:
                self.beta1 = Beta1
            else:
                raise Exception('beta1 matrix is not square, symmetric or diagonal==0')
        if beta2 is None:
            Beta2 = np.zeros([nc, nc])
            self.beta2 = Beta2
        else:
            Beta2 = np.asarray(beta2)
            shape = Beta2.shape
            isSquare = shape == (nc, nc)
            isSymmetric = np.allclose(Beta2, Beta2.T)
            diagZero = np.all(np.diagonal(Beta2) == 0.)
            if isSquare and isSymmetric and diagZero:
                self.beta2 = Beta2
            else:
                raise Exception('beta2 matrix is not square, symmetric or diagonal==0')

        if beta3 is None:
            Beta3 = np.zeros([nc, nc])
            self.beta3 = Beta3
        else:
            Beta3 = np.asarray(beta3)
            shape = Beta3.shape
            isSquare = shape == (nc, nc)
            isSymmetric = np.allclose(Beta3, Beta3.T)
            diagZero = np.all(np.diagonal(Beta3) == 0.)
            if isSquare and isSymmetric and diagZero:
                self.beta3 = Beta3
            else:
                raise Exception('beta3 matrix is not square, symmetric or diagonal==0')

    def set_betaijsgt(self, i, j, beta0, beta1=0., beta2=0., beta3=0.):
        r"""
        set_betaijsgt(i,j, beta0, beta1, beta2, beta3)

        Method that set betaij correction cross influence parameter between
        component i and j.
        The beta correction is computed as follows:

        .. math::
            \beta_{ij} =  \beta_{ij,0} + \beta_{ij,1} \cdot T +  \beta_{ij,2} \cdot T^2 + \frac{\beta_{ij,3}}{T}

        Parameters
        ----------
        i : int
            index of component i.
        j : int
            index of component j.
        beta0 : float
            beta0 value between component i and j [Adim]
        beta1 : float, optional
            beta1 value between component i and j [1/K]. Default to zero.
        beta2 : float, optional
            beta2 value between component i and j [1/K^2]. Default to zero.
        beta3 : float, optional
            beta3 value between component i and j [K]. Default to zero.

        """
        typei = type(i) == int
        typej = type(j) == int

        nc = self.nc
        nc_i = 0 <= i <= (nc - 1)
        nc_j = 0 <= j <= (nc - 1)

        i_j = i != j

        if (not nc_i) or (not nc_j):
            raise Exception('Index i or j bigger than (nc-1)')
        if not i_j:
            raise Exception('Cannot set betaij for i=j')

        if typei and typej and nc_i and nc_j and i_j:
            self.beta0[i, j] = beta0
            self.beta0[j, i] = beta0

            self.beta1[i, j] = beta1
            self.beta1[j, i] = beta1

            self.beta2[i, j] = beta2
            self.beta2[j, i] = beta2

            self.beta3[i, j] = beta3
            self.beta3[j, i] = beta3

    def ci(self, T):
        """
        Method that computes the matrix of cij interaction parameter for SGT at
        given temperature.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        ci : array_like
            influence parameter matrix at given temperature [J m^5 / mol^2]

        """
        n = self.nc
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i], T)
        cij = np.sqrt(np.outer(ci, ci))

        beta = self.beta0 + self.beta1*T + self.beta2*T**2 + self.beta3/T

        cij *= (1 - beta)
        return cij

    def EntropyR(self, x, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        EntropyR(x, T, P, state, v0, Xass0, T_step)

        Method that computes the residual entropy of the mixture at given
        temperature and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Sr : float
            residual entropy [J/mol K]

        """

        temp_aux = self.temperature_aux(T)
        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0

        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)

        v = 1/rho
        rhomolecular = Na * rho
        a, Xass = ares(self, x, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta
        Z = P * v / RT

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, x, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, x, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, x, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, x, rhomolecular, temp_aux_2, Xass)

        F = a
        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Sr_TPN = Sr_TVN + np.log(Z)  # residual entropy (TPN) divided by R
        Sr_TPN *= R
        return Sr_TPN

    def EnthalpyR(self, x, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        EnthalpyR(x, T, P, state, v0, Xass0, T_step)

        Method that computes the residual enthalpy of the mixture at given
        temperature and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Hr : float
            residual enthalpy [J/mol]

        """
        temp_aux = self.temperature_aux(T)
        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)

        v = 1/rho
        rhomolecular = Na * rho
        a, Xass = ares(self, x, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta
        Z = P * v / RT

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, x, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, x, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, x, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, x, rhomolecular, temp_aux_2, Xass)

        F = a
        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Hr_TPN = F + Sr_TVN + Z - 1.  # residual entalphy divided by RT
        Hr_TPN *= RT
        return Hr_TPN

    def CvR(self, x, rho, T, Xass0=None, T_step=0.1):
        """
        CvR(x, rho, T, Xass0, T_step)

        Method that computes the residual isochoric heat capacity of the
        mixture at given density and temperature.

        Parameters
        ----------
        x: array_like
            molar fraction array
        rho : float
            density [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        T_step: float, optional
            Step to compute temperature numerical derivates of Helmholtz
            free energy

        Returns
        -------
        Cv: float
            isochoric heat capacity [J/mol K]
        """
        temp_aux = self.temperature_aux(T)

        rhomolecular = Na * rho

        a, Xass = ares(self, x, rhomolecular, temp_aux, Xass0)

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, x, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, x, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, x, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, x, rhomolecular, temp_aux_2, Xass)

        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        d2FdT = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12)/h**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= R
        return Cvr_TVN

    def CpR(self, x, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        Cpr(T, P, state, v0, Xass0, T_step)

        Method that computes the residual heat capacity of the mixture at given
        temperature and pressure.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Cp: float
            residual heat capacity [J/mol K]
        """

        temp_aux = self.temperature_aux(T)
        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)

        rhomolecular = Na * rho

        d2a, Xass = d2ares_drho(self, x, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = dares_drho(self, x, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = dares_drho(self, x, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = dares_drho(self, x, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = dares_drho(self, x, rhomolecular, temp_aux_2, Xass)

        a = d2a[:2]
        da_drho = a[1] * Na
        d2a_drho = d2a[2] * Na**2

        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        dFdT[1] *= Na

        d2FdT = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12) / h**2
        d2FdT[1] *= Na

        dP_dT = RT*(rho**2 * dFdT[1]) + P/T

        dP_drho = 2*rho*da_drho + 2.
        dP_drho += rho**2 * d2a_drho - 1.
        dP_drho *= RT

        dP_dV = -rho**2 * dP_drho
        # residual isochoric heat capacity
        Cvr_TVN = R * (-T**2*d2FdT[0] - 2*T*dFdT[0])
        # residual heat capacity
        Cpr = Cvr_TVN - R - T*dP_dT**2/dP_dV
        return Cpr

    def speed_sound(self, x, T, P, state, v0=None, Xass0=None, T_step=0.1,
                    CvId=3*R/2, CpId=5*R/2):
        """
        speed_sound(x, T, P, state, v0, Xass0, T_step)

        Method that computes the speed of sound of the mixture at given
        temperature and pressure.

        This calculation requires that the molar weight of the fluids has been
        set in the component function.

        By default the ideal gas Cv and Cp are set to 3R/2 and 5R/2, the user
        can supply better values if available.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy
        CvId: float, optional
            Ideal gas isochoric heat capacity, set to 3R/2 by default [J/mol K]
        CpId: float, optional
            Ideal gas heat capacity, set to 3R/2 by default [J/mol K]

        Returns
        -------
        w: float
            speed of sound [m/s]
        """

        temp_aux = self.temperature_aux(T)
        if v0 is None:
            rho0 = None
        else:
            rho0 = 1./v0
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)

        rhomolecular = Na * rho

        d2a, Xass = d2ares_drho(self, x, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = dares_drho(self, x, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = dares_drho(self, x, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = dares_drho(self, x, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = dares_drho(self, x, rhomolecular, temp_aux_2, Xass)

        a = d2a[:2]
        da_drho = a[1] * Na
        d2a_drho = d2a[2] * Na**2

        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        dFdT[1] *= Na

        d2FdT = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12) / h**2
        d2FdT[1] *= Na

        dP_dT = RT*(rho**2 * dFdT[1]) + P/T

        dP_drho = 2*rho*da_drho + 2.
        dP_drho += rho**2 * d2a_drho - 1.
        dP_drho *= RT

        dP_dV = -rho**2 * dP_drho
        # residual isochoric heat capacity
        Cvr_TVN = R * (-T**2*d2FdT[0] - 2*T*dFdT[0])
        # residual heat capacity
        Cpr = Cvr_TVN - R - T*dP_dT**2/dP_dV

        # speed of sound calculation
        Cp = CpId + Cpr
        Cv = CvId + Cvr_TVN

        betas = -rho * (Cv/Cp) / dP_dV

        Mwx = np.dot(x, self.Mw)
        w2 = 1000./(rho * betas * Mwx)
        w = np.sqrt(w2)

        return w

    def get_lnphi_pure(self, T, P, state):
        """
        get_lnphi_pure(T, P, state)

        Method that computes the logarithm of the pure component's fugacity
        coefficient at given state, temperature T and pressure P.

        Parameters
        ----------
        T: float
            absolute temperature [K]
        P: float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapor phase

        Returns
        -------
        lnphi_pure: float
            logarithm of pure component's fugacity coefficient
        """

        lnphi_pure = np.zeros(self.nc)
        for i, pure_eos in enumerate(self.pure_eos):
            lnphi_pure[i], _ = pure_eos.logfug(T, P, state)
        return lnphi_pure
    
    def get_lngamma(self, x, T, P, v0=None, Xass0=None, lnphi_pure=None):
        """
        get_lngamma(x, T, P, v0, Xass0)

        Method that computes the activity coefficient of the mixture at given
        composition x, temperature T and pressure P.

        Parameters
        ----------
        x: array_like
            molar fraction array
        T: float
            absolute temperature [K]
        P: float
            pressure [Pa]
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        lnphi_pure: array, optional
            logarithm of the pure components's fugacity coefficient.
            Computed if not provided.

        Returns
        -------
        lngamma: float
            logarithm of activity coefficient model
        """


        if isinstance(x, (float, int)):
            x = np.array([x, 1.-x])
        elif not isinstance(x, np.ndarray):
            x = np.array(x)

        if self.nc > 2 and x.shape[0] < 2:
            raise ValueError('Please supply the whole molfrac vector for non-binary mixtures')

        lnphi_mix, _ = self.logfugef(x, T, P, 'L', v0=v0, Xass0=Xass0)

        if lnphi_pure is None:
            lnphi_pure = np.zeros_like(x)
            for i, pure_eos in enumerate(self.pure_eos):
                lnphi_pure[i], _ = pure_eos.logfug(T, P, 'L')

        lngamma = lnphi_mix - lnphi_pure
        return lngamma
