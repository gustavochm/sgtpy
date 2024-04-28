from __future__ import division, print_function, absolute_import
import numpy as np
from ..constants import kb, Na
from ..math import gauss
from .a1sB_monomer import x0lambda_evalm, x0lambda_evalc
from .monomer_aux import I_lam, J_lam
from .ares import ares, dares_drho, d2ares_drho
from .ideal import aideal, daideal_drho, d2aideal_drho
from .density_solver import density_topliss, density_newton
from .psat_saft import psat
from .tsat_saft import tsat
from .critical_pure import get_critical
from .association_aux import association_solver, association_check

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


class saftgammamie_pure():
    '''
    Pure component SAFT-Gamma-Mie EoS Object

    This object have implemeted methods for phase equilibrium
    as for interfacial properties calculations.

    Parameters
    ----------
    pure : object
        pure component created with component class
    compute_critical: bool
        If True the critical point of the fluid will attempt to be computed
        (it might fail for some fluids).

    Attributes
    ----------
    vki: array, number of groups i
    subgroups: array, groups present in the fluid
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

    xs_k: array, fraction of group k
    xs_m: float, equivalent chain lenght

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


    cii : influence factor for SGT [J m^5 / mol^2]

    Methods
    -------
    diameter : computes the diameter at given temperature
    temperature_aux : computes temperature dependent parameters of the fluid
    density : computes the density of the fluid
    psat : computes saturation pressure
    tsat : computes saturation temperature
    get_critical : attemps to compute the critical point of the fluid

    ares: computes the residual energy of the fluid
    dares_drho : computes residual Helmholtz energy and its density derivative
    d2ares_drho : computes residual Helmholtz energy and it density derivatives
    afcn: computes total Helmholtz energy
    dafcn_drho : computes total Helmholtz energy and its density derivative
    d2afcn_drho : computes total Helmholtz energy and it density derivatives
    pressure : computes the pressure
    dP_drho : computes pressure and its density derivative
    logfug : computes the fugacity coefficient
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential
    dOm : computes adimentional Thermodynamic Grand Potential
    ci :  computes influence parameters matrix for SGT
    sgt_adim : computes adimentional factors for SGT

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
    logfug_aux : computes logfug
    a0ad_aux : compute a0ad
    muad_aux : computes muad
    dOm_aux : computes dOm
    '''
    def __init__(self, pure, compute_critical=True):

        self.mw_kk = pure.mw_kk
        self.Mw = pure.Mw

        self.vki = pure.vki
        self.subgroups = pure.subgroups
        self.ngroups = pure.ngroups
        self.groups_index = pure.groups_index
        self.groups_indexes = pure.groups_indexes

        self.vk = pure.vk
        self.Sk = pure.Sk

        self.lr_kl = pure.lr_kl
        self.la_kl = pure.la_kl
        self.lar_kl = self.lr_kl + self.la_kl
        self.eps_kl = pure.eps_kl * kb
        self.sigma_kl = pure.sigma_kl * 1e-10
        self.sigma_kl3 = self.sigma_kl**3

        self.lr_kk = pure.lr_kk
        self.la_kk = pure.la_kk
        self.lar_kk = self.lr_kk + self.la_kk
        self.eps_kk = pure.eps_kk * kb
        self.sigma_kk = pure.sigma_kk * 1e-10
        self.sigma_kk3 = self.sigma_kk**3

        lr_kl, la_kl = self.lr_kl, self.la_kl
        lar_kl = lr_kl + la_kl
        dif_ckl = lr_kl - la_kl
        self.Ckl = lr_kl/dif_ckl*(lr_kl/la_kl)**(la_kl/dif_ckl)
        self.alphakl = self.Ckl * (1./(la_kl - 3.) - 1./(lr_kl - 3.))

        self.diag_index = np.diag_indices(self.ngroups)

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

        Sk = self.Sk
        vki = self.vki
        vk = self.vk

        self.xs_ki = Sk*vki*vk
        self.xs_m = np.sum(self.xs_ki)
        self.xs_k = self.xs_ki / self.xs_m

        # Chain needed parameters
        zs_ki = Sk*vki*vk
        zs_m = np.sum(zs_ki)
        zs_k = zs_ki / zs_m

        self.zs_k = zs_k

        self.eps_ii = np.matmul(np.matmul(zs_k, self.eps_kl), zs_k)
        self.la_ii = np.matmul(np.matmul(zs_k, la_kl), zs_k)
        self.lr_ii = np.matmul(np.matmul(zs_k, lr_kl), zs_k)
        self.lar_ii = self.la_ii + self.lr_ii
        self.sigma_ii3 = np.matmul(np.matmul(zs_k, self.sigma_kl3), zs_k)
        self.sigma_ii = np.cbrt(self.sigma_ii3)

        lr_ii, la_ii, lar_ii = self.lr_ii, self.la_ii, self.lar_ii
        dif_cii = lr_ii - la_ii
        self.Cii = lr_ii/dif_cii*(lr_ii/la_ii)**(la_ii/dif_cii)
        self.alphaii = self.Cii * (1./(la_ii - 3.) - 1./(lr_ii - 3.))
        self.Cii2 = self.Cii*2

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

        self.asso_bool = pure.asso_bool
        if self.asso_bool:
            self.S = pure.S
            self.sites_asso = pure.sites_asso
            self.kAB_kl = pure.kAB_kl * 1e-30
            self.epsAB_kl = pure.epsAB_kl * kb
            self.DIJ = pure.DIJ
            self.diagasso = pure.diagasso
            self.nsites = pure.nsites
            # self.molecule_id_index_asso = pure.molecule_id_index_asso
            self.group_asso_index = pure.group_asso_index
            # self.group_id_asso = pure.group_id_asso
            # self.n_sites_molecule = pure.n_sites_molecule

        # for SGT calculation
        self.cii = pure.cii

        # computing critical point
        self.critical = False
        if compute_critical:
            out = get_critical(self, None, None, method='hybr',
                               full_output=True)
            if out.success:
                self.critical = True
                self.Tc = out.Tc
                self.Pc = out.Pc
                self.rhoc = out.rhoc

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
        beta3 = beta2*beta
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
        d_ii3 = np.matmul(np.matmul(self.zs_k, d_kl3), self.zs_k)
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
        # for a1 and a2 pertubation evaluation in chain contributon
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
            # T_ad = 1/(self.eps_ii*beta)
            T_ad = 1/beps_ii
            Fklab = np.exp(self.epsAB_kl * beta) - 1
            temp_aux.append(T_ad)
            temp_aux.append(Fklab)
        return temp_aux

    def association_solver(self, rho, T, Xass0=None):
        """
        association_solver(rho, T, Xass0)
        Method that computes the fraction of non-bonded sites.

        Parameters
        ----------
        rho: float
            molecular density [mol/m3]
        T: float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        Xass: array
            fraction of non-bonded sites. (None if the fluid is non-associating)
        """

        if self.asso_bool:
            temp_aux = self.temperature_aux(T)
            rhom = rho * Na
            Xass = association_solver(self, rhom, temp_aux, Xass0)
        else:
            Xass = None

        return Xass

    def association_check(self, rho, T, Xass):
        """
        association_check(rho, T, Xass)
        Method that checks the fraction of non-bonded sites.

        Parameters
        ----------
        rho: float
            molecular density [mol/m3]
        T: float
            absolute temperature [K]
        Xass: array
            fraction of non-bonded sites

        Returns
        -------
        fo: array
            objective function that should be zero if the association sites
            are consistent. (Zero if the fluid is non-associating)
        """

        if self.asso_bool:
            temp_aux = self.temperature_aux(T)
            rhom = rho * Na
            fo = association_check(self, rhom, temp_aux, Xass)
        else:
            fo = 0.

        return fo

    def ares(self, rho, T, Xass0=None):
        """
        ares(x, rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the fluid.

        Parameters
        ----------
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
        a, Xass = ares(self, rho, temp_aux, Xass0)
        return a, Xass

    def dares_drho(self, rho, T, Xass0=None):
        """
        dares_drho(rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the fluid
        and its first density derivative.

        Parameters
        ----------
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
        a, Xass = dares_drho(self, rho, temp_aux, Xass0)
        return a, Xass

    def d2ares_drho(self, rho, T, Xass0=None):
        """
        d2ares_drho(rho, T, Xass0)
        Method that computes the residual Helmholtz free energy of the fluid
        and its first and second density derivatives.

        Parameters
        ----------
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
        a, Xass = d2ares_drho(self, rho, temp_aux, Xass0)
        return a, Xass

    def afcn_aux(self, rho, temp_aux, Xass0=None):
        """
        afcn_aux(rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the fluid.

        Parameters
        ----------
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           Helmholtz free energy [J/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = ares(self, rho, temp_aux, Xass0)
        a += aideal(rho, beta)
        a *= (Na/beta)
        return a, Xass

    def dafcn_aux(self, rho, temp_aux, Xass0=None):
        """
        dafcn_aux(rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the fluid and
        its first density derivative.

        Parameters
        ----------
        rho: float
            density [mol/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array
           Helmholtz free energy and its derivative  [J/mol, J m^3/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = dares_drho(self, rho, temp_aux, Xass0)
        a += daideal_drho(rho, beta)
        a *= (Na/beta)
        return a, Xass

    def d2afcn_aux(self, rho, temp_aux, Xass0=None):
        """
        d2afcn_aux(rho, temp_aux, Xass0)
        Method that computes the total Helmholtz free energy of the fluid and
        its first ans second density derivative.

        Parameters
        ----------
        rho: float
            molecular density [molecules/m3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array
           Helmholtz free energy and its derivatives: a, da, d2a
           [J/mol, J m^3/mol^2,  J m^6/mol^3]
        Xass : array
            computed fraction of nonbonded sites
        """
        beta = temp_aux[0]
        a, Xass = d2ares_drho(self, rho, temp_aux, Xass0)
        a += d2aideal_drho(rho, beta)
        a *= (Na/beta)
        return a, Xass

    def afcn(self, rho, T, Xass0=None):
        """
        afcn(rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the fluid.

        Parameters
        ----------
        rho: float
            molecular density [molecules/m3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: float
           Helmholtz free energy [J/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.afcn_aux(rho, temp_aux, Xass0)
        return a

    def dafcn_drho(self, rho, T, Xass0=None):
        """
        dafcn_drho(rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the fluid and
        its first density derivative.

        Parameters
        ----------
        rho: float
            molecular density [molecules/m3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array
           Helmholtz free energy and its derivative  [J/mol, J m^3/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.dafcn_aux(rho, temp_aux, Xass0)
        return a

    def d2afcn_drho(self, rho, T, Xass0=None):
        """
        d2afcn_drho(rho, T, Xass0)
        Method that computes the total Helmholtz free energy of the fluid and
        its first ans second density derivative.

        Parameters
        ----------
        rho: float
            molecular density [molecules/m3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a: array
           Helmholtz free energy and its derivatives: a, da, d2a
           [J/mol, J m^3/mol,  J m^6/mol]
        """
        temp_aux = self.temperature_aux(T)
        a, Xass = self.d2afcn_aux(rho, temp_aux, Xass0)
        return a

    def pressure_aux(self, rho, temp_aux, Xass0=None):
        """
        pressure_aux(rho, temp_aux, Xass0)

        Method that computes the pressure at given density [mol/m3] and
        temperature [K]

        Parameters
        ----------
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
        da, Xass = self.dafcn_aux(rhomolecular, temp_aux, Xass0)
        afcn, dafcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        return Psaft, Xass

    def dP_drho_aux(self, rho, temp_aux, Xass0=None):
        """
        dP_drho_aux(rho, temp_aux, Xass0)

        Method that computes the pressure and its density derivative at given
        density [mol/m3] and temperature [K]

        Parameters
        ----------
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
        da, Xass = self.d2afcn_aux(rhomolecular, temp_aux, Xass0)
        afcn, dafcn, d2afcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        dPsaft = 2 * rhomolecular * dafcn + rhomolecular**2 * d2afcn
        return Psaft, dPsaft, Xass

    def pressure(self, rho, T, Xass0=None):
        """
        pressure(rho, T, Xass0)

        Method that computes the pressure at given density [mol/m3] and
        temperature [K]

        Parameters
        ----------
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
        Psaft, Xass = self.pressure_aux(rho, temp_aux, Xass0)
        return Psaft

    def dP_drho(self, rho, T, Xass0=None):
        """
        dP_drho(rho, T, Xass0)

        Method that computes the pressure and its density derivative at given
        density [mol/m3] and temperature [K]

        Parameters
        ----------
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
        Psaft, dPsaft, Xass = self.dP_drho_aux(rho, temp_aux, Xass0)

        return Psaft, dPsaft

    def density_aux(self, temp_aux, P, state, rho0=None, Xass0=None):
        """
        density_aux(T, temp_aux, state, rho0, Xass0)
        Method that computes the density of the fluid at T, P

        Parameters
        ----------
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
            rho, Xass = density_topliss(state, temp_aux, P, Xass0, self)
        else:
            rho, Xass = density_newton(rho0, temp_aux, P, Xass0, self)
        return rho, Xass

    def density(self, T, P, state, rho0=None, Xass0=None):
        """
        density(T, P, state)
        Method that computes the density of the fluid at T, P

        Parameters
        ----------

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
        rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)
        return rho

    def psat(self, T, P0=None, v0=[None, None], Xass0=[None, None],
             full_output=False):
        """
        psat(T, P0)

        Method that computes saturation pressure at fixed T

        Parameters
        ----------

        T : float
            absolute temperature [K]
        P0 : float, optional
            initial value to find saturation pressure [Pa]
        v0: list, optional
            initial guess for liquid and vapor phase, respectively [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        full_output: bool, optional
            whether to outputs or not all the calculation info.

        Returns
        -------
        psat : float
            saturation pressure [Pa]
        vl : float
            liquid saturation volume [m3/mol]
        vv : float
            vapor saturation volume [m3/mol]
        """
        out = psat(self, T, P0, v0, Xass0, full_output)
        return out

    def tsat(self, P,  T0=None, Tbounds=None, v0=[None, None],
             Xass0=[None, None], full_output=False):
        """
        tsat(P, Tbounds)

        Method that computes saturation temperature at given pressure.

        Parameters
        ----------

        P : float
            absolute pressure [Pa]
        T0 : float, optional
             Temperature to start iterations [K]
        Tbounds : tuple, optional
                (Tmin, Tmax) Temperature interval to start iterations [K]
        v0: list, optional
            initial guess for liquid and vapor phase, respectively [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites
        full_output: bool, optional
            whether to outputs or not all the calculation info.

        Returns
        -------
        tsat : float
            saturation temperature [K]
        vl : float
            liquid saturation volume [m^3/mol]
        vv : float
            vapor saturation volume [m^3/mol]
        """
        out = tsat(self, P, T0, Tbounds, v0, Xass0, full_output)
        return out

    def get_critical(self, Tc0=None, rhoc0=None, method='hybr',
                     full_output=False, overwrite=False):
        """
        get_critical(Tc0, rhoc0, method)

        Method that solves the critical coordinate of the fluid.
        This metho requires good initial guesses for the critical temperature
        and density to converge.

        Second derivative of pressure against volume is estimated numerically.

        Parameters
        ----------
        Tc0 : float
            initial guess for critical temperature [K]
        rhoc : float
            initial guess for critical density [mol/m^3]
        method : string, optional
            SciPy; root method to solve critical coordinate
        full_output: bool, optional
            whether to outputs or not all the calculation info
        overwrite: bool, optional
            wheter to overwrite already computed critical points

        Returns
        -------
        Tc: float
            Critical temperature [K]
        Pc: float
            Critical pressure [Pa]
        rhoc: float
            Critical density [mol/m3]
        """
        out = get_critical(self, Tc0, rhoc0, method, full_output)

        if overwrite:
            if full_output:
                if out.success:
                    self.critical = True
                    self.Tc = out.Tc
                    self.Pc = out.Pc
                    self.rhoc = out.rhoc
            else:
                Tc0 = out[0]
                rhoc0 = out[2]
                out2 = get_critical(self, Tc0, rhoc0, method, full_output=True)
                if out2.success:
                    self.critical = True
                    self.Tc = out2.Tc
                    self.Pc = out2.Pc
                    self.rhoc = out2.rhoc
        return out

    def logfug_aux(self, temp_aux, P, state, v0=None, Xass0=None):
        """
        logfug_aux(T, P, state, v0, Xass0)

        Method that computes the fugacity coefficient at given
        composition, temperature and pressure.

        Parameters
        ----------
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        P : float
            pressure [Pa]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [m^3/mol]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        logfug : float
            fugacity coefficient
        v : float
            computed volume of the phase [m^3/mol]
        Xass : array
            computed fraction of nonbonded sites
        """
        if v0 is None:
            rho, Xass = self.density_aux(temp_aux, P, state, None, Xass0)
        else:
            rho0 = 1./v0
            rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)
        v = 1./rho
        rhomolecular = Na * rho
        ar, Xass = ares(self, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta
        Z = P * v / RT
        lnphi = ar + (Z - 1.) - np.log(Z)
        return lnphi, v, Xass

    def logfug(self, T, P, state, v0=None, Xass0=None):
        """
        logfug(T, P, state, v0, Xass0)

        Method that computes the fugacity coefficient at given temperature
        and pressure.

        Parameters
        ----------
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

        Returns
        -------
        logfug: float
            fugacity coefficient
        v: float
            computed volume of the phase [m^3/mol]
        """
        temp_aux = self.temperature_aux(T)
        lnphi, v, Xass = self.logfug_aux(temp_aux, P, state, v0, Xass0)
        return lnphi, v

    def ci(self, T):
        '''
        ci(T)

        Method that evaluates the polynomial for the influence parameters used
        in the SGT theory for surface tension calculations.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        ci: float
            influence parameters [J m5 mol-2]
        '''

        return np.polyval(self.cii, T)

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

        cii = self.ci(T)  # computing temperature dependent cii

        Tfactor = 1.
        Pfactor = 1.
        rofactor = 1.
        tenfactor = np.sqrt(cii) * 1000  # To give tension in mN/m
        zfactor = 10**-10

        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def sgt_adim_fit(self, T):

        Tfactor = 1.
        Pfactor = 1.
        rofactor = 1.
        tenfactor = 1. * 1000  # To give tension in mN/m

        return Tfactor, Pfactor, rofactor, tenfactor

    def a0ad_aux(self, rho, temp_aux, Xass0=None):
        """
        a0ad_aux(ro, temp_aux, Xass0)

        Method that computes the adimenstional Helmholtz density energy at
        given density and temperature.

        Parameters
        ----------

        rho : float
            density [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a0ad: float
            Helmholtz density energy [J/m^3]
        Xass : array
            computed fraction of nonbonded sites
        """
        rhomolecular = rho * Na
        a0, Xass = self.afcn_aux(rhomolecular, temp_aux, Xass0)
        a0 *= rho

        return a0, Xass

    def a0ad(self, rho, T, Xass0=None):
        """
        a0ad(ro, T, Xass0)

        Method that computes the adimenstional Helmholtz density energy at
        given density and temperature.

        Parameters
        ----------

        rho : float
            density [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        a0ad: float
            Helmholtz density energy [J/m^3]
        """
        temp_aux = self.temperature_aux(T)
        a0, Xass = self.a0ad_aux(rho, temp_aux, Xass0)
        return a0

    def muad_aux(self, rho, temp_aux, Xass0=None):
        """
        muad_aux(rho, temp_aux, Xass0)

        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------
        rho : float
            density [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: float
            chemical potential [J/mol]
        Xass : array
            computed fraction of nonbonded sites
        """

        rhomolecular = rho * Na
        da, Xass = self.dafcn_aux(rhomolecular, temp_aux, Xass0)
        afcn, dafcn = da
        mu = afcn + rhomolecular * dafcn

        return mu, Xass

    def muad(self, rho, T, Xass0=None):
        """
        muad(rho, T, Xass0)

        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------
        rho : float
            density [mol/m^3]
        T : float
            absolute temperature [K]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        muad: float
            chemical potential [J/mol]
        """
        temp_aux = self.temperature_aux(T)
        mu, Xass = self.muad_aux(rho, temp_aux, Xass0)
        return mu

    def dOm_aux(self, rho, temp_aux, mu, Psat, Xass0=None):
        """
        dOm_aux(rho, temp_aux, mu, Psat, Xass0)

        Method that computes the adimenstional Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------
        rho : float
            density [mol/m^3]
        temp_aux : list
            temperature dependend parameters computed with temperature_aux(T)
        mu : float
            adimentional chemical potential at equilibrium
        Psat : float
            adimentional pressure [Pa]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        GPT: float
            Thermodynamic Grand potential [Pa]
        Xass : array
            computed fraction of nonbonded sites
        """
        a0, Xass = self.a0ad_aux(rho, temp_aux, Xass0)
        GPT = a0 - rho*mu + Psat

        return GPT, Xass

    def dOm(self, rho, T, mu, Psat, Xass0=None):
        """
        dOm(rho, T, mu, Psat, Xass0)

        Method that computes the adimenstional Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------
        rho : float
            density [mol/m^3]
        T : float
            absolute temperature [K]
        mu : float
            adimentional chemical potential at equilibrium
        Psat : float
            adimentional pressure [Pa]
        Xass0: array, optional
            Initial guess for the calculation of fraction of non-bonded sites

        Returns
        -------
        Out: float
            Thermodynamic Grand potential [Pa]
        """
        temp_aux = self.temperature_aux(T)
        GPT, Xass = self.dOm_aux(rho, temp_aux, mu, Psat, Xass0)
        return GPT

    def EntropyR(self, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        EntropyR(T, P, state, v0, Xass0, T_step)

        Method that computes the residual entropy at given temperature and
        pressure.

        Parameters
        ----------
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
            rho, Xass = self.density_aux(temp_aux, P, state, None, Xass0)
        else:
            rho0 = 1./v0
            rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)
        v = 1./rho
        rhomolecular = Na * rho

        a, Xass = ares(self, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta
        Z = P * v / RT

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, rhomolecular, temp_aux_2, Xass)

        F = a
        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Sr_TPN = Sr_TVN + np.log(Z)  # residual entropy (TPN) divided by R
        Sr_TPN *= R
        return Sr_TPN

    def EnthalpyR(self, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        EnthalpyR(T, P, state, v0, Xass0, T_step)

        Method that computes the residual enthalpy at given temperature and
        pressure.

        Parameters
        ----------
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
            rho, Xass = self.density_aux(temp_aux, P, state, None, Xass0)
        else:
            rho0 = 1./v0
            rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)
        v = 1./rho
        rhomolecular = Na * rho

        a, Xass = ares(self, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta
        Z = P * v / RT

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, rhomolecular, temp_aux_2, Xass)

        F = a
        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        Sr_TVN = -T*dFdT - F  # residual entropy divided by R
        Hr_TPN = F + Sr_TVN + Z - 1.  # residual entalphy divided by RT
        Hr_TPN *= RT
        return Hr_TPN

    def CvR(self, rho, T, Xass0=None, T_step=0.1):
        """
        CvR(rho, T, Xass0, T_step)

        Method that computes the residual isochoric heat capacity at given
        density and temperature.

        Parameters
        ----------
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

        a, Xass = ares(self, rhomolecular, temp_aux, Xass0)

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = ares(self, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = ares(self, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = ares(self, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = ares(self, rhomolecular, temp_aux_2, Xass)

        dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        d2FdT = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12)/h**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= R
        return Cvr_TVN

    def CpR(self, T, P, state, v0=None, Xass0=None, T_step=0.1):
        """
        Cpr(T, P, state, v0, Xass0, T_step)

        Method that computes the residual heat capacity at given temperature
        and pressure.

        Parameters
        ----------
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
            rho, Xass = self.density_aux(temp_aux, P, state, None, Xass0)
        else:
            rho0 = 1./v0
            rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)

        rhomolecular = Na * rho

        d2a, Xass = d2ares_drho(self, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = dares_drho(self, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = dares_drho(self, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = dares_drho(self, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = dares_drho(self, rhomolecular, temp_aux_2, Xass)

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

    def speed_sound(self, T, P, state, v0=None, Xass0=None, T_step=0.1,
                    CvId=3*R/2, CpId=5*R/2):
        """
        speed_sound(T, P, state, v0, Xass0, T_step, CvId, CpId)

        Method that computes the speed of sound at given temperature
        and pressure.

        This calculation requires that the molar weight of the fluid has been
        set in the component function.

        By default the ideal gas Cv and Cp are set to 3R/2 and 5R/2, the user
        can supply better values if available.

        Parameters
        ----------
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
            rho, Xass = self.density_aux(temp_aux, P, state, None, Xass0)
        else:
            rho0 = 1./v0
            rho, Xass = self.density_aux(temp_aux, P, state, rho0, Xass0)

        rhomolecular = Na * rho

        d2a, Xass = d2ares_drho(self, rhomolecular, temp_aux, Xass)
        beta = temp_aux[0]
        RT = Na/beta

        h = T_step
        temp_aux1 = self.temperature_aux(T+h)
        temp_aux2 = self.temperature_aux(T+2*h)
        temp_aux_1 = self.temperature_aux(T-h)
        temp_aux_2 = self.temperature_aux(T-2*h)

        a1, Xass1 = dares_drho(self, rhomolecular, temp_aux1, Xass)
        a2, Xass2 = dares_drho(self, rhomolecular, temp_aux2, Xass)
        a_1, Xass_1 = dares_drho(self, rhomolecular, temp_aux_1, Xass)
        a_2, Xass_2 = dares_drho(self, rhomolecular, temp_aux_2, Xass)

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

        w2 = 1000./(rho * betas * self.Mw)
        w = np.sqrt(w2)

        return w
