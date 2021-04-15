import numpy as np
from ..math import gauss

from .a1sB_monomer import x0lambda_eval
from .monomer_aux import J_lam, I_lam

from .ares import ares, dares_drho, d2ares_drho, dares_dx, dares_dxrho

from .ideal import aideal, daideal_drho, d2aideal_drho, daideal_dx
from .ideal import daideal_dxrho

from .association_aux import association_config
from .polarGV import aij, bij, cij

from .density_solver import density_topliss, density_newton
from ..constants import kb, Na


R = Na * kb


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


class saftvrmie_mix():
    '''
    SAFT-VR-Mie EoS for mixtures Object

    This object have implemeted methods for phase equilibrium
    as for interfacial properties calculations.

    Parameters
    ----------
    mixture : object
        mixture created with mixture class

    Attributes
    ----------
    nc: integrer
        number of component in the mixture
    ms: array_like
        number of chain segments
    sigma: array_like
        size parameter of Mie potential [m]
    eps: array_like
        well-depth of Mie potential [J]
    la: array_like
        attractive exponent for Mie potential
    lr: array_like
        repulsive exponent for Mie potential

    ring: geometric parameter for ring molecules
          (see Langmuir 2017, 33, 11518-11529, Table I.)

    sigmaij: array_like
        size parameters matrix for Mie potential [m]
    epsij: array_like
        well-depth energy matrix of Mie potential [J]
    laij: array_like
        attractive exponent matrix for Mie potential
    lrij: array_like
        repulsive exponent matrix for Mie potential
    alphaij: array_like
        matrix of alpha van der waals constant

    eABij: array_like
        association energy matrix [J]
    rcij: array_like
        association range matrix [m]
    rdij: array_like
        association site position matrix [m]
    sites: list
        triplet of number of association sites [B, P, N]

    mupol: array_like
        dipolar moment [Debye]
    npol: array_like
        number of dipolar sites

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

    def __init__(self, mixture):

        self.mixture = mixture
        self.Mw = np.asarray(mixture.Mw)

        # Pure component parameters
        self.lr = np.asarray(mixture.lr)
        self.la = np.asarray(mixture.la)
        self.lar = self.lr + self.la
        self.sigma = np.asarray(mixture.sigma)
        self.eps = np.asarray(mixture.eps)
        self.ms = np.asarray(mixture.ms)
        self.ring = np.asarray(mixture.ring)

        self.nc = mixture.nc

        sigma3 = self.sigma**3
        # Eq A45
        self.sigmaij = np.add.outer(self.sigma, self.sigma) / 2
        self.sigmaij3 = self.sigmaij**3
        # Eq A51

        if hasattr(mixture, 'KIJsaft'):
            kij = mixture.KIJsaft
        else:
            kij = np.zeros([mixture.nc, mixture.nc])
        self.epsij = np.multiply.outer(sigma3, sigma3)
        self.epsij *= np.multiply.outer(self.eps, self.eps)
        self.epsij **= 0.5
        self.epsij /= self.sigmaij3
        self.epsij *= (1-kij)

        # Eq A48
        self.lrij = np.sqrt(np.multiply.outer(self.lr-3., self.lr-3.)) + 3
        self.laij = np.sqrt(np.multiply.outer(self.la-3., self.la-3.)) + 3
        self.larij = self.lrij + self.laij

        # Eq A3
        dif_cij = self.lrij - self.laij
        self.Cij = self.lrij/dif_cij*(self.lrij/self.laij)**(self.laij/dif_cij)
        # Eq A24
        self.alphaij = self.Cij * (1./(self.laij - 3.) - 1./(self.lrij - 3.))
        self.Cij2 = self.Cij**2

        self.diag_index = np.diag_indices(self.nc)
        self.C = self.Cij[self.diag_index]
        self.C2 = self.Cij[self.diag_index]**2
        self.alpha = self.alphaij[self.diag_index]

        # For Second perturbation
        self.f1 = fi(self.alphaij, 1)
        self.f2 = fi(self.alphaij, 2)
        self.f3 = fi(self.alphaij, 3)
        self.f4 = fi(self.alphaij, 4)
        self.f5 = fi(self.alphaij, 5)
        self.f6 = fi(self.alphaij, 6)

        # Eq A18
        c_matrix = np.array([[0.81096, 1.7888, -37.578, 92.284],
                            [1.0205, -19.341, 151.26, -463.5],
                            [-1.9057, 22.845, -228.14, 973.92],
                            [1.0885, -6.1962, 106.98, -677.64]])

        lrij_power = np.power.outer(self.lrij, [0, -1, -2, -3])
        self.cctes_lrij = np.tensordot(lrij_power, c_matrix, axes=(-1, -1))

        laij_power = np.power.outer(self.laij, [0, -1, -2, -3])
        self.cctes_laij = np.tensordot(laij_power, c_matrix, axes=(-1, -1))

        lrij_2power = np.power.outer(2*self.lrij, [0, -1, -2, -3])
        self.cctes_2lrij = np.tensordot(lrij_2power, c_matrix, axes=(-1, -1))

        laij_2power = np.power.outer(2*self.laij, [0, -1, -2, -3])
        self.cctes_2laij = np.tensordot(laij_2power, c_matrix, axes=(-1, -1))

        larij_power = np.power.outer(self.larij, [0, -1, -2, -3])
        self.cctes_larij = np.tensordot(larij_power, c_matrix, axes=(-1, -1))

        # Monomer necessary term
        self.lambdasij = (self.laij, self.lrij, self.larij)
        self.cctesij = (self.cctes_laij, self.cctes_lrij,
                        self.cctes_2laij, self.cctes_2lrij, self.cctes_larij)

        # Chain neccesary term
        self.cctes_la = self.cctes_laij[self.diag_index]
        self.cctes_lr = self.cctes_lrij[self.diag_index]
        self.cctes_lar = self.cctes_larij[self.diag_index]
        self.cctes_2la = self.cctes_2laij[self.diag_index]
        self.cctes_2lr = self.cctes_2lrij[self.diag_index]

        self.lambdas = (self.la, self.lr, self.lar)
        self.cctes = (self.cctes_la, self.cctes_lr, self.cctes_2la,
                      self.cctes_2lr, self.cctes_lar)

        # For diameter calculation
        roots, weights = gauss(30)
        self.roots = roots
        self.weights = weights
        self.umie = U_mie(1./roots, self.C, self.eps, self.lr, self.la)

        self.dxhi00_drho = np.pi / 6

        # association config
        self.eABij = np.sqrt(np.multiply.outer(mixture.eAB, mixture.eAB))
        if hasattr(mixture, 'LIJsaft'):
            lij = mixture.LIJsaft
        else:
            lij = np.zeros([mixture.nc, mixture.nc])
        self.eABij *= (1.-lij)
        self.rcij = np.add.outer(mixture.rc, mixture.rc)/2
        self.rdij = np.add.outer(mixture.rd, mixture.rd)/2

        S, DIJ, compindex, indexabij, indexab, nsites, \
        dxjdx, diagasso = association_config(mixture.sitesmix, self)
        assoc_bool = nsites != 0
        self.assoc_bool = assoc_bool
        if assoc_bool:
            self.S = S
            self.DIJ = DIJ
            self.compindex = compindex
            self.indexab = indexab
            self.indexabij = indexabij
            self.nsites = nsites
            self.dxjdx = dxjdx
            self.diagasso = diagasso

        self.secondorder = False
        self.secondordersgt = True

        # Polar Contribution
        self.mupol = np.asarray(mixture.mupol)
        self.npol = np.asarray(mixture.npol)
        polar_bool = np.any(self.npol != 0)
        self.polar_bool = polar_bool
        if polar_bool:
            ms = self.ms
            self.mpol = ms * (ms < 2) + 2 * (ms >= 2)
            msij = np.sqrt(np.outer(ms, ms))
            msijk = np.cbrt(np.multiply.outer(ms, np.outer(ms, ms)))
            msij[msij > 2.] = 2.
            msijk[msijk > 2.] = 2.

            aux1 = np.array([np.ones_like(msij), (msij-1)/msij,
                            (msij-1)/msij * (msij-2)/msij])
            aux2 = np.array([np.ones_like(msijk), (msijk-1)/msijk,
                            (msijk-1)/msijk * (msijk-2)/msijk])
            self.anij = np.tensordot(aij, aux1, axes=(1))
            self.bnij = np.tensordot(bij, aux1, axes=(1))
            self.cnij = np.tensordot(cij, aux2, axes=(1))

            sigmaij = self.sigmaij
            mult = np.multiply.outer(sigmaij, sigmaij)
            listpolar = np.arange(self.nc)
            self.sigmaijk3 = mult[listpolar, :, listpolar] * sigmaij
            self.sigma3 = self.sigma**3
            # 1 D = 3.33564e-30 C * m
            # 1 C^2 = 9e9 N m^2
            cte = (3.33564e-30)**2 * (9e9)
            self.mupolad2 = self.mupol**2*cte/(self.ms*self.eps*self.sigma3)

        self.cii = mixture.cii
        self.cij = np.sqrt(np.outer(self.cii, self.cii))
        self.beta = np.zeros([self.nc, self.nc])

    def cii_correlation(self, overwrite=False):
        """
        cii_corelation()

        Method that computes the influence parameter of coarsed-grained
        molecules

        AIChE Journal, 62, 5, 1781-1794 (2016)
        Eq. (23)

        Parameters
        ----------

        overwrite : bool
            If true it will overwrite the actual influence parameter.

        Returns
        -------
        cii : array_like
            correlated influence parameter [J m^5 / mol^2]
        """
        cii = self.ms * (0.12008072630855947 + 2.2197907527439655 * self.alpha)
        cii *= np.sqrt(Na**2 * self.eps * self.sigma**5)
        cii **= 2
        if overwrite:
            self.cii = cii
            self.cij = np.sqrt(np.outer(self.cii, self.cii))
        return cii

    def diameter(self, beta):
        """
        diameter(beta)

        Method that computes the diameter of the fluids at given
        beta = 1 / kb T

        Journal of Chemical Physics, 139(15), 1–37 (2013)
        Eq. (7)

        Parameters
        ----------

        beta : float
            Boltzmann's factor: beta = 1 / kb T [1/J]

        Returns
        -------
        d : array_like
            computed diameters [m]
        """
        # umie = U_mie(1/roots, c, eps, lambda_r, lambda_a)
        integrer = np.exp(-beta * self.umie)
        d = self.sigma * (1. - np.matmul(self.weights, integrer))
        return d

    def temperature_aux(self, T):
        """
        temperature_aux(T)

        Method that computes temperature dependent parameters.
        It returns the following list:

        temp_aux = [beta, beta2, beta3, dii, dij, x0, x0i, di03, dij3,
                    I_lambdasij, J_lambdasij, a1vdw_cteij, a1vdwij,
                    beps, beps2, a1vdw_cte, x0i_matrix, tetha,
                    x0_a1, x0_a2, x0_g1, x0_g2, x0_a1ii, x0_a2ii,
                    Fab, aux_dii, aux_dii2, Kab, epsa, epsija]

        Journal of Chemical Physics, 139(15), 1–37 (2013)

        beta: Boltzmann's factor [1/J]
        beta2: beta**2  [1/J^2]
        beta3: beta**3  [1/J^3]
        dii: computed diameters [m] (Eq A9)
        dij: diameters matrix [m] (Eq A46)
        x0: sigmaij/dij [Adim] (Below Eq. A11)
        x0i: sigma/dii [Adim]
        di03: dii^3 [m^3]
        dij3: dij^3 [m^3]
        I_lambdasij: (Eq A14)
        J_lambdasij: (Eq A15)
        a1vdw_cteij: -12 * epsij * dij3
        a1vdwij: tuple using a1vdw_cteij with different lamda_ij exponents
        beps: beta*eps [Adim]
        beps2: betps**2 [Adim]
        a1vdw_cte: -12 * eps * dii3
        x0i_matrix: np.array([x0i**0, x0i**1, x0i**2, x0i**3]) used in (Eq A29)
        tetha: exp(beta*eps)-1 [Adim] (Below Eq. A37)
        x0_a1, x0_a2, x0_g1, x0_g2: used to compute a1ij, a2ij
        x0_a1ii, x0_a2ii: diagonal of x0_a1, x0_a2
        Fab: association strength [Adim] (Eq. A41)
        aux_dii, aux_dii2: auxiliar variables used in association contribution
        Kab: association volume computed with  Eq. A43
        epsa: eps / kb / T [Adim]
        epsija: epsij / kb / T [Adim]

        Parameters
        ----------

        T : float
            Absolute temperature [K]

        Returns
        -------
        temp_aux : list
             list of computed parameters
        """
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        beta2 = beta**2
        beta3 = beta**3

        dii = self.diameter(beta)
        # Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        # Defining xhi0 without x dependence
        di03 = np.power.outer(dii, np.arange(4))
        dij3 = dij**3

        # used in a1, a2, g1 and g2
        out = x0lambda_eval(x0, self.la, self.lr, self.lar, self.laij,
                            self.lrij, self.larij, diag_index)
        x0_a1, x0_a2, x0_g1, x0_g2 = out
        x0_a1ii = x0_a1[:, diag_index[0], diag_index[1]]
        x0_a2ii = x0_a2[:, diag_index[0], diag_index[1]]

        # I and J used B term
        I_la = I_lam(x0, self.laij)
        I_lr = I_lam(x0, self.lrij)
        I_2la = I_lam(x0, 2*self.laij)
        I_2lr = I_lam(x0, 2*self.lrij)
        I_lar = I_lam(x0, self.larij)
        I_lambdasij = (I_la, I_lr, I_2la, I_2lr, I_lar)

        J_la = J_lam(x0, self.laij)
        J_lr = J_lam(x0, self.lrij)
        J_2la = J_lam(x0, 2*self.laij)
        J_2lr = J_lam(x0, 2*self.lrij)
        J_lar = J_lam(x0, self.larij)
        J_lambdasij = (J_la, J_lr, J_2la, J_2lr, J_lar)

        # Monomer necessary terms
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij,
                   a1vdw_larij)

        # Chain necessary terms
        a1vdw_cte = a1vdw_cteij[diag_index]

        x0i_matrix = np.array([x0i**0, x0i, x0i**2, x0i**3])

        beps = beta * self.eps
        beps2 = beps**2

        # tetha = np.exp(beta * self.eps) - 1.
        tetha = np.exp(beps) - 1.
        # For associating mixtures
        Fab = np.exp(beta * self.eABij) - 1.
        aux_dii = np.multiply.outer(dii, dii)/np.add.outer(dii, dii)
        aux_dii2 = aux_dii**2
        dij2 = dij**2

        rcij = self.rcij
        rcij2 = self.rcij**2
        rcij3 = rcij2*rcij
        rdij = self.rdij
        rdij2 = self.rdij**2
        rdij3 = rdij2*rdij

        Kab = np.log((rcij + 2*rdij)/dij)
        Kab *= 6*rcij3 + 18 * rcij2*rdij - 24 * rdij3
        aux1 = (rcij + 2 * rdij - dij)
        aux2 = (22*rdij2 - 5*rcij*rdij - 7*rdij*dij - 8*rcij2+rcij*dij+dij2)
        Kab += aux1 * aux2
        Kab /= (72*rdij2 * self.sigmaij3)
        Kab *= 4 * np.pi * dij2

        # For Polar mixtures
        epsa = beps
        epsija = self.epsij * beta

        temp_aux = [beta, beta2, beta3, dii, dij, x0, x0i, di03, dij3,
                    I_lambdasij, J_lambdasij, a1vdw_cteij, a1vdwij,
                    beps, beps2, a1vdw_cte, x0i_matrix, tetha,
                    x0_a1, x0_a2, x0_g1, x0_g2, x0_a1ii, x0_a2ii,
                    Fab, aux_dii, aux_dii2, Kab, epsa, epsija]
        return temp_aux

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
        a, ax, Xass = dares_dxrho(self, x, rho, temp_aux, Xass0)
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
        ar, aresx, Xass = dares_dxrho(self, x, rho, temp_aux, Xass0)
        aideal, aidealx = daideal_dxrho(x, rho, beta)
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
        ares, aresx, Xass = dares_dxrho(self, x, rhom, temp_aux, Xass0)
        aideal, aidealx = daideal_dxrho(x, rhom, beta)
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

        Tfactor = 1.
        Pfactor = 1. / RT
        rofactor = 1.
        tenfactor = np.sqrt(self.cii[0]*RT) * 1000  # To give tension in mN/m
        zfactor = 10**-10 * np.sqrt(RT / self.cii[0])
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def beta_sgt(self, beta):
        r"""
        beta_sgt(beta)

        Method that adds beta correction for cross influence parameters used
        in SGT.

        """
        nc = self.nc
        Beta = np.asarray(beta)
        shape = Beta.shape
        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(Beta, Beta.T)
        if isSquare and isSymmetric:
            self.beta = Beta
        else:
            raise Exception('beta matrix is not square or symmetric')

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
        return self.cij * (1 - self.beta)

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
