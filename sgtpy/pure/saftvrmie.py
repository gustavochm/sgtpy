import numpy as np
from ..math import gauss
from .ideal import aideal, daideal_drho, d2aideal_drho
from .association_aux import association_config
from .polarGV import aij, bij, cij
from .ares import ares, dares_drho, d2ares_drho
from .density_solver import density_topliss, density_newton
from .psat_saft import psat

from ..constants import kb, Na


R = Na * kb


def U_mie(r, c, eps, lambda_r, lambda_a):
    u = c * eps * (r**lambda_r - r**lambda_a)
    return u


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


# Equation 20
def fi(alpha, i):
    phi = phi16[i-1]
    num = np.dot(phi[nfi_num], np.power(alpha, nfi_num))
    den = 1 + np.dot(phi[nfi_den], np.power(alpha, nfi_den - 3))
    return num/den


class saftvrmie_pure():
    '''
    Pure component SAFT-VR-Mie EoS Object

    This object have implemeted methods for phase equilibrium
    as for interfacial properties calculations.

    Parameters
    ----------
    pure : object
        pure component created with component class

    Attributes
    ----------
    ms: number of chain segments
    sigma: size parameter of Mie potential [m]
    eps: well-depth of Mie potential [J]
    lambda_a: attractive exponent for Mie potential
    lambda_r: repulsive exponent for Mie potential

    ring: geometric parameter for ring molecules
          (see Langmuir 2017, 33, 11518-11529, Table I.)

    eABij: association energy [J]
    rcij: association range [m]
    rdij: association site position [m]
    sites: triplet of number of association sites [B, P, N]

    mupol: dipolar moment [Debye]
    npol: number of dipolar sites

    cii : influence factor for SGT [J m^5 / mol^2]

    Methods
    -------
    cii_correlation : correlates the influence parameter of the fluid.
    d : computes the diameter at given temperature.
    temperature_aux : computes temperature depedent parameters of the fluid.
    density : computes the density of the fluid.
    psat : computes saturation pressure.
    afcn: computes total Helmholtz energy.
    dafcn_drho : computes total Helmholtz energy and its density derivative.
    d2afcn_drho : computes total Helmholtz energy and it density derivatives.
    pressure : computes the pressure.
    dP_drho : computes pressure and its density derivative.
    logfug : computes the fugacity coefficient.
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential.
    dOm : computes adimentional Thermodynamic Grand Potential.
    ci :  computes influence parameters matrix for SGT.
    sgt_adim : computes adimentional factors for SGT.

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

    def __init__(self, pure):

        self.ms = pure.ms
        self.sigma = pure.sigma
        self.eps = pure.eps
        self.ring = pure.ring
        self.lambda_a = pure.lambda_a
        self.lambda_r = pure.lambda_r
        self.lambda_ar = self.lambda_r + self.lambda_a

        dif_c = self.lambda_r - self.lambda_a
        expc = self.lambda_a/dif_c
        self.c = self.lambda_r/dif_c*(self.lambda_r/self.lambda_a)**expc
        alpha = self.c*(1/(self.lambda_a - 3) - 1/(self.lambda_r - 3))
        self.alpha = alpha

        self.sigma3 = pure.sigma**3

        self.f1 = fi(alpha, 1)
        self.f2 = fi(alpha, 2)
        self.f3 = fi(alpha, 3)
        self.f4 = fi(alpha, 4)
        self.f5 = fi(alpha, 5)
        self.f6 = fi(alpha, 6)

        roots, weights = gauss(100)
        self.roots = roots
        self.weights = weights

        self.umie = U_mie(1./roots, self.c, self.eps, self.lambda_r,
                          self.lambda_a)

        c_matrix = np.array([[0.81096, 1.7888, -37.578, 92.284],
                            [1.0205, -19.341, 151.26, -463.5],
                            [-1.9057, 22.845, -228.14, 973.92],
                            [1.0885, -6.1962, 106.98, -677.64]])

        lam_exp = np.array([0, -1, -2, -3])

        self.cctes_lr = np.matmul(c_matrix, self.lambda_r**lam_exp)
        self.cctes_la = np.matmul(c_matrix, self.lambda_a**lam_exp)
        self.cctes_lar = np.matmul(c_matrix, self.lambda_ar**lam_exp)
        self.cctes_2lr = np.matmul(c_matrix, (2*self.lambda_r)**lam_exp)
        self.cctes_2la = np.matmul(c_matrix, (2*self.lambda_a)**lam_exp)
        self.cctes = (self.cctes_la, self.cctes_lr,
                      self.cctes_2la, self.cctes_2lr, self.cctes_lar)

        # association configuration
        self.eABij = pure.eAB
        self.rcij = pure.rcAB
        self.rdij = pure.rdAB
        self.sites = pure.sites
        S, DIJ, indexabij, nsites, diagasso = association_config(self)
        assoc_bool = nsites != 0
        self.assoc_bool = assoc_bool
        if assoc_bool:
            self.S = S
            self.DIJ = DIJ
            self.indexabij = indexabij
            self.nsites = nsites
            self.diagasso = diagasso

        # Polar Contribution
        self.mupol = pure.mupol
        self.npol = pure.npol
        polar_bool = self.npol != 0
        self.polar_bool = polar_bool
        if polar_bool:
            mpol = self.ms * (self.ms < 2) + 2 * (self.ms > 2)
            self.mpol = mpol
            aux1 = np.array([1, (mpol-1)/mpol, (mpol-1)/mpol*(mpol-2)/mpol])
            self.anij = aij@aux1
            self.bnij = bij@aux1
            self.cnijk = cij@aux1

            # 1 D = 3.33564e-30 C * m
            # 1 C^2 = 9e9 N m^2
            cte = (3.33564e-30)**2 * (9e9)
            self.mupolad2 = self.mupol**2*cte/(self.ms*self.eps*self.sigma3)

        # For SGT Computations
        self.cii = np.array(pure.cii, ndmin=1)

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
        cii : float
            correlated influence parameter [J m^5 / mol^2]
        """
        cii = self.ms * (0.12008072630855947 + 2.2197907527439655 * self.alpha)
        cii *= np.sqrt(Na**2 * self.eps * self.sigma**5)
        cii **= 2
        if overwrite:
            self.cii = cii
        return cii

    def d(self, beta):
        """
        d(beta)

        Method that computes the diameter of the fluid at given
        beta = 1 / kb T

        Journal of Chemical Physics, 139(15), 1–37 (2013)
        Eq. (7)

        Parameters
        ----------

        beta : float
            Boltzmann's factor: beta = 1 / kb T [1/J]

        Returns
        -------
        d : float
            computed diameter [m]
        """

        integrer = np.exp(-beta * self.umie)
        d = self.sigma * (1. - np.dot(integrer, self.weights))
        return d

    def eta_sigma(self, rho):
        """
        eta_sigma(rho)

        Method that computes packing fraction of the fluid at diameter=sigma.

        Parameters
        ----------

        rho : float
            molecular density [molecules/m^3]

        Returns
        -------
        eta : float
            packing fraction [Adim]
        """
        return self.ms * rho * np.pi * self.sigma**3 / 6

    def eta_bh(self, rho, d):
        """
        eta_sigma(rho, d)

        Method that computes packing fraction of the fluid at given diameter.

        Parameters
        ----------

        rho : float
            molecular density [molecules/m^3]
        d : float
            diameter [m]

        Returns
        -------
        eta : float
            packing fraction [Adim]
        deta : float
            derivative of packing fraction respect to density [m^3]
        """
        deta_drho = self.ms * np.pi * d**3 / 6
        eta = deta_drho * rho
        return eta, deta_drho

    def temperature_aux(self, T):
        """
        temperature_aux(T)

        Method that computes temperature dependent parameters.
        It returns the following list:

        temp_aux = [beta, dia, tetha, x0, x03, Fab, epsa]

        Journal of Chemical Physics, 139(15), 1–37 (2013)

        beta: Boltzmann's factor [1/J]
        dia: computed diameter [m] (Eq 7)
        tetha: exp(beta*eps)-1 [Adim] (Below Eq. 63)
        x0: sigma/dia [Adim] (Below Eq. 17)
        x03: x0^3 [Adim]
        Fab: association strength [Adim] (Below Eq. 77)
        epsa: eps / kb / T [Adim]

        Parameters
        ----------

        T : float
            Absolute temperature [K]

        Returns
        -------
        temp_aux : list
             list of computed parameters
        """

        beta = 1 / (kb*T)
        dia = self.d(beta)
        tetha = np.exp(beta*self.eps)-1
        x0 = self.sigma/dia
        x03 = x0**3
        # For Association
        Fab = np.exp(beta * self.eABij) - 1
        # For polar
        epsa = self.eps / T / kb

        temp_aux = [beta, dia, tetha, x0, x03, Fab, epsa]
        return temp_aux

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

    def psat(self, T, P0=None, v0=[None, None], Xass0=[None, None]):
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

        Returns
        -------
        psat : float
            saturation pressure
        """
        P, vl, vv = psat(self, T, P0, v0, Xass0)
        return P, vl, vv

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

        Tfactor = 1
        Pfactor = 1
        rofactor = 1
        tenfactor = np.sqrt(self.cii) * 1000  # To give tension in mN/m
        zfactor = 10**-10

        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def sgt_adim_fit(self, T):

        Tfactor = 1
        Pfactor = 1
        rofactor = 1
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

    def speed_sound(self, T, P, state, v0=None, Mw=1.):

        if v0 is None:
            rho = self.density(T, P, state, None)
        else:
            rho0 = 1./v0
            rho = self.density(T, P, state, rho0)
        rhomolecular = Na * rho

        h = 1e-2
        a2 = self.dafcn_drho(rhomolecular, T + 2*h)
        a1 = self.dafcn_drho(rhomolecular, T + h)
        d2a = self.d2afcn_drho(rhomolecular, T)
        a_1 = self.dafcn_drho(rhomolecular, T - h)
        a_2 = self.dafcn_drho(rhomolecular, T - 2*h)

        da_dt = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
        da_dt[1] *= Na

        a = d2a[:2]
        da_drho = a[1] * Na
        d2a_drho = d2a[2] * Na**2

        d2a_dt = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12) / h**2
        d2a_dt[1] *= Na

        dP_dt = rho**2 * da_dt[1]
        dP_drho = 2 * rho * da_drho + rho**2 * d2a_drho
        dP_dV = -rho**2 * dP_drho
        Cv = - T * d2a_dt[0]
        Cp = Cv - T*dP_dt**2 / dP_dV

        beta = -rho * (Cv/Cp) / dP_dV

        w2 = 1000./(rho * beta * Mw)
        w = np.sqrt(w2)
        return w
