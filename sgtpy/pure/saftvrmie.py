import numpy as np
from ..math import gauss
from .ideal import aideal, daideal_drho, d2aideal_drho
from .association_aux import association_config
from .polarGV import aij, bij, cij
from .monomer_aux import I_lam, J_lam
from .a1sB_monomer import x0lambda_eval
from .ares import ares, dares_drho, d2ares_drho
from .density_solver import density_topliss, density_newton
from .psat_saft import psat
from .tsat_saft import tsat
from .critical_pure import get_critical

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
    cii_correlation : correlates the influence parameter of the fluid
    diameter : computes the diameter at given temperature
    temperature_aux : computes temperature dependent parameters of the fluid
    density : computes the density of the fluid
    psat : computes saturation pressure
    tsat : computes saturation temperature
    get_critical : attemps to compute the critical point of the fluid
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

    def __init__(self, pure):

        self.pure = pure
        self.Mw = pure.Mw
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
        self.c2 = self.c**2
        alpha = self.c*(1/(self.lambda_a - 3) - 1/(self.lambda_r - 3))
        self.alpha = alpha

        self.lambdas = self.lambda_a, self.lambda_r, self.lambda_ar

        self.sigma3 = pure.sigma**3

        self.cte_a2m = 0.5*self.eps*self.c2
        self.eps3 = self.eps**3

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
        self.rcij2 = self.rcij**2
        self.rcij3 = self.rcij**3
        self.rdij2 = self.rdij**2
        self.rdij3 = self.rdij**3

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
            mpol = self.ms * (self.ms < 2) + 2 * (self.ms >= 2)
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

    def diameter(self, beta):
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
        return self.ms * rho * np.pi * self.sigma3 / 6

    def eta_bh(self, rho, dia3):
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
        deta_drho = self.ms * np.pi * dia3 / 6
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
        beta2 = beta**2
        beta3 = beta2*beta
        dia = self.diameter(beta)
        dia3 = dia**3

        x0 = self.sigma/dia
        x03 = x0**3

        # Parameters needed for evaluating the helmothlz contributions
        la, lr, lar = self.lambda_a, self.lambda_r, self.lambda_ar
        out = x0lambda_eval(x0, la, lr, lar)
        x0_a1, x0_a2, x0_a12, x0_a22 = out

        I_la = I_lam(x0, la)
        I_lr = I_lam(x0, lr)
        I_2la = I_lam(x0, 2*la)
        I_2lr = I_lam(x0, 2*lr)
        I_lar = I_lam(x0, lar)
        I_lambdas = (I_la, I_lr, I_2la, I_2lr, I_lar)

        J_la = J_lam(x0, la)
        J_lr = J_lam(x0, lr)
        J_2la = J_lam(x0, 2*la)
        J_2lr = J_lam(x0, 2*lr)
        J_lar = J_lam(x0, lar)
        J_lambdas = (J_la, J_lr, J_2la, J_2lr, J_lar)

        # for chain contribution
        beps = beta*self.eps
        beps2 = beps**2
        tetha = np.exp(beps)-1
        x0_vector = np.array([1, x0, x0**2, x0**3])

        cte_g1s = 1/(2*np.pi*self.eps*self.ms*dia3)
        cte_g2s = cte_g1s / self.eps
        # For Association
        Fab = np.exp(beta * self.eABij) - 1
        rc, rc2, rc3 = self.rcij, self.rcij2, self.rcij3
        rd, rd2, rd3 = self.rdij, self.rdij2, self.rdij3
        dia2 = dia**2

        Kab = np.log((rc + 2*rd)/dia)
        Kab *= 6*rc3 + 18 * rc2*rd - 24 * rd3
        aux1 = (rc + 2 * rd - dia)
        aux2 = (22*rd2 - 5*rc*rd - 7*rd*dia - 8*rc2 + rc*dia + dia2)
        Kab += aux1 * aux2
        Kab /= (72*rd2 * self.sigma3)
        Kab *= 4 * np.pi * dia2

        # For polar
        epsa = self.eps / T / kb

        temp_aux = [beta, beta2, beta3, dia, dia3, x0, x03, x0_a1, x0_a2,
                    x0_a12, x0_a22, I_lambdas, J_lambdas, beps, beps2, tetha,
                    x0_vector, cte_g1s, cte_g2s, Fab, Kab, epsa]
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

    def get_critical(self, Tc0, rhoc0, method='hybr', full_output=False):
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
            whether to outputs or not all the calculation info.

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
        return out

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
