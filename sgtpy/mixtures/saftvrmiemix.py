import numpy as np
from ..math import gauss

from .ares import ares, dares_drho, d2ares_drho, dares_dx, dares_dxrho

from .ideal import aideal, daideal_drho, d2aideal_drho, daideal_dx
from .ideal import daideal_dxrho

from .association_aux import association_config
from .polarGV import aij, bij, cij

from .density_solver import density_topliss, density_newton
from ..constants import kb, Na


# from .ares import ares as ares2
# from .ares import dares_drho as dares_drho2
# from .ares import d2ares_drho as d2ares_drho2
# from .ares import dares_dx as dares_dx2
# from .ares import dares_dxrho as dares_dxrho2

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

    def __init__(self, mixture):

        # Pure component parameters
        self.lr = np.asarray(mixture.lr)
        self.la = np.asarray(mixture.la)
        self.lar = self.lr + self.la
        self.sigma = np.asarray(mixture.sigma)
        self.eps = np.asarray(mixture.eps)
        self.ms = np.asarray(mixture.ms)

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

        self.diag_index = np.diag_indices(self.nc)
        self.C = self.Cij[self.diag_index]
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
        roots, weights = gauss(100)
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
            self.mpol = ms * (ms < 2) + 2 * (ms > 2)
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
        cii = self.ms * (0.12008072630855947 + 2.2197907527439655 * self.alpha)
        cii *= np.sqrt(Na**2 * self.eps * self.sigma**5)
        cii **= 2
        if overwrite:
            self.cii = cii
            self.cij = np.sqrt(np.outer(self.cii, self.cii))
        return cii

    def diameter(self, beta):
        # umie = U_mie(1/roots, c, eps, lambda_r, lambda_a)
        integrer = np.exp(-beta * self.umie)
        d = self.sigma * (1. - np.matmul(self.weights, integrer))
        return d

    def temperature_aux(self, T):

        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        # Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        # Defining xhi0 without x dependence
        di03 = np.power.outer(dii, np.arange(4))
        dij3 = dij**3

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
        a1vdw_la = a1vdw_laij[diag_index]
        a1vdw_lr = a1vdw_lrij[diag_index]
        a1vdw_2la = a1vdw_2laij[diag_index]
        a1vdw_2lr = a1vdw_2lrij[diag_index]
        a1vdw_lar = a1vdw_larij[diag_index]
        a1vdw = (a1vdw_la, a1vdw_lr, a1vdw_2la, a1vdw_2lr, a1vdw_lar)

        tetha = np.exp(beta * self.eps) - 1.
        # For associating mixtures
        Fab = np.exp(beta * self.eABij) - 1.
        # For Polar mixtures
        epsa = self.eps / kb / T
        epsija = self.epsij / kb / T
        temp_aux = [beta, dii, dij, x0, x0i, di03, dij3, a1vdw_cteij, a1vdwij,
                    tetha, a1vdw_cte, a1vdw, Fab, epsa, epsija]
        return temp_aux

    def ares(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = ares(self, x, rho, temp_aux, Xass0)
        return a

    def dares_drho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = dares_drho(self, x, rho, temp_aux, Xass0)
        return a

    def d2ares_drho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = d2ares_drho(self, x, rho, temp_aux, Xass0)
        return a

    def dares_dx(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = dares_dx(self, x, rho, temp_aux, Xass0)
        return a, ax

    def dares_dxrho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = dares_dxrho(self, x, rho, temp_aux, Xass0)
        return a, ax

    def afcn_aux(self, x, rho, temp_aux, Xass0=None):
        beta = temp_aux[0]
        a, Xass = ares(self, x, rho, temp_aux, Xass0)
        a += aideal(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def dafcn_drho_aux(self, x, rho, temp_aux, Xass0=None):
        beta = temp_aux[0]
        a, Xass = dares_drho(self, x, rho, temp_aux, Xass0)
        a += daideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def d2afcn_drho_aux(self, x, rho, temp_aux, Xass0=None):
        beta = temp_aux[0]
        a, Xass = d2ares_drho(self, x, rho, temp_aux, Xass0)
        a += d2aideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a, Xass

    def dafcn_dx_aux(self, x, rho, temp_aux, Xass0=None):
        beta = temp_aux[0]
        ar, aresx, Xass = dares_dx(self, x, rho, temp_aux, Xass0)
        aideal, aidealx = daideal_dx(x, rho, beta)
        a = (ar + aideal)
        a *= (Na/beta)
        ax = (aresx + aidealx)
        ax *= (Na/beta)
        return a, ax, Xass

    def dafcn_dxrho_aux(self, x, rho, temp_aux, Xass0=None):
        beta = temp_aux[0]
        ar, aresx, Xass = dares_dxrho(self, x, rho, temp_aux, Xass0)
        aideal, aidealx = daideal_dxrho(x, rho, beta)
        a = (ar + aideal)
        a *= (Na/beta)
        ax = (aresx + aidealx)
        ax *= (Na/beta)
        return a, ax, Xass

    def afcn(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = self.afcn_aux(x, rho, temp_aux, Xass0)
        return a

    def dafcn_drho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = self.dafcn_drho_aux(x, rho, temp_aux, Xass0)
        return a

    def d2afcn_drho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, Xass = self.d2afcn_drho_aux(x, rho, temp_aux, Xass0)
        return a

    def dafcn_dx(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = self.dafcn_dx_aux(x, rho, temp_aux, Xass0)
        return a, ax

    def dafcn_dxrho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a, ax, Xass = self.dafcn_dx_aux(x, rho, temp_aux, Xass0)
        return a, ax

    def density_aux(self, x, temp_aux, P, state, rho0=None, Xass0=None):
        if rho0 is None:
            rho, Xass = density_topliss(state, x, temp_aux, P, Xass0, self)
        else:
            rho, Xass = density_newton(rho0, x, temp_aux, P, Xass0, self)
        return rho, Xass

    def density(self, x, T, P, state, rho0=None, Xass0=None):
        temp_aux = self.temperature_aux(T)
        rho, Xass = self.density_aux(x, temp_aux, P, state, rho0, Xass0)
        return rho

    def pressure_aux(self, x, rho, temp_aux, Xass0=None):
        rhomolecular = Na * rho
        da, Xass = self.dafcn_drho_aux(x, rhomolecular, temp_aux, Xass0)
        afcn, dafcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        return Psaft, Xass

    def dP_drho_aux(self, x, rho, temp_aux, Xass0=None):
        rhomolecular = Na * rho
        da, Xass = self.d2afcn_drho_aux(x, rhomolecular, temp_aux, Xass0)
        afcn, dafcn, d2afcn = da
        Psaft = rhomolecular**2 * dafcn / Na
        dPsaft = 2 * rhomolecular * dafcn + rhomolecular**2 * d2afcn
        return Psaft, dPsaft, Xass

    def pressure(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        Psaft, Xass = self.pressure_aux(x, rho, temp_aux, Xass0)
        return Psaft

    def dP_drho(self, x, rho, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        Psaft, dPsaft, Xass = self.dP_drho_aux(x, rho, temp_aux, Xass0)
        return Psaft, dPsaft

    def logfugmix_aux(self, x, temp_aux, P, state, v0=None, Xass0=None):
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
        temp_aux = self.temperature_aux(T)
        lnphi, v, Xass = self.logfugmix_aux(x, temp_aux, P, state, v0, Xass0)
        return lnphi, v

    def logfugef_aux(self, x, temp_aux, P, state, v0=None, Xass0=None):

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
        temp_aux = self.temperature_aux(T)
        lnphi, v, Xass = self.logfugef_aux(x, temp_aux, P, state, v0, Xass0)
        return lnphi, v

    def a0ad_aux(self, rhoi, temp_aux, Xass0=None):

        rho = np.sum(rhoi)
        x = rhoi/rho
        rhomolecular = Na * rho
        beta = temp_aux[0]

        a, Xass = ares(self, x, rhomolecular, temp_aux, Xass0)
        a += aideal(x, rhomolecular, beta)

        a0 = a*rho

        return a0, Xass

    def a0ad(self, rhoi, T, Xass0=None):
        temp_aux = self.temperature_aux(T)
        a0, Xass = self.a0ad_aux(rhoi, temp_aux, Xass0)
        return a0

    def muad_aux(self, rhoi, temp_aux, Xass0=None):
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
        temp_aux = self.temperature_aux(T)
        mu, Xass = self.muad_aux(rhoi, temp_aux, Xass0)
        return mu

    def dmuad_aux(self, rhoi, temp_aux, Xass0=None):
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
        temp_aux = self.temperature_aux(T)
        mu, dmu, Xass = self.dmuad_aux(rhoi, temp_aux, Xass0)
        return mu, dmu

    def dOm_aux(self, rhoi, temp_aux, mu, Psat, Xass0=None):
        a0ad, Xass = self.a0ad_aux(rhoi, temp_aux, Xass0)
        dom = a0ad - np.sum(np.nan_to_num(rhoi*mu)) + Psat
        return dom, Xass

    def dOm(self, rhoi, T, mu, Psat, Xass0=None):
        temp_aux = self.temperature_aux(T)
        dom, Xass = self.dOm_aux(rhoi, temp_aux, mu, Psat, Xass0)
        return dom, Xass

    def sgt_adim(self, T):
        beta = 1 / (kb*T)
        RT = (Na/beta)

        Tfactor = 1.
        Pfactor = 1. / RT
        rofactor = 1.
        tenfactor = np.sqrt(self.cii[0]*RT) * 1000  # To give tension in mN/m
        zfactor = 10**-10 * np.sqrt(RT / self.cii[0])
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def beta_sgt(self, beta):
        self.beta = beta

    def ci(self, T):
        return self.cij * (1 - self.beta)
