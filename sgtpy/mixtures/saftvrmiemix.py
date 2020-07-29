import numpy as np
from ..math import gauss

from .monomer_aux import dkHS_dxhi00, d2kHS_dxhi00, d3kHS_dxhi00
from .monomer_aux import dkHS_dx_dxhi00, dkHS_dx_dxhi002

from .a1sB_monomer import x0lambda_eval
from .a1sB_monomer import da1sB_dxhi00_eval, d2a1sB_dxhi00_eval
from .a1sB_monomer import d3a1sB_dxhi00_eval
from .a1sB_monomer import da1sB_dx_eval, da1sB_dx_dxhi00_eval

from .ideal import aideal, daideal_drho, d2aideal_drho, daideal_dx
from .ideal import  daideal_dxrho

from .a1_monomer import a1, da1_dxhi00, d2a1_dxhi00,  da1_dx, da1_dxxhi
from .a2_monomer import a2, da2_dxhi00, d2a2_dxhi00,  da2_dx, da2_dxxhi
from .ahs_monomer import ahs, dahs_dxhi00, d2ahs_dxhi00,  dahs_dx, dahs_dxxhi
from .a3_monomer import a3, da3_dxhi00, d2a3_dxhi00,  da3_dx, da3_dxxhi

from .ghs_chain import gdHS, dgdHS_dxhi00, d2gdHS_dxhi00, dgdHS_dx, dgdHS_dxxhi
from .g2mca_chain import g2mca, dg2mca_dxhi00, d2g2mca_dxhi00, dg2mca_dx, dg2mca_dxxhi
from .gammac_chain import gammac, dgammac_dxhi00, d2gammac_dxhi00, dgammac_dx, dgammac_dxxhi
from .g1sigma_chain import g1sigma, dg1sigma_dxhi00, d2g1sigma_dxhi00, dg1sigma_dx, dg1sigma_dxxhi
from .a2new_chain import da2new_dxhi00, d2a2new_dxhi00, d3a2new_dxhi00, da2new_dx_dxhi00, da2new_dxxhi_dxhi00
from .lngmie_chain import lngmie, dlngmie_dxhi00, d2lngmie_dxhi00, dlngmie_dx, dlngmie_dxxhi

from .association_aux import association_config, Iab, dIab_drho, d2Iab_drho, Xass_solver
from .association_aux import dIab_dx, dIab_dxrho, CIJ_matrix, dXass_drho, d2Xass_drho, dXass_dx

from .polarGV import aij, bij, cij
from .polarGV import Apolar, dApolar_drho, d2Apolar_drho
from .polarGV import dApolar_dx, dApolar_dxrho

from .density_solver import density_topliss, density_newton
from ..constants import kb, Na

R = Na * kb

def U_mie(r, c, eps, lambda_r, lambda_a):
    u = c * eps * (np.power.outer(r, lambda_r) - np.power.outer(r, lambda_a))
    return u


def xhi_eval(xhi00, xs, xmi, xm, di03):
    xhi = xhi00 * xm * np.matmul(xs,di03)
    dxhi_dxhi00 = np.matmul(xmi, di03)
    dxhi_dxhi00[0] = xm
    return xhi, dxhi_dxhi00


def xhix_eval(xhi00, xs, xm, dij3):
    aux1 = xs * dij3
    aux2 = np.dot(xs, aux1)
    aux3 = aux2.sum()
    dxhix_dxhi00 = xm * aux3
    xhix = xhi00 * dxhix_dxhi00
    return xhix, dxhix_dxhi00


def dxhi_dx_eval(xhi00, xs, xmi, xm, ms, di03):
    xhi = xhi00 * xm * np.matmul(xs,di03)
    dxhi_dxhi00 = np.matmul(xmi,di03)
    dxhi_dxhi00[0] = xm
    dxhi_dx = (xhi00 * di03.T * ms)
    return xhi, dxhi_dxhi00, dxhi_dx


def dxhix_dx_eval(xhi00, xs, dxs_dx, xm, ms, dij3):
    aux1 = xs * dij3
    aux2 = np.dot(xs, aux1)
    aux3 = aux2.sum()
    dxhix_dxhi00 = xm * aux3
    xhix = xhi00 * dxhix_dxhi00
    suma1 = 2*np.sum(dxs_dx.T@aux1, axis=1)
    dxhix_dx_dxhi00 = (ms * aux3 + xm * suma1)
    dxhix_dx = xhi00 * dxhix_dx_dxhi00
    return xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00

#Second perturbation
phi16 = np.array([[7.5365557, -37.60463, 71.745953, -46.83552, -2.467982, -0.50272, 8.0956883],
[-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.7090],
[1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0],
[-1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377, 40.53683],
[-1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61, 2556.181],
[9236.9, -129430., 357230., -315530., 1390.2, -4518.2, 4241.6]])

nfi = np.arange(0,7)
nfi_num = nfi[:4]
nfi_den = nfi[4:]


# Eq A26
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
        self.epsij = np.sqrt(np.multiply.outer(sigma3, sigma3) * np.multiply.outer(self.eps, self.eps))
        self.epsij /= self.sigmaij3
        self.epsij *= (1-kij)

        # Eq A48
        self.lrij = np.sqrt(np.multiply.outer(self.lr-3., self.lr-3.)) + 3
        self.laij = np.sqrt(np.multiply.outer(self.la-3., self.la-3.)) + 3
        self.larij = self.lrij + self.laij


        # Eq A3
        dif_cij = self.lrij - self.laij
        self.Cij = self.lrij / dif_cij * (self.lrij / self.laij) ** (self.laij / dif_cij)
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
            self.mpol = self.ms * (self.ms < 2) + 2 * (self.ms > 2)
            msij = np.sqrt(np.outer(self.ms, self.ms))
            msijk = np.cbrt(np.multiply.outer(self.ms, np.outer(self.ms, self.ms)))
            msij[msij > 2.] = 2.
            msijk[msijk > 2.] = 2.

            aux1 = np.array([np.ones_like(msij), (msij -1 )/msij, (msij -1)/msij * (msij-2)/msij])
            aux2 = np.array([np.ones_like(msijk), (msijk -1 )/msijk, (msijk -1)/msijk * (msijk-2)/msijk])
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
            self.mupolad2 = self.mupol**2 * cte / (self.ms * self.eps * self.sigma3)

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
        #umie = U_mie(1/roots, c, eps, lambda_r, lambda_a)
        integrer = np.exp(-beta * self.umie)
        d = self.sigma * (1. - np.matmul(self.weights, integrer))
        return d

    def ares(self, x, rho, T):

        dxhi00_drho = self.dxhi00_drho
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        #Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        xmi = x * self.ms
        xm = np.sum(xmi)
        #Eq A8
        xs = xmi / xm

        #definiendo xhi0 sin dependencia de x
        di03 = np.power.outer(dii, np.arange(4))
        xhi00 = dxhi00_drho * rho

        #Eq A7
        xhi, dxhi_dxhi00=  xhi_eval(xhi00, xs, xmi, xm, di03)
        #xhi x Eq A13
        dij3 = dij**3
        xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)

        #xhi m Eq A23
        xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

        #Terminos necesarios monomero
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij)


        x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr, self.lar,
                                                   self.laij, self.lrij, self.larij, diag_index)


        da1, da2  = da1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                      self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00)

        suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis = 1)
        suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis = 1)

        khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)

        #Evaluacion monomero
        a1ij = suma1_monomer[0]
        a2ij = suma2_monomer[0]
        aHS = ahs(xhi) #solo valor
        a1m = a1(xs, a1ij) #solo valor
        a2m = a2(xs, khs, xhixm, a2ij, self.epsij, self.f1, self.f2, self.f3) #solo valor
        a3m = a3(xs, xhixm, self.epsij, self.f4, self.f5, self.f6)
        am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
        amono = xm * am

        #USARLOS PARA LOS A1 Y A2 DE LA CADENA
        suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
        suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

        da1c = da1[:, :, diag_index[0], diag_index[1]]
        da2c = da2[:, :, diag_index[0], diag_index[1]]

        #USARLOS EN LA SUMATORIA A1SB DE LA CADENA
        suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis = 1)
        suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis = 1)

        tetha = np.exp(beta * self.eps) - 1.

        #Evaluacion cadena
        gHS = gdHS(x0i, xhix) #casi exacto
        gc = gammac(xhixm, self.alpha, tetha) #casi exacto

        da1ii = suma1_chain[1]
        a1sB = suma1_chain2[0]
        g1s = g1sigma(xhi00, xm, da1ii, a1sB, self.eps, dii) #casi exacto

        da2new = da2new_dxhi00(khs, dkhs, suma2_chain[:2], self.eps) #casi exacto
        suma_a2 = suma2_chain2[0] #casi exacto
        g2m = g2mca(xhi00, khs, xm, da2new, suma_a2, self.eps, dii) #casi exacto
        g2s = (1 + gc) * g2m

        lng = lngmie(gHS, g1s, g2s, beta, self.eps)
        achain = - np.dot(x * (self.ms - 1), lng)

        ares = amono + achain

        if self.assoc_bool:
            xj = x[self.compindex]
            iab = Iab(xhi, dii, dij, self.rcij, self.rdij, self.sigmaij3)

            Fab = np.exp(beta * self.eABij) - 1.
            Dab = self.sigmaij3 * Fab * iab
            Dabij = np.zeros([self.nsites, self.nsites])
            Dabij[self.indexabij] = Dab[self.indexab]
            KIJ = rho * np.outer(xj, xj) * (self.DIJ * Dabij)
            Xass = Xass_solver(self.nsites, xj, KIJ, self.diagasso, Xass0 = None)
            ares += np.dot(self.S * xj, (np.log(Xass) - Xass/2 + 1/2))

        if self.polar_bool:
            eta = xhi[-1]
            apolar = Apolar(rho, x, T, self.anij, self.bnij, self.cnij,
            eta, self.eps, self.epsij, self.sigma3, self.sigmaij3,
            self.sigmaijk3,  self.npol, self.mupolad2)
            ares += apolar

        return ares

    def dares_drho(self, x, rho, T):

        dxhi00_drho = self.dxhi00_drho
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        #Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        xmi = x * self.ms
        xm = np.sum(xmi)
        #Eq A8
        xs = xmi / xm

        #definiendo xhi0 sin dependencia de x
        di03 = np.power.outer(dii, np.arange(4))
        xhi00 = dxhi00_drho * rho

        #Eq A7
        xhi, dxhi_dxhi00=  xhi_eval(xhi00, xs, xmi, xm, di03)

        #xhi x Eq A13
        dij3 = dij**3
        xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)

        #xhi m Eq A23
        xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

        #Terminos necesarios monomero
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij)

        x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr, self.lar,
                                                   self.laij, self.lrij, self.larij, diag_index)


        da1, da2  = d2a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                      self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00)

        suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis = 1)
        suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis = 1)

        khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)

        #Evaluacion monomero
        a1ij = suma1_monomer[:2]
        a2ij = suma2_monomer[:2]
        aHS = dahs_dxhi00(xhi, dxhi_dxhi00) #valor y derivadas correctas
        a1m = da1_dxhi00(xs, a1ij) #valor y derivadas correctas
        a2m = da2_dxhi00(xs, khs, dkhs, xhixm, dxhixm_dxhi00, a2ij,
                         self.epsij, self.f1, self.f2, self.f3) #valor y derivadas correctas
        a3m = da3_dxhi00(xs, xhixm, dxhixm_dxhi00, self.epsij, self.f4, self.f5, self.f6) #valor y derivada correcta
        am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
        amono = xm * am #valor y derivada correcta

        #USARLOS PARA LOS A1 Y A2 DE LA CADENA
        suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
        suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

        da1c = da1[:, :, diag_index[0], diag_index[1]]
        da2c = da2[:, :, diag_index[0], diag_index[1]]

        #USARLOS EN LA SUMATORIA A1SB DE LA CADENA
        suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis = 1)
        suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis = 1)

        tetha = np.exp(beta * self.eps) - 1.

        #Evaluacion cadena
        gHS = dgdHS_dxhi00(x0i, xhix, dxhix_dxhi00) #valor y derivada exacta
        gc = dgammac_dxhi00(xhixm, dxhixm_dxhi00, self.alpha, tetha) #valor y derivada exacta

        da1ii = suma1_chain[1:3] #primera derivada exacta
        a1sB = suma1_chain2[:2] #valor y derivadas exactas
        g1s = dg1sigma_dxhi00(xhi00, xm, da1ii, a1sB, self.eps, dii) #valor y derivada exacta

        #da2new = da2new_dxhi00(xs, xhix, dxhix_dxhi00, suma2_chain[:2], self.eps) #exacto
        da2new, d2a2new = d2a2new_dxhi00(khs, dkhs, d2khs, suma2_chain[:3], self.eps) #no exacto
        suma_a2 = suma2_chain2[[0,1]] #valor y derivadas exactas
        g2m = dg2mca_dxhi00(xhi00, khs, dkhs, xm, da2new, d2a2new,
                            suma_a2, self.eps, dii)
        g2s = g2m * (1 + gc[0])
        g2s[1] += g2m[0] * gc[1]
        #be = beta * self.eps
        lng = dlngmie_dxhi00(gHS, g1s, g2s, beta, self.eps)
        achain = - lng@(x * (self.ms - 1))

        ares = amono + achain
        ares *= np.array([1, dxhi00_drho])

        if self.assoc_bool:
            xj = x[self.compindex]
            iab, diab = dIab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii, dij,
                            self.rcij, self.rdij, self.sigmaij3)
            Fab = np.exp(beta * self.eABij) - 1.
            Dab = self.sigmaij3 * Fab * iab
            dDab_drho = self.sigmaij3 * Fab * diab
            Dabij = np.zeros([self.nsites, self.nsites])
            dDabij_drho = np.zeros([self.nsites, self.nsites])
            Dabij[self.indexabij] = Dab[self.indexab]
            dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
            KIJ = rho * np.outer(xj, xj) * (self.DIJ * Dabij)
            Xass = Xass_solver(self.nsites, xj, KIJ, self.diagasso, Xass0 = None)
            CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
            dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
            ares[0] += np.dot(self.S * xj, (np.log(Xass) - Xass/2 + 1/2))
            ares[1] += np.dot(self.S*xj, (1/Xass - 1/2) * dXass)

        if self.polar_bool:
            eta = xhi[-1]
            deta_dxhi00 = dxhi_dxhi00[-1]
            deta = deta_dxhi00 * self.dxhi00_drho
            dapolar = dApolar_drho(rho, x, T, self.anij, self.bnij, self.cnij,
            eta, deta, self.eps, self.epsij, self.sigma3, self.sigmaij3,
            self.sigmaijk3,  self.npol, self.mupolad2)
            ares += dapolar

        return ares


    def d2ares_drho(self, x, rho, T):

        dxhi00_drho = self.dxhi00_drho
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        #Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        xmi = x * self.ms
        xm = np.sum(xmi)
        #Eq A8
        xs = xmi / xm

        #definiendo xhi0 sin dependencia de x
        di03 = np.power.outer(dii, np.arange(4))
        xhi00 = dxhi00_drho * rho

        #Eq A7
        xhi, dxhi_dxhi00 =  xhi_eval(xhi00, xs, xmi, xm, di03)

        #xhi x Eq A13
        dij3 = dij**3
        xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)

        #xhi m Eq A23
        xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

        #Terminos necesarios monomero
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij)



        x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr, self.lar,
                                                   self.laij, self.lrij, self.larij, diag_index)


        da1, da2  = d3a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                      self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00)

        suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis = 1)
        suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis = 1)

        khs, dkhs, d2khs, d3khs = d3kHS_dxhi00(xhix, dxhix_dxhi00)

        #Evaluacion monomero
        suma_a1 = suma1_monomer[:3]
        suma_a2 = suma2_monomer[:3]

        aHS = d2ahs_dxhi00(xhi, dxhi_dxhi00)
        a1m = d2a1_dxhi00(xs,  suma_a1)
        a2m = d2a2_dxhi00(xs, khs, dkhs, d2khs, xhixm, dxhixm_dxhi00, suma_a2,
                         self.epsij, self.f1, self.f2, self.f3)
        a3m = d2a3_dxhi00(xs, xhixm, dxhixm_dxhi00, self.epsij, self.f4, self.f5, self.f6)
        am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
        amono = xm * am #valor y derivada correcta

        #USARLOS PARA LOS A1 Y A2 DE LA CADENA
        suma1_chain = suma1_monomer[:, diag_index[0], diag_index[1]]
        suma2_chain = suma2_monomer[:, diag_index[0], diag_index[1]]

        da1c = da1[:, :, diag_index[0], diag_index[1]]
        da2c = da2[:, :, diag_index[0], diag_index[1]]

        #USARLOS EN LA SUMATORIA A1SB DE LA CADENA
        sumg1_chain = self.C * np.sum(da1c * x0_g1, axis = 1)
        sumg2_chain = self.C**2 * np.sum(da2c * x0_g2, axis = 1)

        #Evaluacion cadena
        gHS = d2gdHS_dxhi00(x0i, xhix, dxhix_dxhi00)

        tetha = np.exp(beta * self.eps) - 1.
        gc = d2gammac_dxhi00(xhixm, dxhixm_dxhi00, self.alpha, tetha)

        da1ii = suma1_chain[1:4]
        suma_g1 = sumg1_chain[:3]
        g1s = d2g1sigma_dxhi00(xhi00, xm, da1ii, suma_g1, self.eps, dii)

        #da2new, d2a2new, d3a2new = d3a2new_dxhi00(khs, dkhs, d2khs, d3khs, suma2_chain[:4], self.eps)
        da2new, d2a2new, d3a2new = d3a2new_dxhi00(khs, dkhs, d2khs, d3khs, suma2_chain, self.eps)
        suma_g2 = sumg2_chain[[0,1,2]]
        g2m = d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xm, da2new, d2a2new, d3a2new,
                            suma_g2, self.eps, dii)

        g2s = g2m * (1 + gc[0])
        g2s[1] += g2m[0] * gc[1]
        g2s[2] += 2. *g2m[1] * gc[1] + g2m[0] * gc[2]

        lng = d2lngmie_dxhi00(gHS, g1s, g2s, beta, self.eps)
        achain = - lng@(x * (self.ms - 1))

        ares = amono + achain

        ares *= np.array([1, dxhi00_drho, dxhi00_drho**2])

        if self.assoc_bool:
            xj = x[self.compindex]
            iab, diab, d2iab = d2Iab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii, dij,
                                  self.rcij, self.rdij, self.sigmaij3)
            Fab = np.exp(beta * self.eABij) - 1.
            Dab = self.sigmaij3 * Fab * iab
            dDab_drho = self.sigmaij3 * Fab * diab
            d2Dab_drho = self.sigmaij3 * Fab * d2iab

            Dabij = np.zeros([self.nsites, self.nsites])
            dDabij_drho = np.zeros([self.nsites, self.nsites])
            d2Dabij_drho = np.zeros([self.nsites, self.nsites])

            Dabij[self.indexabij] = Dab[self.indexab]
            dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
            d2Dabij_drho[self.indexabij] = d2Dab_drho[self.indexab]

            KIJ = rho * np.outer(xj, xj) * (self.DIJ * Dabij)
            Xass = Xass_solver(self.nsites, xj, KIJ, self.diagasso, Xass0 = None)
            CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
            dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
            d2Xass = d2Xass_drho(rho, xj, Xass, dXass, self.DIJ, Dabij,
                                         dDabij_drho, d2Dabij_drho, CIJ)
            ares[0] += np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
            ares[1] += np.dot(self.S*xj, (1/Xass - 1/2) * dXass)
            ares[2] += np.dot(self.S*xj, -(dXass/Xass)**2 + d2Xass * (1/Xass - 1/2))

        if self.polar_bool:
            eta = xhi[-1]
            deta_dxhi00 = dxhi_dxhi00[-1]
            deta = deta_dxhi00 * self.dxhi00_drho
            dapolar = d2Apolar_drho(rho, x, T, self.anij, self.bnij, self.cnij,
            eta, deta, self.eps, self.epsij, self.sigma3, self.sigmaij3,
            self.sigmaijk3,  self.npol, self.mupolad2)
            ares += dapolar

        return ares

    def dares_dx(self, x, rho, T):

        dxhi00_drho = self.dxhi00_drho
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        #Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        xmi = x * self.ms
        xm = np.sum(xmi)
        #Eq A8
        xs = xmi / xm
        dxs_dx = - np.multiply.outer(self.ms * x, self.ms) / xm**2
        dxs_dx[diag_index] += self.ms / xm

        #definiendo xhi0 sin dependencia de x
        di03 = np.power.outer(dii, np.arange(4))
        xhi00 = dxhi00_drho * rho

        #Eq A7
        xhi, dxhi_dxhi00, dxhi_dx =  dxhi_dx_eval(xhi00, xs, xmi, xm, self.ms, di03)

        #xhi x Eq A13
        dij3 = dij**3
        xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = dxhix_dx_eval(xhi00, xs, dxs_dx, xm,
                                                                        self.ms, dij3)

        #xhi m Eq A23
        xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = dxhix_dx_eval(xhi00, xs, dxs_dx, xm,
                                                                        self.ms, self.sigmaij3)

        #Terminos necesarios monomero
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij)

        #Terminos necesarios cadena
        a1vdw_cte = a1vdw_cteij[diag_index]
        a1vdw_la = a1vdw_laij[diag_index]
        a1vdw_lr = a1vdw_lrij[diag_index]
        a1vdw_2la = a1vdw_2laij[diag_index]
        a1vdw_2lr = a1vdw_2lrij[diag_index]
        a1vdw_lar = a1vdw_larij[diag_index]
        a1vdw = (a1vdw_la, a1vdw_lr, a1vdw_2la, a1vdw_2lr, a1vdw_lar)

        x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr, self.lar,
                                                   self.laij, self.lrij, self.larij, diag_index)


        da1, da2  = da1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                      self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00)

        da1x, da2x  = da1sB_dx_eval(xhi00, xhix, x0, xm, self.ms, self.lambdasij,
                                    self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dx)

        da1xxhi, da2xxhi  = da1sB_dx_dxhi00_eval(xhi00, xhix, x0i, xm, self.ms, self.lambdas,
                         self.cctes, a1vdw, a1vdw_cte, dxhix_dxhi00, dxhix_dx,dxhix_dx_dxhi00)

        suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis = 1)
        suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis = 1)

        suma1_monomerx = self.Cij * (da1x[0] * x0_a1[0] + da1x[1] * x0_a1[1])
        suma2_monomerx = self.Cij**2 * (da2x[0] * x0_a2[0] + da2x[1] * x0_a2[1]
                                        + da2x[2] * x0_a2[2])

        da1ijx = suma1_monomerx
        da2ijx = suma2_monomerx

        da1x_a = da1x[0]
        da1x_r = da1x[1]

        da2x_2a = da2x[0]
        da2x_2r = da2x[2]
        da2x_ar = da2x[1]


        khs, dkhs, dkhsx, dkhsxxhi = dkHS_dx_dxhi00(xhix, dxhix_dxhi00,
                                                    dxhix_dx, dxhix_dx_dxhi00)

        aHS, daHS = dahs_dx(xhi, dxhi_dx)
        a1m, da1m = da1_dx(xs, dxs_dx, suma1_monomer[0], suma1_monomerx)
        a2m, da2m = da2_dx(xs, dxs_dx, khs, dkhsx, xhixm, dxhixm_dx, suma2_monomer[0],
                           suma2_monomerx, self.epsij, self.f1, self.f2, self.f3)
        a3m, da3m = da3_dx(xs, dxs_dx, xhixm, dxhixm_dx, self.epsij, self.f4, self.f5, self.f6)

        am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
        dam = daHS + beta * da1m + beta**2 * da2m + beta**3 * da3m
        amono = xm * am
        damonox = self.ms * am + xm * dam

        #USARLOS PARA LOS A1 Y A2 DE LA CADENA
        suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
        suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

        da1c = da1[:, :, diag_index[0], diag_index[1]]
        da2c = da2[:, :, diag_index[0], diag_index[1]]

        #USARLOS EN LA SUMATORIA A1SB DE LA CADENA
        suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis = 1)
        suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis = 1)

        #cadena
        gHS, dgHS = dgdHS_dx(x0i, xhix, dxhix_dx) #error e-12
        tetha = np.exp(beta * self.eps) - 1.
        gc, dgc = dgammac_dx(xhixm, dxhixm_dx, self.alpha, tetha)#e-12


        #Esto lo modifique
        dg1x_a = da1x_a[:, diag_index[0], diag_index[1]]

        dg1x_r = da1x_r[:, diag_index[0], diag_index[1]]

        da1 = 1.*suma1_chain[1]
        #dxa, dxr = da1xxhi
        da1x = da1xxhi[0]*x0_a1[0, diag_index[0], diag_index[1]] + da1xxhi[1]*x0_a1[1, diag_index[0], diag_index[1]]
        da1x *= self.C

        suma_g1 = 1.*suma1_chain2[0]
        suma_g1x =  dg1x_a * x0_g1[0] +  dg1x_r * x0_g1[1]
        suma_g1x *= self.C
        g1s, dg1s = dg1sigma_dx(xhi00, xm, self.ms, da1, da1x, suma_g1, suma_g1x, self.eps, dii)

        suma_a2x = da2ijx[:, diag_index[0], diag_index[1]]

        dxa = da2x_2a[:, diag_index[0], diag_index[1]]
        dxar = da2x_ar[:, diag_index[0], diag_index[1]]
        dxr = da2x_2r[:, diag_index[0], diag_index[1]]
        suma_g2x = dxa * x0_g2[0]
        suma_g2x += dxar *  x0_g2[1]
        suma_g2x += dxr * x0_g2[2]
        suma_g2x *= self.C**2


        #EStos terminos estan buenos
        suma_a2xxhi = da2xxhi[0] * x0_a2[0, diag_index[0], diag_index[1]]
        suma_a2xxhi += da2xxhi[1] * x0_a2[1, diag_index[0], diag_index[1]]
        suma_a2xxhi += da2xxhi[2] * x0_a2[2, diag_index[0], diag_index[1]]
        suma_a2xxhi *= self.C**2

        da2new, da2newx = da2new_dx_dxhi00(khs, dkhs, dkhsx, dkhsxxhi,
                             suma2_chain[:2], suma_a2x, suma_a2xxhi, self.eps)

        suma_g2 = suma2_chain2[0]
        g2m, dg2m = dg2mca_dx(xhi00, khs, dkhsx, xm, self.ms, da2new, da2newx,
                                            suma_g2, suma_g2x, self.eps, dii)
        g2s = g2m * (1 + gc)
        dg2s = dgc* g2m + (1+gc)*dg2m

        lng, dlngx = dlngmie_dx(gHS, g1s, g2s, dgHS, dg1s, dg2s, beta, self.eps)

        achain = - lng@(x * (self.ms - 1))
        dachainx = - ((self.ms - 1) * lng + dlngx@(x *(self.ms - 1)))

        ares = amono + achain
        daresx = damonox + dachainx

        if self.assoc_bool:
            xj = x[self.compindex]
            iab, diab = dIab_dx(xhi, dxhi_dx, dii, dij,
                                  self.rcij, self.rdij, self.sigmaij3)
            Fab = np.exp(beta * self.eABij) - 1.
            Dab = self.sigmaij3 * Fab * iab
            dDab_dx = self.sigmaij3 * Fab * diab
            Dabij = np.zeros([self.nsites, self.nsites])
            dDabij_dx = np.zeros([self.nc, self.nsites, self.nsites])


            Dabij[self.indexabij] = Dab[self.indexab]

            dDabij_dx[:, self.indexabij[0], self.indexabij[1]] = dDab_dx[:, self.indexab[0], self.indexab[1]]

            KIJ = rho * np.outer(xj, xj) * (self.DIJ * Dabij)
            Xass = Xass_solver(self.nsites, xj, KIJ, self.diagasso, Xass0 = None)
            CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
            dXassx =  dXass_dx(rho, xj, Xass, self.DIJ, Dabij,
                               dDabij_dx, self.dxjdx,CIJ)

            aasso = np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
            daassox = (self.dxjdx * (np.log(Xass) - Xass/2 + 1/2) + dXassx * xj * (1/Xass - 1/2))@self.S
            ares += aasso
            daresx += daassox

        if self.polar_bool:
            eta = xhi[-1]
            deta_dx = dxhi_dx[-1]
            a, dax = dApolar_dx(rho, x, T, self.anij, self.bnij, self.cnij,
                                eta, deta_dx, self.eps, self.epsij, self.sigma3,
                                self.sigmaij3, self.sigmaijk3,  self.npol, self.mupolad2)
            ares += a
            daresx += dax
            '''
        print('amono:', amono)
        print('achain:', achain)
        print('assoc:', aasso)
        print('apol', a)

        print('damono_dx:', damonox)
        print('dachain_dx:', dachainx)
        print('dassoc_dx:', daassox)
        print('dapol_dx', dax)
        '''
        return ares, daresx

    def dares_dxrho(self, x, rho, T):

        dxhi00_drho = self.dxhi00_drho
        diag_index = self.diag_index

        beta = 1 / (kb * T)
        dii = self.diameter(beta)
        #Eq A46
        dij = np.add.outer(dii, dii) / 2
        x0 = self.sigmaij / dij
        x0i = x0[diag_index]

        xmi = x * self.ms
        xm = np.sum(xmi)
        #Eq A8
        xs = xmi / xm
        dxs_dx = - np.multiply.outer(self.ms * x, self.ms) / xm**2
        dxs_dx[diag_index] += self.ms / xm

        #definiendo xhi0 sin dependencia de x
        di03 = np.power.outer(dii, np.arange(4))
        xhi00 = dxhi00_drho * rho

        #Eq A7
        xhi, dxhi_dxhi00, dxhi_dx =  dxhi_dx_eval(xhi00, xs, xmi, xm, self.ms, di03)

        #xhi x Eq A13
        dij3 = dij**3
        xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = dxhix_dx_eval(xhi00, xs, dxs_dx, xm,
                                                                      self.ms, dij3)

        #xhi m Eq A23
        xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = dxhix_dx_eval(xhi00, xs, dxs_dx, xm,
                                                                        self.ms, self.sigmaij3)

        #Terminos necesarios monomero
        a1vdw_cteij = -12 * self.epsij * dij3

        a1vdw_laij = a1vdw_cteij / (self.laij - 3)
        a1vdw_lrij = a1vdw_cteij / (self.lrij - 3)
        a1vdw_2laij = a1vdw_cteij / (2*self.laij - 3)
        a1vdw_2lrij = a1vdw_cteij / (2*self.lrij - 3)
        a1vdw_larij = a1vdw_cteij / (self.larij - 3)
        a1vdwij = (a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij)

        #Terminos necesarios cadena
        a1vdw_cte = a1vdw_cteij[diag_index]
        a1vdw_la = a1vdw_laij[diag_index]
        a1vdw_lr = a1vdw_lrij[diag_index]
        a1vdw_2la = a1vdw_2laij[diag_index]
        a1vdw_2lr = a1vdw_2lrij[diag_index]
        a1vdw_lar = a1vdw_larij[diag_index]
        a1vdw = (a1vdw_la, a1vdw_lr, a1vdw_2la, a1vdw_2lr, a1vdw_lar)

        x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr, self.lar,
                                                 self.laij, self.lrij, self.larij, diag_index)


        da1, da2  = d2a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                    self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00)

        da1x, da2x  = da1sB_dx_eval(xhi00, xhix, x0, xm, self.ms, self.lambdasij,
                                  self.cctesij, a1vdwij, a1vdw_cteij, dxhix_dx)

        da1xxhi, da2xxhi  = da1sB_dx_dxhi00_eval(xhi00, xhix, x0i, xm, self.ms, self.lambdas,
                                               self.cctes, a1vdw, a1vdw_cte, dxhix_dxhi00, dxhix_dx,dxhix_dx_dxhi00)

        suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis = 1)
        suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis = 1)

        suma1_monomerx = self.Cij * (da1x[0] * x0_a1[0] + da1x[1] * x0_a1[1])
        suma2_monomerx = self.Cij**2 * (da2x[0] * x0_a2[0] + da2x[1] * x0_a2[1]
                                      + da2x[2] * x0_a2[2])

        da1ijx = suma1_monomerx
        da2ijx = suma2_monomerx

        da1x_a = da1x[0]
        da1x_r = da1x[1]

        da2x_2a = da2x[0]
        da2x_2r = da2x[2]
        da2x_ar = da2x[1]


        khs, dkhs, d2khs, dkhsx, dkhsxxhi = dkHS_dx_dxhi002(xhix, dxhix_dxhi00,
                                                  dxhix_dx, dxhix_dx_dxhi00)

        aHS, daHS = dahs_dxxhi(xhi, dxhi_dxhi00, dxhi_dx)
        a1m, da1m = da1_dxxhi(xs, dxs_dx, suma1_monomer[:2], suma1_monomerx)
        a2m, da2m = da2_dxxhi(xs, dxs_dx, khs, dkhs, dkhsx, xhixm, dxhixm_dxhi00, dxhixm_dx, suma2_monomer[:2],
                         suma2_monomerx, self.epsij, self.f1, self.f2, self.f3)
        a3m, da3m = da3_dxxhi(xs, dxs_dx, xhixm, dxhixm_dxhi00, dxhixm_dx, self.epsij, self.f4, self.f5, self.f6)

        am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
        dam = daHS + beta * da1m + beta**2 * da2m + beta**3 * da3m
        amono = xm * am
        damonox = self.ms * am[0] + xm * dam


        #USARLOS PARA LOS A1 Y A2 DE LA CADENA
        suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
        suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

        da1c = da1[:, :, diag_index[0], diag_index[1]]
        da2c = da2[:, :, diag_index[0], diag_index[1]]

        #USARLOS EN LA SUMATORIA A1SB DE LA CADENA
        suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis = 1)
        suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis = 1)

        #cadena
        gHS, dgHS = dgdHS_dxxhi(x0i, xhix, dxhix_dxhi00, dxhix_dx) #error e-12
        tetha = np.exp(beta * self.eps) - 1.
        gc, dgc = dgammac_dxxhi(xhixm, dxhixm_dxhi00, dxhixm_dx, self.alpha, tetha)#e-12


        #Esto lo modifique
        dg1x_a = da1x_a[:, diag_index[0], diag_index[1]]
        #print(dg1x_a)
        dg1x_r = da1x_r[:, diag_index[0], diag_index[1]]

        da1 = 1.*suma1_chain[1:3]
        #dxa, dxr = da1xxhi
        da1x = da1xxhi[0]*x0_a1[0, diag_index[0], diag_index[1]] + da1xxhi[1]*x0_a1[1, diag_index[0], diag_index[1]]
        da1x *= self.C

        suma_g1 = 1.*suma1_chain2[:2]
        suma_g1x =  dg1x_a * x0_g1[0] +  dg1x_r * x0_g1[1]
        suma_g1x *= self.C
        g1s, dg1s = dg1sigma_dxxhi(xhi00, xm, self.ms, da1, da1x, suma_g1, suma_g1x, self.eps, dii)



        suma_a2x = da2ijx[:, diag_index[0], diag_index[1]]

        dxa = da2x_2a[:, diag_index[0], diag_index[1]]
        dxar = da2x_ar[:, diag_index[0], diag_index[1]]
        dxr = da2x_2r[:, diag_index[0], diag_index[1]]
        suma_g2x = dxa * x0_g2[0]
        suma_g2x += dxar *  x0_g2[1]
        suma_g2x += dxr * x0_g2[2]
        suma_g2x *= self.C**2


        #dxa, dxar, dxr = da2xxhi
        #EStos terminos estan buenos
        suma_a2xxhi = da2xxhi[0] * x0_a2[0, diag_index[0], diag_index[1]]
        suma_a2xxhi += da2xxhi[1] * x0_a2[1, diag_index[0], diag_index[1]]
        suma_a2xxhi += da2xxhi[2] * x0_a2[2, diag_index[0], diag_index[1]]
        suma_a2xxhi *= self.C**2

        da2new, d2anew, da2newx = da2new_dxxhi_dxhi00(khs, dkhs, d2khs, dkhsx, dkhsxxhi,
                                         suma2_chain[:3], suma_a2x, suma_a2xxhi, self.eps)

        suma_g2 = suma2_chain2[[0,1]]
        g2m, dg2m = dg2mca_dxxhi(xhi00, khs, dkhs, dkhsx, xm, self.ms, da2new, d2anew, da2newx,
                            suma_g2, suma_g2x, self.eps, dii)

        g2s = g2m * (1 + gc[0])
        g2s[1] += g2m[0] * gc[1]
        dg2s = dgc* g2m[0] + (1+gc[0])*dg2m
        lng, dlngx = dlngmie_dxxhi(gHS, g1s, g2s, dgHS, dg1s, dg2s, beta, self.eps)
        achain = - lng@(x * (self.ms - 1))
        dachainx = - ((self.ms - 1) * lng[0] + dlngx@(x *(self.ms - 1)))

        ares = amono + achain
        daresx = damonox + dachainx
        ares *= np.array([1, dxhi00_drho])

        if self.assoc_bool:
            xj = x[self.compindex]
            iab, diab, diabx = dIab_dxrho(xhi, dxhi_dxhi00, dxhi00_drho, dxhi_dx, dii, dij,
                                  self.rcij, self.rdij, self.sigmaij3)

            Fab = np.exp(beta * self.eABij) - 1.
            Dab = self.sigmaij3 * Fab * iab
            dDab_drho = self.sigmaij3 * Fab * diab
            dDab_dx = self.sigmaij3 * Fab * diabx

            Dabij = np.zeros([self.nsites, self.nsites])
            dDabij_drho = np.zeros([self.nsites, self.nsites])
            dDabij_dx = np.zeros([self.nc, self.nsites, self.nsites])


            Dabij[self.indexabij] = Dab[self.indexab]
            dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
            dDabij_dx[:, self.indexabij[0], self.indexabij[1]] = dDab_dx[:, self.indexab[0], self.indexab[1]]

            KIJ = rho * np.outer(xj, xj) * (self.DIJ * Dabij)
            Xass = Xass_solver(self.nsites, xj, KIJ, self.diagasso, Xass0 = None)

            CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
            dXassx =  dXass_dx(rho, xj, Xass, self.DIJ, Dabij,
                               dDabij_dx, self.dxjdx,CIJ)

            dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)

            aasso = np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
            daasso = np.dot(self.S*xj, (1/Xass - 1/2) * dXass)

            ares[0] += aasso
            ares[1] += daasso

            daassox = (self.dxjdx * (np.log(Xass) - Xass/2 + 1/2) + dXassx * xj * (1/Xass - 1/2))@self.S
            daresx += daassox

        if self.polar_bool:
            eta = xhi[-1]
            deta_dxhi00 = dxhi_dxhi00[-1]
            deta = deta_dxhi00 * self.dxhi00_drho
            deta_dx = dxhi_dx[-1]
            a, dax = dApolar_dxrho(rho, x, T, self.anij, self.bnij, self.cnij,
                                eta, deta, deta_dx, self.eps, self.epsij, self.sigma3,
                                self.sigmaij3, self.sigmaijk3,  self.npol, self.mupolad2)
            ares += a
            daresx += dax
            '''
        print('amono:', amono)
        print('achain:', achain)
        print('assoc:', aasso)
        print('apol', a)

        print('damono_dx:', damonox)
        print('dachain_dx:', dachainx)
        print('dassoc_dx:', daassox)
        print('dapol_dx', dax)
        '''
        return ares, daresx

    def afcn(self, x, rho, T):
        a = self.ares(x, rho, T)
        beta = 1 / (kb*T)
        a += aideal(x, rho, beta)
        a *= (Na/beta)
        return a

    def dafcn_drho(self, x, rho, T):
        a = self.dares_drho(x, rho, T)
        beta = 1 / (kb*T)
        a += daideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a

    def d2afcn_drho(self, x, rho, T):
        a = self.d2ares_drho(x, rho, T)
        beta = 1 / (kb*T)
        a += d2aideal_drho(x, rho, beta)
        a *= (Na/beta)
        return a

    def dafcn_dx(self, x, rho, T):
        ares, aresx = self.dares_dx(x, rho, T)
        beta = 1 / (kb*T)
        aideal, aidealx = daideal_dx(x, rho, beta)
        a = (ares + aideal)
        a *= (Na/beta)
        ax = (aresx + aidealx)
        ax *= (Na/beta)
        return a, ax

    def density(self, x, T, P, state, rho0=None):
        if rho0 is None:
            rho = density_topliss(state, x, T, P, self)
        else:
            rho = density_newton(rho0, x, T, P, self)
        return rho

    def pressure(self, x, rho, T):
        rhomolecular = Na * rho
        afcn, dafcn = self.dafcn_drho(x, rhomolecular, T)
        Psaft = rhomolecular**2 * dafcn / Na
        return Psaft

    def dP_drho(self, x, rho, T):
        rhomolecular = Na * rho
        afcn, dafcn, d2afcn = self.d2afcn_drho(x, rhomolecular, T)
        Psaft = rhomolecular**2 * dafcn / Na
        dPsaft = 2 * rhomolecular * dafcn + rhomolecular**2 * d2afcn
        # dPsaft /= Na
        return Psaft, dPsaft

    def logfugmix(self, x, T, P, state, v0=None):
        if v0 is None:
            rho = self.density(x, T, P, state, None)
        else:
            rho0 = 1./v0
            rho = self.density(x, T, P, state, rho0)
        v = 1./rho
        rhomolecular = Na * rho
        ares = self.ares(x, rhomolecular, T)
        Z = P * v / (R * T)
        lnphi = ares + (Z - 1.) - np.log(Z)
        return lnphi, v

    def logfugef(self, x, T, P, state, v0=None):
        if v0 is None:
            rho = self.density(x, T, P, state, None)
        else:
            rho0 = 1./v0
            rho = self.density(x, T, P, state, rho0)
        v = 1./rho
        rhomolecular = Na * rho
        ares, daresx = self.dares_dx(x, rhomolecular, T)
        Z = P * v / (R * T)
        mures = ares + (Z - 1.) + daresx - np.dot(x, daresx)
        lnphi = mures - np.log(Z)
        return lnphi, v

    def a0ad(self, rhoi, T):
        rho = np.sum(rhoi)
        x = rhoi/rho
        rhomolecular = Na * rho

        a = self.ares(x, rhomolecular, T)
        beta = 1 / (kb*T)
        a += aideal(x, rhomolecular, beta)
        a0 = a*rho
        '''
        a = self.afcn(x, rhomolecular, T)
        a0 = a * rho

        beta = 1 / (kb*T)
        RT = (Na/beta)
        a0 /= RT
        '''
        return a0

    def muad(self, rhoi, T):
        rho = np.sum(rhoi)
        x = rhoi/rho
        rhomolecular = Na * rho
        ares, aresx = self.dares_dxrho(x, rhomolecular, T)
        beta = 1 / (kb*T)
        aideal, aidealx = daideal_dxrho(x, rhomolecular, beta)
        afcn, dafcn = (ares + aideal)
        ax = (aresx + aidealx)
        Z = dafcn * rhomolecular
        mu = afcn + ax - np.dot(x, ax) + (Z)
        # mu *= (Na/beta)
        return mu

    def dmuad(self, rhoi, T):
        nc = self.nc
        h = 1e-3
        diff = h * np.eye(nc)

        mu = self.muad(rhoi, T)

        arr = []
        for i in range(nc):
            muad1 = self.muad(rhoi + diff[i], T)
            muad_1 = self.muad(rhoi - diff[i], T)
            arr.append((muad1 - muad_1)/(2*h))
        '''
        mu = self.muad(rhoi, T)
        arr = []
        for i in range(nc):
            muad1 = self.muad(rhoi + diff[i], T)
            arr.append((muad1 -mu)/h)

        arr = []
        for i in range(nc):
            muad1 = self.muad(rhoi + diff[i], T)
            muad2 = self.muad(rhoi + 2* diff[i], T)
            muad_1 = self.muad(rhoi - diff[i], T)
            muad_2 = self.muad(rhoi - 2 *diff[i], T)
            arr.append((muad_2/12 -2*muad_1/3 + 2*muad1/3 - muad2/12)/h)
        '''
        dmu = np.column_stack(arr)
        return mu, dmu

    def dOm(self, rhoi, T, mu, Psat):
        dom = self.a0ad(rhoi, T) - np.sum(np.nan_to_num(rhoi*mu)) + Psat
        return dom

    def sgt_adim(self, T):
        '''
        Tfactor = 1.
        Pfactor = 1.
        rofactor = 1.
        tenfactor = np.sqrt(self.cii[0]) * 1000 #To give tension in mN/m
        zfactor = 10**-10 / np.sqrt(self.cii[0])
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
        self.beta = beta

    def ci(self, T):
        return self.cij * (1 - self.beta)
