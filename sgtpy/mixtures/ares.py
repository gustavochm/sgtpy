import numpy as np
from .monomer_aux import dkHS_dxhi00, d2kHS_dxhi00, d3kHS_dxhi00
from .monomer_aux import dkHS_dx_dxhi00, dkHS_dx_dxhi002

from .a1sB_monomer import x0lambda_eval
from .a1sB_monomer import da1sB_dxhi00_eval, d2a1sB_dxhi00_eval
from .a1sB_monomer import d3a1sB_dxhi00_eval
from .a1sB_monomer import da1sB_dx_eval, da1sB_dx_dxhi00_eval

from .a1_monomer import a1, da1_dxhi00, d2a1_dxhi00,  da1_dx, da1_dxxhi
from .a2_monomer import a2, da2_dxhi00, d2a2_dxhi00,  da2_dx, da2_dxxhi
from .ahs_monomer import ahs, dahs_dxhi00, d2ahs_dxhi00,  dahs_dx, dahs_dxxhi
from .a3_monomer import a3, da3_dxhi00, d2a3_dxhi00,  da3_dx, da3_dxxhi

from .ghs_chain import gdHS, dgdHS_dxhi00, d2gdHS_dxhi00, dgdHS_dx, dgdHS_dxxhi
from .g2mca_chain import g2mca, dg2mca_dxhi00, d2g2mca_dxhi00, dg2mca_dx
from .g2mca_chain import dg2mca_dxxhi
from .gammac_chain import gammac, dgammac_dxhi00, d2gammac_dxhi00, dgammac_dx
from .gammac_chain import dgammac_dxxhi
from .g1sigma_chain import g1sigma, dg1sigma_dxhi00, d2g1sigma_dxhi00
from .g1sigma_chain import dg1sigma_dx, dg1sigma_dxxhi
from .a2new_chain import da2new_dxhi00, d2a2new_dxhi00, d3a2new_dxhi00
from .a2new_chain import da2new_dx_dxhi00, da2new_dxxhi_dxhi00
from .lngmie_chain import lngmie, dlngmie_dxhi00, d2lngmie_dxhi00
from .lngmie_chain import dlngmie_dx, dlngmie_dxxhi

from .association_aux import Iab, dIab_drho, d2Iab_drho
from .association_aux import Xass_solver, dXass_drho, d2Xass_drho, dXass_dx
from .association_aux import dIab_dx, dIab_dxrho, CIJ_matrix

from .polarGV import Apolar, dApolar_drho, d2Apolar_drho
from .polarGV import dApolar_dx, dApolar_dxrho


def xhi_eval(xhi00, xs, xmi, xm, di03):
    xhi = xhi00 * xm * np.matmul(xs, di03)
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
    xhi = xhi00 * xm * np.matmul(xs, di03)
    dxhi_dxhi00 = np.matmul(xmi, di03)
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


def ares(self, x, rho, temp_aux, Xass0=None):

    beta, dii, dij, x0, x0i, di03, dij3 = temp_aux[0:7]
    a1vdw_cteij, a1vdwij, tetha = temp_aux[7:10]

    dxhi00_drho = self.dxhi00_drho
    diag_index = self.diag_index

    xmi = x * self.ms
    xm = np.sum(xmi)
    # Eq A8
    xs = xmi / xm

    # Defining xhi0 without x dependence
    xhi00 = dxhi00_drho * rho

    # Eq A7
    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs, xmi, xm, di03)
    # xhi x Eq A13
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)
    # xhi m Eq A23
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

    x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr,
                                               self.lar, self.laij,
                                               self.lrij, self.larij,
                                               diag_index)

    da1, da2 = da1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                 self.cctesij, a1vdwij, a1vdw_cteij,
                                 dxhix_dxhi00)

    suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis=1)
    suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis=1)

    khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)

    # Monomer evaluation
    a1ij = suma1_monomer[0]
    a2ij = suma2_monomer[0]
    aHS = ahs(xhi)
    a1m = a1(xs, a1ij)
    a2m = a2(xs, khs, xhixm, a2ij, self.epsij, self.f1, self.f2, self.f3)
    a3m = a3(xs, xhixm, self.epsij, self.f4, self.f5, self.f6)
    am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
    amono = xm * am

    # To be used in a1 and a2 for chain
    suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
    suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

    da1c = da1[:, :, diag_index[0], diag_index[1]]
    da2c = da2[:, :, diag_index[0], diag_index[1]]

    # to be used in a1sb in chain summatory
    suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis=1)
    suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis=1)

    # Chain evaluation
    gHS = gdHS(x0i, xhix)
    gc = gammac(xhixm, self.alpha, tetha)

    da1ii = suma1_chain[1]
    a1sB = suma1_chain2[0]
    g1s = g1sigma(xhi00, xm, da1ii, a1sB, self.eps, dii)
    da2new = da2new_dxhi00(khs, dkhs, suma2_chain[:2], self.eps)
    suma_a2 = suma2_chain2[0]
    g2m = g2mca(xhi00, khs, xm, da2new, suma_a2, self.eps, dii)
    g2s = (1 + gc) * g2m

    lng = lngmie(gHS, g1s, g2s, beta, self.eps)
    achain = - np.dot(x * (self.ms - 1), lng)

    ares = amono + achain

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        xj = x[self.compindex]
        iab = Iab(xhi, dii, dij, self.rcij, self.rdij, self.sigmaij3)
        Fab = temp_aux[12]
        Dab = self.sigmaij3 * Fab * iab
        Dabij = np.zeros([self.nsites, self.nsites])
        Dabij[self.indexabij] = Dab[self.indexab]

        Xass = Xass_solver(self.nsites, xj, rho, self.DIJ, Dabij,
                           self.diagasso, Xass)
        ares += np.dot(self.S * xj, (np.log(Xass) - Xass/2 + 1/2))
    else:
        Xass = None

    if self.polar_bool:
        eta = xhi[-1]
        epsa, epsija = temp_aux[13:]
        apolar = Apolar(rho, x, self.anij, self.bnij, self.cnij, eta,
                        epsa, epsija, self.sigma3, self.sigmaij3,
                        self.sigmaijk3,  self.npol, self.mupolad2)
        ares += apolar

    return ares, Xass


def dares_drho(self, x, rho, temp_aux, Xass0=None):

    beta, dii, dij, x0, x0i, di03, dij3 = temp_aux[0:7]
    a1vdw_cteij, a1vdwij, tetha = temp_aux[7:10]

    dxhi00_drho = self.dxhi00_drho
    diag_index = self.diag_index

    xmi = x * self.ms
    xm = np.sum(xmi)
    # Eq A8
    xs = xmi / xm

    # Defining xhi0 wihtout x depedence
    xhi00 = dxhi00_drho * rho

    # Eq A7
    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs, xmi, xm, di03)
    # xhi x Eq A13
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)
    # xhi m Eq A23
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

    x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr,
                                               self.lar, self.laij,
                                               self.lrij, self.larij,
                                               diag_index)

    da1, da2 = d2a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                  self.cctesij, a1vdwij, a1vdw_cteij,
                                  dxhix_dxhi00)

    suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis=1)
    suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis=1)

    khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)

    # Monomer evaluation
    a1ij = suma1_monomer[:2]
    a2ij = suma2_monomer[:2]
    aHS = dahs_dxhi00(xhi, dxhi_dxhi00)
    a1m = da1_dxhi00(xs, a1ij)
    a2m = da2_dxhi00(xs, khs, dkhs, xhixm, dxhixm_dxhi00, a2ij,
                     self.epsij, self.f1, self.f2, self.f3)
    a3m = da3_dxhi00(xs, xhixm, dxhixm_dxhi00, self.epsij, self.f4,
                     self.f5, self.f6)
    am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
    amono = xm * am

    # To be used in a1 and a2 of chain
    suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
    suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

    da1c = da1[:, :, diag_index[0], diag_index[1]]
    da2c = da2[:, :, diag_index[0], diag_index[1]]

    # to be used in a1sb of chain
    suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis=1)
    suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis=1)

    # chain evaluation
    gHS = dgdHS_dxhi00(x0i, xhix, dxhix_dxhi00)
    gc = dgammac_dxhi00(xhixm, dxhixm_dxhi00, self.alpha, tetha)

    da1ii = suma1_chain[1:3]
    a1sB = suma1_chain2[:2]
    g1s = dg1sigma_dxhi00(xhi00, xm, da1ii, a1sB, self.eps, dii)

    da2new, d2a2new = d2a2new_dxhi00(khs, dkhs, d2khs, suma2_chain[:3],
                                     self.eps)
    suma_a2 = suma2_chain2[[0, 1]]
    g2m = dg2mca_dxhi00(xhi00, khs, dkhs, xm, da2new, d2a2new,
                        suma_a2, self.eps, dii)
    g2s = g2m * (1 + gc[0])
    g2s[1] += g2m[0] * gc[1]
    lng = dlngmie_dxhi00(gHS, g1s, g2s, beta, self.eps)
    achain = - lng@(x * (self.ms - 1))

    ares = amono + achain
    ares *= np.array([1, dxhi00_drho])

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        xj = x[self.compindex]
        iab, diab = dIab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii, dij,
                              self.rcij, self.rdij, self.sigmaij3)
        Fab = temp_aux[12]
        Fab = np.exp(beta * self.eABij) - 1.
        Dab = self.sigmaij3 * Fab * iab
        dDab_drho = self.sigmaij3 * Fab * diab
        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])
        Dabij[self.indexabij] = Dab[self.indexab]
        dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
        Xass = Xass_solver(self.nsites, xj, rho, self.DIJ, Dabij,
                           self.diagasso, Xass)
        CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
        dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
        ares[0] += np.dot(self.S * xj, (np.log(Xass) - Xass/2 + 1/2))
        ares[1] += np.dot(self.S*xj, (1/Xass - 1/2) * dXass)
    else:
        Xass = None

    if self.polar_bool:
        eta = xhi[-1]
        deta_dxhi00 = dxhi_dxhi00[-1]
        deta = deta_dxhi00 * self.dxhi00_drho
        epsa, epsija = temp_aux[13:]
        dapolar = dApolar_drho(rho, x, self.anij, self.bnij, self.cnij,
                               eta, deta, epsa, epsija,
                               self.sigma3, self.sigmaij3, self.sigmaijk3,
                               self.npol, self.mupolad2)
        ares += dapolar

    return ares, Xass


def d2ares_drho(self, x, rho, temp_aux, Xass0=None):

    beta, dii, dij, x0, x0i, di03, dij3 = temp_aux[0:7]
    a1vdw_cteij, a1vdwij, tetha = temp_aux[7:10]

    dxhi00_drho = self.dxhi00_drho
    diag_index = self.diag_index

    xmi = x * self.ms
    xm = np.sum(xmi)
    # Equation A8
    xs = xmi / xm

    # Defining xhi0 without x depedence
    xhi00 = dxhi00_drho * rho

    # Equation A7
    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs, xmi, xm, di03)
    # xhi x Eq A13
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs, xm, dij3)
    # xhi m Eq A23
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs, xm, self.sigmaij3)

    x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr,
                                               self.lar, self.laij,
                                               self.lrij, self.larij,
                                               diag_index)

    da1, da2 = d3a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                  self.cctesij, a1vdwij, a1vdw_cteij,
                                  dxhix_dxhi00)

    suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis=1)
    suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis=1)

    khs, dkhs, d2khs, d3khs = d3kHS_dxhi00(xhix, dxhix_dxhi00)

    # Monomer evaluation
    suma_a1 = suma1_monomer[:3]
    suma_a2 = suma2_monomer[:3]

    aHS = d2ahs_dxhi00(xhi, dxhi_dxhi00)
    a1m = d2a1_dxhi00(xs,  suma_a1)
    a2m = d2a2_dxhi00(xs, khs, dkhs, d2khs, xhixm, dxhixm_dxhi00, suma_a2,
                      self.epsij, self.f1, self.f2, self.f3)
    a3m = d2a3_dxhi00(xs, xhixm, dxhixm_dxhi00, self.epsij, self.f4,
                      self.f5, self.f6)
    am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
    amono = xm * am

    # To be used in a1 and a2 of chain
    suma1_chain = suma1_monomer[:, diag_index[0], diag_index[1]]
    suma2_chain = suma2_monomer[:, diag_index[0], diag_index[1]]

    da1c = da1[:, :, diag_index[0], diag_index[1]]
    da2c = da2[:, :, diag_index[0], diag_index[1]]

    # To be used in the sumatory a1sb of chain
    sumg1_chain = self.C * np.sum(da1c * x0_g1, axis=1)
    sumg2_chain = self.C**2 * np.sum(da2c * x0_g2, axis=1)

    # Chain contribution
    gHS = d2gdHS_dxhi00(x0i, xhix, dxhix_dxhi00)
    gc = d2gammac_dxhi00(xhixm, dxhixm_dxhi00, self.alpha, tetha)

    da1ii = suma1_chain[1:4]
    suma_g1 = sumg1_chain[:3]
    g1s = d2g1sigma_dxhi00(xhi00, xm, da1ii, suma_g1, self.eps, dii)

    da2new, d2a2new, d3a2new = d3a2new_dxhi00(khs, dkhs, d2khs, d3khs,
                                              suma2_chain, self.eps)
    suma_g2 = sumg2_chain[[0, 1, 2]]
    g2m = d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xm, da2new, d2a2new,
                         d3a2new, suma_g2, self.eps, dii)
    g2s = g2m * (1 + gc[0])
    g2s[1] += g2m[0] * gc[1]
    g2s[2] += 2. * g2m[1] * gc[1] + g2m[0] * gc[2]

    lng = d2lngmie_dxhi00(gHS, g1s, g2s, beta, self.eps)
    achain = - lng@(x * (self.ms - 1))

    ares = amono + achain

    ares *= np.array([1., dxhi00_drho, dxhi00_drho**2])

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        xj = x[self.compindex]
        iab, diab, d2iab = d2Iab_drho(xhi, dxhi_dxhi00, dxhi00_drho, dii,
                                      dij, self.rcij, self.rdij,
                                      self.sigmaij3)
        Fab = temp_aux[12]
        Dab = self.sigmaij3 * Fab * iab
        dDab_drho = self.sigmaij3 * Fab * diab
        d2Dab_drho = self.sigmaij3 * Fab * d2iab

        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])
        d2Dabij_drho = np.zeros([self.nsites, self.nsites])

        Dabij[self.indexabij] = Dab[self.indexab]
        dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
        d2Dabij_drho[self.indexabij] = d2Dab_drho[self.indexab]

        Xass = Xass_solver(self.nsites, xj, rho, self.DIJ, Dabij,
                           self.diagasso, Xass)
        CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
        dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho,
                           CIJ)
        d2Xass = d2Xass_drho(rho, xj, Xass, dXass, self.DIJ, Dabij,
                             dDabij_drho, d2Dabij_drho, CIJ)

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        # ares[0] += np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
        # ares[1] += np.dot(self.S*xj, (1/Xass - 1/2) * dXass)
        # ares[2] += np.dot(self.S*xj, -(dXass/Xass)**2+d2Xass*(1/Xass-1/2))

        ares[0] += np.dot(self.S*xj, aux1)
        ares[1] += np.dot(self.S*xj, aux2 * dXass)
        ares[2] += np.dot(self.S*xj, -(dXass/Xass)**2+d2Xass*aux2)
    else:
        Xass = None

    if self.polar_bool:
        eta = xhi[-1]
        deta_dxhi00 = dxhi_dxhi00[-1]
        deta = deta_dxhi00 * self.dxhi00_drho
        epsa, epsija = temp_aux[13:]
        dapolar = d2Apolar_drho(rho, x, self.anij, self.bnij, self.cnij,
                                eta, deta, epsa, epsija,
                                self.sigma3, self.sigmaij3, self.sigmaijk3,
                                self.npol, self.mupolad2)
        ares += dapolar

    return ares, Xass


def dares_dx(self, x, rho, temp_aux, Xass0=None):

    beta, dii, dij, x0, x0i, di03, dij3 = temp_aux[0:7]
    a1vdw_cteij, a1vdwij, tetha, a1vdw_cte, a1vdw = temp_aux[7:12]

    dxhi00_drho = self.dxhi00_drho
    diag_index = self.diag_index

    xmi = x * self.ms
    xm = np.sum(xmi)
    # Equation A8
    xs = xmi / xm
    dxs_dx = - np.multiply.outer(self.ms * x, self.ms) / xm**2
    dxs_dx[diag_index] += self.ms / xm

    # Defining xhi0 wihtout x dependence
    xhi00 = dxhi00_drho * rho

    # Equation A7
    out = dxhi_dx_eval(xhi00, xs, xmi, xm, self.ms, di03)
    xhi, dxhi_dxhi00, dxhi_dx = out
    # xhi x Eq A13
    out = dxhix_dx_eval(xhi00, xs, dxs_dx, xm, self.ms, dij3)
    xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = out
    # xhi m Eq A23
    out = dxhix_dx_eval(xhi00, xs, dxs_dx, xm, self.ms, self.sigmaij3)
    xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = out

    x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr,
                                               self.lar, self.laij,
                                               self.lrij, self.larij,
                                               diag_index)

    da1, da2 = da1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                 self.cctesij, a1vdwij, a1vdw_cteij,
                                 dxhix_dxhi00)

    da1x, da2x = da1sB_dx_eval(xhi00, xhix, x0, xm, self.ms,
                               self.lambdasij, self.cctesij, a1vdwij,
                               a1vdw_cteij, dxhix_dx)

    da1xxhi, da2xxhi = da1sB_dx_dxhi00_eval(xhi00, xhix, x0i, xm,
                                            self.ms, self.lambdas,
                                            self.cctes, a1vdw, a1vdw_cte,
                                            dxhix_dxhi00, dxhix_dx,
                                            dxhix_dx_dxhi00)

    suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis=1)
    suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis=1)

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
    a2m, da2m = da2_dx(xs, dxs_dx, khs, dkhsx, xhixm, dxhixm_dx,
                       suma2_monomer[0], suma2_monomerx, self.epsij,
                       self.f1, self.f2, self.f3)
    a3m, da3m = da3_dx(xs, dxs_dx, xhixm, dxhixm_dx, self.epsij,
                       self.f4, self.f5, self.f6)

    am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
    dam = daHS + beta * da1m + beta**2 * da2m + beta**3 * da3m
    amono = xm * am
    damonox = self.ms * am + xm * dam

    # To be used in a1 and a2 of chain
    suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
    suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

    da1c = da1[:, :, diag_index[0], diag_index[1]]
    da2c = da2[:, :, diag_index[0], diag_index[1]]

    # To be used in a1sb of chain
    suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis=1)
    suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis=1)

    # Chain contribution
    gHS, dgHS = dgdHS_dx(x0i, xhix, dxhix_dx)
    gc, dgc = dgammac_dx(xhixm, dxhixm_dx, self.alpha, tetha)

    dg1x_a = da1x_a[:, diag_index[0], diag_index[1]]
    dg1x_r = da1x_r[:, diag_index[0], diag_index[1]]

    da1 = 1.*suma1_chain[1]
    da1x = da1xxhi[0]*x0_a1[0, diag_index[0], diag_index[1]]
    da1x += da1xxhi[1]*x0_a1[1, diag_index[0], diag_index[1]]
    da1x *= self.C

    suma_g1 = 1.*suma1_chain2[0]
    suma_g1x = dg1x_a * x0_g1[0] + dg1x_r * x0_g1[1]
    suma_g1x *= self.C
    g1s, dg1s = dg1sigma_dx(xhi00, xm, self.ms, da1, da1x, suma_g1,
                            suma_g1x, self.eps, dii)

    suma_a2x = da2ijx[:, diag_index[0], diag_index[1]]
    dxa = da2x_2a[:, diag_index[0], diag_index[1]]
    dxar = da2x_ar[:, diag_index[0], diag_index[1]]
    dxr = da2x_2r[:, diag_index[0], diag_index[1]]
    suma_g2x = dxa * x0_g2[0]
    suma_g2x += dxar * x0_g2[1]
    suma_g2x += dxr * x0_g2[2]
    suma_g2x *= self.C**2

    suma_a2xxhi = da2xxhi[0] * x0_a2[0, diag_index[0], diag_index[1]]
    suma_a2xxhi += da2xxhi[1] * x0_a2[1, diag_index[0], diag_index[1]]
    suma_a2xxhi += da2xxhi[2] * x0_a2[2, diag_index[0], diag_index[1]]
    suma_a2xxhi *= self.C**2

    da2new, da2newx = da2new_dx_dxhi00(khs, dkhs, dkhsx, dkhsxxhi,
                                       suma2_chain[:2], suma_a2x,
                                       suma_a2xxhi, self.eps)

    suma_g2 = suma2_chain2[0]
    g2m, dg2m = dg2mca_dx(xhi00, khs, dkhsx, xm, self.ms, da2new, da2newx,
                          suma_g2, suma_g2x, self.eps, dii)
    g2s = g2m * (1 + gc)
    dg2s = dgc * g2m + (1+gc)*dg2m

    lng, dlngx = dlngmie_dx(gHS, g1s, g2s, dgHS, dg1s, dg2s, beta,
                            self.eps)
    achain = - lng@(x * (self.ms - 1))
    dachainx = - ((self.ms - 1) * lng + dlngx@(x*(self.ms - 1)))

    ares = amono + achain
    daresx = damonox + dachainx

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        xj = x[self.compindex]
        iab, diab = dIab_dx(xhi, dxhi_dx, dii, dij, self.rcij, self.rdij,
                            self.sigmaij3)
        Fab = temp_aux[12]
        Dab = self.sigmaij3 * Fab * iab
        dDab_dx = self.sigmaij3 * Fab * diab
        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_dx = np.zeros([self.nc, self.nsites, self.nsites])

        Dabij[self.indexabij] = Dab[self.indexab]

        dDabij_dx[:, self.indexabij[0], self.indexabij[1]] = dDab_dx[:, self.indexab[0], self.indexab[1]]

        Xass = Xass_solver(self.nsites, xj, rho, self.DIJ, Dabij,
                           self.diagasso, Xass)
        CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
        dXassx = dXass_dx(rho, xj, Xass, self.DIJ, Dabij, dDabij_dx,
                          self.dxjdx, CIJ)

        # aasso = np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
        # daassox = (self.dxjdx * (np.log(Xass) - Xass/2 + 1/2) + dXassx * xj * (1/Xass - 1/2))@self.S

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        aasso = np.dot(self.S*xj, aux1)
        daassox = (self.dxjdx * aux1 + dXassx * xj * aux2)@self.S
        ares += aasso
        daresx += daassox
    else:
        Xass = None
    if self.polar_bool:
        eta = xhi[-1]
        deta_dx = dxhi_dx[-1]
        epsa, epsija = temp_aux[13:]
        a, dax = dApolar_dx(rho, x, self.anij, self.bnij, self.cnij,
                            eta, deta_dx, epsa, epsija, self.sigma3,
                            self.sigmaij3, self.sigmaijk3,  self.npol,
                            self.mupolad2)
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
    return ares, daresx, Xass


def dares_dxrho(self, x, rho, temp_aux, Xass0=None):
    beta, dii, dij, x0, x0i, di03, dij3 = temp_aux[0:7]
    a1vdw_cteij, a1vdwij, tetha, a1vdw_cte, a1vdw = temp_aux[7:12]

    dxhi00_drho = self.dxhi00_drho
    diag_index = self.diag_index

    xmi = x * self.ms
    xm = np.sum(xmi)
    # Equation A8
    xs = xmi / xm
    dxs_dx = - np.multiply.outer(self.ms * x, self.ms) / xm**2
    dxs_dx[diag_index] += self.ms / xm

    # defining xhi0 without x dependence
    xhi00 = dxhi00_drho * rho

    # Equation A7
    xhi, dxhi_dxhi00, dxhi_dx = dxhi_dx_eval(xhi00, xs, xmi, xm,
                                             self.ms, di03)

    # xhi x Eq A13
    out = dxhix_dx_eval(xhi00, xs, dxs_dx, xm, self.ms, dij3)
    xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = out

    # xhi m Eq A23
    out = dxhix_dx_eval(xhi00, xs, dxs_dx, xm, self.ms, self.sigmaij3)
    xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = out

    x0_a1, x0_a2, x0_g1, x0_g2 = x0lambda_eval(x0, self.la, self.lr,
                                               self.lar, self.laij,
                                               self.lrij, self.larij,
                                               diag_index)

    da1, da2 = d2a1sB_dxhi00_eval(xhi00, xhix, x0, xm, self.lambdasij,
                                  self.cctesij, a1vdwij, a1vdw_cteij,
                                  dxhix_dxhi00)

    da1x, da2x = da1sB_dx_eval(xhi00, xhix, x0, xm, self.ms,
                               self.lambdasij, self.cctesij, a1vdwij,
                               a1vdw_cteij, dxhix_dx)

    da1xxhi, da2xxhi = da1sB_dx_dxhi00_eval(xhi00, xhix, x0i, xm, self.ms,
                                            self.lambdas, self.cctes,
                                            a1vdw, a1vdw_cte, dxhix_dxhi00,
                                            dxhix_dx, dxhix_dx_dxhi00)

    suma1_monomer = self.Cij * np.sum(da1 * x0_a1, axis=1)
    suma2_monomer = self.Cij**2 * np.sum(da2 * x0_a2, axis=1)

    suma1_monomerx = self.Cij * (da1x[0] * x0_a1[0] + da1x[1] * x0_a1[1])
    suma2_monomerx = (da2x[0]*x0_a2[0]+da2x[1]*x0_a2[1]+da2x[2]*x0_a2[2])
    suma2_monomerx *= self.Cij**2

    da1ijx = suma1_monomerx
    da2ijx = suma2_monomerx

    da1x_a = da1x[0]
    da1x_r = da1x[1]

    da2x_2a = da2x[0]
    da2x_2r = da2x[2]
    da2x_ar = da2x[1]

    khs, dkhs, d2khs, dkhsx, dkhsxxhi = dkHS_dx_dxhi002(xhix, dxhix_dxhi00,
                                                        dxhix_dx,
                                                        dxhix_dx_dxhi00)

    aHS, daHS = dahs_dxxhi(xhi, dxhi_dxhi00, dxhi_dx)
    a1m, da1m = da1_dxxhi(xs, dxs_dx, suma1_monomer[:2], suma1_monomerx)
    a2m, da2m = da2_dxxhi(xs, dxs_dx, khs, dkhs, dkhsx, xhixm,
                          dxhixm_dxhi00, dxhixm_dx, suma2_monomer[:2],
                          suma2_monomerx, self.epsij,
                          self.f1, self.f2, self.f3)
    a3m, da3m = da3_dxxhi(xs, dxs_dx, xhixm, dxhixm_dxhi00, dxhixm_dx,
                          self.epsij, self.f4, self.f5, self.f6)

    am = aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m
    dam = daHS + beta * da1m + beta**2 * da2m + beta**3 * da3m
    amono = xm * am
    damonox = self.ms * am[0] + xm * dam

    # To be used in a1 and a2 of chain
    suma1_chain = 1.*suma1_monomer[:, diag_index[0], diag_index[1]]
    suma2_chain = 1.*suma2_monomer[:, diag_index[0], diag_index[1]]

    da1c = da1[:, :, diag_index[0], diag_index[1]]
    da2c = da2[:, :, diag_index[0], diag_index[1]]

    # To be used in a1sb of chain
    suma1_chain2 = self.C * np.sum(da1c * x0_g1, axis=1)
    suma2_chain2 = self.C**2 * np.sum(da2c * x0_g2, axis=1)

    # Chain contribution
    gHS, dgHS = dgdHS_dxxhi(x0i, xhix, dxhix_dxhi00, dxhix_dx)
    gc, dgc = dgammac_dxxhi(xhixm, dxhixm_dxhi00, dxhixm_dx, self.alpha,
                            tetha)

    dg1x_a = da1x_a[:, diag_index[0], diag_index[1]]
    dg1x_r = da1x_r[:, diag_index[0], diag_index[1]]

    da1 = 1.*suma1_chain[1:3]
    da1x = da1xxhi[0]*x0_a1[0, diag_index[0], diag_index[1]]
    da1x += da1xxhi[1]*x0_a1[1, diag_index[0], diag_index[1]]
    da1x *= self.C

    suma_g1 = 1.*suma1_chain2[:2]
    suma_g1x = dg1x_a * x0_g1[0] + dg1x_r * x0_g1[1]
    suma_g1x *= self.C
    g1s, dg1s = dg1sigma_dxxhi(xhi00, xm, self.ms, da1, da1x, suma_g1,
                               suma_g1x, self.eps, dii)

    suma_a2x = da2ijx[:, diag_index[0], diag_index[1]]

    dxa = da2x_2a[:, diag_index[0], diag_index[1]]
    dxar = da2x_ar[:, diag_index[0], diag_index[1]]
    dxr = da2x_2r[:, diag_index[0], diag_index[1]]
    suma_g2x = dxa * x0_g2[0]
    suma_g2x += dxar * x0_g2[1]
    suma_g2x += dxr * x0_g2[2]
    suma_g2x *= self.C**2

    suma_a2xxhi = da2xxhi[0] * x0_a2[0, diag_index[0], diag_index[1]]
    suma_a2xxhi += da2xxhi[1] * x0_a2[1, diag_index[0], diag_index[1]]
    suma_a2xxhi += da2xxhi[2] * x0_a2[2, diag_index[0], diag_index[1]]
    suma_a2xxhi *= self.C**2

    da2new, d2anew, da2newx = da2new_dxxhi_dxhi00(khs, dkhs, d2khs, dkhsx,
                                                  dkhsxxhi,
                                                  suma2_chain[:3],
                                                  suma_a2x, suma_a2xxhi,
                                                  self.eps)

    suma_g2 = suma2_chain2[[0, 1]]
    g2m, dg2m = dg2mca_dxxhi(xhi00, khs, dkhs, dkhsx, xm, self.ms, da2new,
                             d2anew, da2newx, suma_g2, suma_g2x,
                             self.eps, dii)

    g2s = g2m * (1 + gc[0])
    g2s[1] += g2m[0] * gc[1]
    dg2s = dgc*g2m[0] + (1+gc[0])*dg2m
    lng, dlngx = dlngmie_dxxhi(gHS, g1s, g2s, dgHS, dg1s, dg2s, beta,
                               self.eps)
    achain = - lng@(x * (self.ms - 1))
    dachainx = - ((self.ms - 1) * lng[0] + dlngx@(x*(self.ms - 1)))

    ares = amono + achain
    daresx = damonox + dachainx
    ares *= np.array([1, dxhi00_drho])

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        xj = x[self.compindex]
        iab, diab, diabx = dIab_dxrho(xhi, dxhi_dxhi00, dxhi00_drho,
                                      dxhi_dx, dii, dij, self.rcij,
                                      self.rdij, self.sigmaij3)
        Fab = temp_aux[12]
        Dab = self.sigmaij3 * Fab * iab
        dDab_drho = self.sigmaij3 * Fab * diab
        dDab_dx = self.sigmaij3 * Fab * diabx

        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])
        dDabij_dx = np.zeros([self.nc, self.nsites, self.nsites])

        Dabij[self.indexabij] = Dab[self.indexab]
        dDabij_drho[self.indexabij] = dDab_drho[self.indexab]
        dDabij_dx[:, self.indexabij[0], self.indexabij[1]] = dDab_dx[:, self.indexab[0], self.indexab[1]]

        Xass = Xass_solver(self.nsites, xj, rho, self.DIJ, Dabij,
                           self.diagasso, Xass)
        CIJ = CIJ_matrix(rho, xj, Xass, self.DIJ, Dabij, self.diagasso)
        dXassx = dXass_dx(rho, xj, Xass, self.DIJ, Dabij, dDabij_dx,
                          self.dxjdx, CIJ)
        dXass = dXass_drho(rho, xj, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        aasso = np.dot(self.S*xj, aux1)
        daasso = np.dot(self.S*xj, aux2 * dXass)
        # aasso = np.dot(self.S*xj, (np.log(Xass) - Xass/2 + 1/2))
        # daasso = np.dot(self.S*xj, (1/Xass - 1/2) * dXass)

        ares[0] += aasso
        ares[1] += daasso

        # daassox = (self.dxjdx * (np.log(Xass) - Xass/2 + 1/2) + dXassx * xj * (1/Xass - 1/2))@self.S
        daassox = (self.dxjdx * aux1 + dXassx * xj * aux2)@self.S
        daresx += daassox
    else:
        Xass = None

    if self.polar_bool:
        eta = xhi[-1]
        deta_dxhi00 = dxhi_dxhi00[-1]
        deta = deta_dxhi00 * self.dxhi00_drho
        deta_dx = dxhi_dx[-1]
        epsa, epsija = temp_aux[13:]
        a, dax = dApolar_dxrho(rho, x, self.anij, self.bnij, self.cnij,
                               eta, deta, deta_dx, epsa, epsija,
                               self.sigma3, self.sigmaij3, self.sigmaijk3,
                               self.npol, self.mupolad2)
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
    return ares, daresx, Xass
