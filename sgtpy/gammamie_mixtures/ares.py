from __future__ import division, print_function, absolute_import
import numpy as np
from .a1sB_monomer import a1sB_eval, da1sB_dxhi00_eval, d2a1sB_dxhi00_eval
from .a1sB_monomer import d3a1sB_dxhi00_eval
from .a1sB_monomer import da1sB_dx_eval, da1sB_dx_dxhi00_eval
from .a1sB_monomer import da1sB_dx_dxhi00_dxxhi_eval
from .a1sB_monomer import da1sB_dx_d2xhi00_dxxhi_eval

from .ahs_monomer import ahs, dahs_dxhi00, d2ahs_dxhi00
from .ahs_monomer import dahs_dx, dahs_dxxhi
from .a3_monomer import a3, da3_dxhi00, d2a3_dxhi00
from .a3_monomer import da3_dx, da3_dxxhi
from .a2_monomer import a2, da2_dxhi00, d2a2_dxhi00
from .a2_monomer import da2_dx, da2_dxxhi
from .a1_monomer import a1, da1_dxhi00, d2a1_dxhi00
from .a1_monomer import da1_dx, da1_dxxhi

from .a2new_chain import da2new_dxhi00, d2a2new_dxhi00, d3a2new_dxhi00
from .a2new_chain import da2new_dx_dxhi00, da2new_dxxhi_dxhi00
from .gdHS_chain import gdHS, dgdHS_dxhi00, d2gdHS_dxhi00
from .gdHS_chain import dgdHS_dx, dgdHS_dxxhi
from .gammac_chain import gammac, dgammac_dxhi00, d2gammac_dxhi00
from .gammac_chain import dgammac_dx, dgammac_dxxhi
from .g1sigma_chain import g1sigma, dg1sigma_dxhi00, d2g1sigma_dxhi00
from .g1sigma_chain import dg1sigma_dx, dg1sigma_dxxhi
from .g2mca_chain import g2mca, dg2mca_dxhi00, d2g2mca_dxhi00
from .g2mca_chain import dg2mca_dx, dg2mca_dxxhi
from .lngmie_chain import lngmie, dlngmie_dxhi00, d2lngmie_dxhi00
from .lngmie_chain import dlngmie_dx, dlngmie_dxxhi

from .monomer_aux import dkHS_dxhi00, d2kHS_dxhi00, d3kHS_dxhi00
from .monomer_aux import dkHS_dx_dxhi00, d2kHS_dx_dxhi00

from .association_aux import Xass_solver, CIJ_matrix
from .association_aux import dXass_drho, d2Xass_drho, dXass_dx
from .association_aux import Iab, dIab_drho, d2Iab_drho, dIab


# Eq. (14) Paper 2014
def xhi_eval(xhi00, xs_k, xs_m, d_kk03):
    dxhi_dxhi00 = xs_m * np.matmul(xs_k, d_kk03)
    xhi = xhi00 * dxhi_dxhi00
    return xhi, dxhi_dxhi00


def dxhi_dx_eval(xhi00, xs_k, xs_m, d_kk03, dxk_dx_aux):
    dxhi_dxhi00 = xs_m * np.matmul(xs_k, d_kk03)
    xhi = xhi00 * dxhi_dxhi00
    dxhi_dx = (xhi00*dxk_dx_aux@d_kk03).T
    return xhi, dxhi_dxhi00, dxhi_dx


# Eq (22) Paper 2014
def xhix_eval(xhi00, xs_k, xs_m, d_kl3):
    sum1 = np.matmul(np.matmul(xs_k, d_kl3), xs_k)
    dxhix_dxhi00 = xs_m * sum1
    xhix = xhi00 * dxhix_dxhi00
    return xhix, dxhix_dxhi00


def dxhix_dx_eval(xhi00, xs_k, dxsk_dx, xs_m, zs_m, d_kl3):
    aux1 = xs_k * d_kl3
    aux2 = np.dot(xs_k, aux1)
    aux3 = aux2.sum()
    dxhix_dxhi00 = xs_m * aux3
    xhix = xhi00 * dxhix_dxhi00
    suma1 = 2*np.sum(dxsk_dx@aux1, axis=1)
    dxhix_dx_dxhi00 = (zs_m * aux3 + xs_m * suma1)
    dxhix_dx = xhi00 * dxhix_dx_dxhi00
    return xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00


def ares(self, x, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3

    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    x_k = x[self.groups_index]

    xs_ki = x_k*Sk*vki*vk
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs_k, xs_m, d_kk03)
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, d_kl3)
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, sigma_kl3)

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3*xhix2, 4*xhix3]])

    # monomer contribution calculation
    Ckl = self.Ckl
    Ckl2 = self.Ckl2
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

    a1kl, a2kl = a1sB_eval(xhi00, xhix, xhix_vec[0], xs_m, I_lambdaskl,
                           J_lambdaskl, ccteskl, a1vdwkl, a1vdw_ctekl)

    # zero order pertubation
    aHS = ahs(xhi)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(a1kl * x0_a1, axis=0)
    a1m = a1(xs_k, suma1_monomer)

    # second order pertubation
    khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    suma2_monomer = Ckl2 * np.sum(a2kl * x0_a2, axis=0)
    a2m = a2(xs_k, khs, xhixm, suma2_monomer, eps_kl, f1, f2, f3)

    # third order pertubaton
    a3m = a3(xs_k, xhixm, eps_kl, f4, f5, f6)

    am = xs_m * (aHS + beta * a1m + beta**2 * a2m + beta**3 * a3m)

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    da1ii, da2ii = da1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xs_m, I_lambdasii,
                                     J_lambdasii, cctesii, a1vdwii,
                                     a1vdw_cteii, dxhix_dxhi00)

    # g hard sphere
    gHS = gdHS(x0i_matrix, xhix)

    # gamma_c
    gc = gammac(xhixm, alphaii, tetha)

    # g1sigma
    da1_chain = Cii * np.sum(da1ii[1] * x0_a1ii, axis=0)
    suma_g1 = Cii * np.sum(da1ii[0] * x0_g1ii, axis=0)
    g1s = g1sigma(xhi00, xs_m, da1_chain, suma_g1, a1vdw_cteii)

    # g2sigma
    dsuma2_chain = Cii2 * np.sum(da2ii * x0_a2ii, axis=1)
    da2new = da2new_dxhi00(khs, dkhs, dsuma2_chain, eps_ii)
    suma_g2 = Cii2 * np.sum(da2ii[0] * x0_g2ii, axis=0)
    g2m = g2mca(xhi00, khs, xs_m, da2new, suma_g2, eps_ii, a1vdw_cteii)
    g2s = (1 + gc) * g2m

    lng = lngmie(gHS, g1s, g2s, beps_ii, beps_ii2)

    ach = - lng@(x * (self.zs_m - 1.))

    ares = am + ach

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        # T_ad = 1/(self.eps_ij*beta)
        T_ad = temp_aux[29]
        sigma_kl3 = self.sigma_kl3
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        rho_ad = rho * xs_m * sigma_x3

        Iijklab = np.zeros([self.nc, self.nc])
        Iab(rho_ad, T_ad, Iijklab)

        diagasso = self.diagasso
        # vki_asso = self.vki[self.group_asso_index]
        vki_asso = self.vki_asso
        DIJ = self.DIJ
        xj_asso = x[self.molecule_id_index_sites]
        xjvk = xj_asso*vki_asso

        # Fklab = np.exp(self.epsAB_kl * beta) - 1
        Fklab = temp_aux[30]
        Dijklab = self.kAB_kl * Fklab
        Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

        Xass = Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass)

        ares += np.dot(self.S * xjvk, (np.log(Xass) - Xass/2 + 1/2))
    else:
        Xass = Xass0
    return ares, Xass


def dares_drho(self, x, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3

    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    x_k = x[self.groups_index]

    xs_ki = x_k*Sk*vki*vk
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs_k, xs_m, d_kk03)
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, d_kl3)
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, sigma_kl3)

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3*xhix2, 4*xhix3],
                         [0., 2., 6*xhix, 12*xhix2]])

    # monomer contribution calculation
    Ckl = self.Ckl
    Ckl2 = self.Ckl2
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl
    da1kl, da2kl = da1sB_dxhi00_eval(xhi00, xhix, xhix_vec[:2], xs_m,
                                     I_lambdaskl, J_lambdaskl, ccteskl,
                                     a1vdwkl, a1vdw_ctekl, dxhix_dxhi00)

    # zero order pertubation
    daHS = dahs_dxhi00(xhi, dxhi_dxhi00)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(da1kl * x0_a1, axis=1)
    da1m = da1_dxhi00(xs_k, suma1_monomer)

    # second order pertubation
    khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    suma2_monomer = Ckl2 * np.sum(da2kl * x0_a2, axis=1)
    da2m = da2_dxhi00(xs_k, khs, dkhs, xhixm, dxhixm_dxhi00, suma2_monomer,
                      eps_kl, f1, f2, f3)

    # third order pertubaton
    da3m = da3_dxhi00(xs_k, xhixm, dxhixm_dxhi00, eps_kl, f4, f5, f6)

    damono = xs_m * (daHS + beta * da1m + beta**2 * da2m + beta**3 * da3m)

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    d2a1ii, d2a2ii = d2a1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xs_m,
                                        I_lambdasii, J_lambdasii, cctesii,
                                        a1vdwii, a1vdw_cteii, dxhix_dxhi00)

    # g hard sphere
    dgHS = dgdHS_dxhi00(x0i_matrix, xhix, dxhix_dxhi00)

    # gamma_c
    dgc = dgammac_dxhi00(xhixm, dxhixm_dxhi00, alphaii, tetha)

    # g1sigma
    d2a1_chain = Cii * np.sum(d2a1ii[1:] * x0_a1ii, axis=1)
    dsuma_g1 = Cii * np.sum(d2a1ii[:2] * x0_g1ii, axis=1)
    dg1s = dg1sigma_dxhi00(xhi00, xs_m, d2a1_chain, dsuma_g1, a1vdw_cteii)

    # g2sigma
    d2suma2_chain = Cii2 * np.sum(d2a2ii * x0_a2ii, axis=1)
    d2a2new = d2a2new_dxhi00(khs, dkhs, d2khs, d2suma2_chain, eps_ii)
    dsuma_g2 = Cii2 * np.sum(d2a2ii[:2] * x0_g2ii, axis=1)
    dg2m = dg2mca_dxhi00(xhi00, khs, dkhs, xs_m, d2a2new, dsuma_g2, eps_ii,
                         a1vdw_cteii)
    dg2s = dg2m * (1 + dgc[0])
    dg2s[1] += dg2m[0] * dgc[1]

    dlng = dlngmie_dxhi00(dgHS, dg1s, dg2s, beps_ii, beps_ii2)

    dachain = - dlng@(x * (self.zs_m - 1.))
    ares = damono + dachain
    ares *= self.dxhi00_1

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        # T_ad = 1/(self.eps_ij*beta)
        T_ad = temp_aux[29]
        sigma_kl3 = self.sigma_kl3
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad

        Iijklab = np.zeros([self.nc, self.nc])
        dIijklab_drho = np.zeros([self.nc, self.nc])
        dIab_drho(rho_ad, T_ad, drho_ad, Iijklab, dIijklab_drho)

        diagasso = self.diagasso
        # vki_asso = self.vki[self.group_asso_index]
        vki_asso = self.vki_asso
        DIJ = self.DIJ
        xj_asso = x[self.molecule_id_index_sites]
        xjvk = xj_asso*vki_asso

        # Fklab = np.exp(self.epsAB_kl * beta) - 1
        Fklab = temp_aux[30]
        Dijklab = self.kAB_kl * Fklab
        Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

        dDijklab_drho = self.kAB_kl * Fklab
        dDijklab_drho[self.indexABij] *= dIijklab_drho[self.indexAB_id]

        Xass = Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, xjvk, Xass, DIJ, Dijklab, diagasso)
        dXass = dXass_drho(rho, xjvk, Xass, DIJ, Dijklab, dDijklab_drho, CIJ)

        ares[0] += np.dot(self.S * xjvk, (np.log(Xass) - Xass/2 + 1/2))
        ares[1] += np.dot(self.S * xjvk, (1/Xass - 1/2) * dXass)
    else:
        Xass = Xass0
    return ares, Xass


def d2ares_drho(self, x, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3

    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    x_k = x[self.groups_index]

    xs_ki = x_k*Sk*vki*vk
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs_k, xs_m, d_kk03)
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, d_kl3)
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, sigma_kl3)

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3.*xhix2, 4.*xhix3],
                         [0., 2., 6*xhix, 12.*xhix2],
                         [0., 0., 6., 24.*xhix]])

    # monomer contribution calculation
    Ckl = self.Ckl
    Ckl2 = self.Ckl2
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs_k, xs_m, d_kk03)
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, d_kl3)
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, sigma_kl3)

    d2a1kl, d2a2kl = d2a1sB_dxhi00_eval(xhi00, xhix, xhix_vec[:3], xs_m,
                                        I_lambdaskl, J_lambdaskl, ccteskl,
                                        a1vdwkl, a1vdw_ctekl, dxhix_dxhi00)

    # zero order pertubation
    d2aHS = d2ahs_dxhi00(xhi, dxhi_dxhi00)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(d2a1kl * x0_a1, axis=1)
    d2a1m = d2a1_dxhi00(xs_k, suma1_monomer)

    # second order pertubation
    khs, dkhs, d2khs, d3khs = d3kHS_dxhi00(xhix, dxhix_dxhi00)
    suma2_monomer = Ckl2 * np.sum(d2a2kl * x0_a2, axis=1)
    d2a2m = d2a2_dxhi00(xs_k, khs, dkhs, d2khs, xhixm, dxhixm_dxhi00,
                        suma2_monomer, eps_kl, f1, f2, f3)

    # third order pertubaton
    d2a3m = d2a3_dxhi00(xs_k, xhixm, dxhixm_dxhi00, eps_kl, f4, f5, f6)

    d2amono = xs_m * (d2aHS + beta * d2a1m + beta**2 * d2a2m + beta**3 * d2a3m)

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    d3a1ii, d3a2ii = d3a1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xs_m,
                                        I_lambdasii, J_lambdasii, cctesii,
                                        a1vdwii, a1vdw_cteii, dxhix_dxhi00)

    # g hard sphere
    d2gHS = d2gdHS_dxhi00(x0i_matrix, xhix, dxhix_dxhi00)

    # gamma_c
    d2gc = d2gammac_dxhi00(xhixm, dxhixm_dxhi00, alphaii, tetha)

    # g1sigma
    d3a1_chain = Cii * np.sum(d3a1ii[1:] * x0_a1ii, axis=1)
    d2suma_g1 = Cii * np.sum(d3a1ii[:3] * x0_g1ii, axis=1)
    d2g1s = d2g1sigma_dxhi00(xhi00, xs_m, d3a1_chain, d2suma_g1, a1vdw_cteii)

    # g2sigma
    d3suma2_chain = Cii2 * np.sum(d3a2ii * x0_a2ii, axis=1)
    d3a2new = d3a2new_dxhi00(khs, dkhs, d2khs, d3khs, d3suma2_chain, eps_ii)
    d2suma_g2 = Cii2 * np.sum(d3a2ii[:3] * x0_g2ii, axis=1)
    d2g2m = d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xs_m, d3a2new, d2suma_g2,
                           eps_ii, a1vdw_cteii)
    d2g2s = d2g2m * (1. + d2gc[0])
    d2g2s[1] += d2g2m[0] * d2gc[1]
    d2g2s[2] += 2. * d2g2m[1] * d2gc[1] + d2g2m[0] * d2gc[2]

    d2lng = d2lngmie_dxhi00(d2gHS, d2g1s, d2g2s, beps_ii, beps_ii2)

    d2achain = - d2lng@(x * (self.zs_m - 1.))
    ares = d2amono + d2achain
    ares *= self.dxhi00_2

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        # T_ad = 1/(self.eps_ij*beta)
        T_ad = temp_aux[29]
        sigma_kl3 = self.sigma_kl3
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad

        Iijklab = np.zeros([self.nc, self.nc])
        dIijklab_drho = np.zeros([self.nc, self.nc])
        d2Iijklab_drho = np.zeros([self.nc, self.nc])
        d2Iab_drho(rho_ad, T_ad, drho_ad, Iijklab, dIijklab_drho,
                   d2Iijklab_drho)

        diagasso = self.diagasso
        # vki_asso = self.vki[self.group_asso_index]
        vki_asso = self.vki_asso
        DIJ = self.DIJ
        xj_asso = x[self.molecule_id_index_sites]
        xjvk = xj_asso*vki_asso

        # Fklab = np.exp(self.epsAB_kl * beta) - 1
        Fklab = temp_aux[30]
        Dijklab = self.kAB_kl * Fklab
        Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

        dDijklab_drho = self.kAB_kl * Fklab
        dDijklab_drho[self.indexABij] *= dIijklab_drho[self.indexAB_id]

        d2Dijklab_drho = self.kAB_kl * Fklab
        d2Dijklab_drho[self.indexABij] *= d2Iijklab_drho[self.indexAB_id]

        Xass = Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, xjvk, Xass, DIJ, Dijklab, diagasso)
        dXass = dXass_drho(rho, xjvk, Xass, DIJ, Dijklab, dDijklab_drho, CIJ)
        d2Xass = d2Xass_drho(rho, xjvk, Xass, dXass, DIJ, Dijklab,
                             dDijklab_drho, d2Dijklab_drho, CIJ)

        aux0 = self.S * xjvk
        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        ares[0] += np.dot(aux0, aux1)
        ares[1] += np.dot(aux0, aux2 * dXass)
        ares[2] += np.dot(aux0, -(dXass/Xass)**2+d2Xass*aux2)

    else:
        Xass = Xass0
    return ares, Xass


def dares_dx(self, x, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3

    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    dxkdx = self.dxkdx
    zs_m = self.zs_m
    x_k = x[self.groups_index]

    aux_Skvksvki = Sk*vki*vk
    xs_ki = x_k*aux_Skvksvki
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    dxk_dx_aux = aux_Skvksvki * dxkdx
    dxsk_dx = dxk_dx_aux * xs_m
    dxsk_dx -= np.outer(zs_m, xs_ki)
    dxsk_dx /= xs_m**2

    out = dxhi_dx_eval(xhi00, xs_k, xs_m, d_kk03, dxk_dx_aux)
    xhi, dxhi_dxhi00, dxhi_dx = out
    out = dxhix_dx_eval(xhi00, xs_k, dxsk_dx, xs_m, zs_m, d_kl3)
    xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = out
    out = dxhix_dx_eval(xhi00, xs_k, dxsk_dx, xs_m, zs_m, sigma_kl3)
    xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = out

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3*xhix2, 4*xhix3],
                         [0., 2, 6*xhix, 12*xhix2]])

    khs, dkhs, dkhsx, dkhsxxhi = dkHS_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx,
                                                dxhix_dx_dxhi00)

    # monomer contribution calculation
    Ckl = self.Ckl
    Ckl2 = self.Ckl2
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

    out = da1sB_dx_eval(xhi00, xhix, xhix_vec[:2], xs_m, zs_m, I_lambdaskl,
                        J_lambdaskl, ccteskl, a1vdwkl, a1vdw_ctekl, dxhix_dx)
    a1kl, a2kl, da1x_kl, da2x_kl = out

    # zero order pertubation
    aHS, daHSx = dahs_dx(xhi, dxhi_dx)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(a1kl * x0_a1, axis=0)
    suma1x_monomer = Ckl * (da1x_kl[0]*x0_a1[0] + da1x_kl[1]*x0_a1[1])

    a1m, da1mx = da1_dx(xs_k, dxsk_dx, suma1_monomer, suma1x_monomer)

    # second order pertubation
    suma2_monomer = Ckl2 * np.sum(a2kl * x0_a2, axis=0)
    suma2x_monomer = da2x_kl[0]*x0_a2[0] + da2x_kl[1]*x0_a2[1]
    suma2x_monomer += da2x_kl[2]*x0_a2[2]
    suma2x_monomer *= Ckl2

    a2m, da2mx = da2_dx(xs_k, dxsk_dx, khs, dkhsx, xhixm, dxhixm_dx,
                        suma2_monomer, suma2x_monomer, eps_kl, f1, f2, f3)

    # third order pertubation
    a3m, da3mx = da3_dx(xs_k, dxsk_dx, xhixm, dxhixm_dx, eps_kl, f4, f5, f6)

    beta2 = beta**2
    beta3 = beta2*beta
    am = aHS + beta * a1m + beta2 * a2m + beta3 * a3m
    damx = daHSx + beta * da1mx + beta2 * da2mx + beta3 * da3mx

    amono = xs_m * am
    damonox = self.zs_m * am + xs_m * damx

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    out = da1sB_dx_dxhi00_dxxhi_eval(xhi00, xhix, xhix_vec, xs_m, zs_m,
                                     I_lambdasii, J_lambdasii, cctesii,
                                     a1vdwii, a1vdw_cteii, dxhix_dxhi00,
                                     dxhix_dx, dxhix_dx_dxhi00)
    da1ii, da2ii, da1x_ii, da2x_ii, da1_xxhi00_ii, da2_xxhi00_ii = out

    # g hard sphere
    ghs, dghsx = dgdHS_dx(x0i_matrix, xhix, dxhix_dx)

    # g1sigma
    da1_chain = Cii * np.sum(da1ii[1] * x0_a1ii, axis=0)
    da1x_chain = Cii*(da1_xxhi00_ii[0]*x0_a1ii[0]+da1_xxhi00_ii[1]*x0_a1ii[1])

    suma_g1 = Cii * np.sum(da1ii[0] * x0_g1ii, axis=0)
    suma_g1x = Cii*(da1x_ii[0] * x0_g1ii[0] + da1x_ii[1] * x0_g1ii[1])
    g1s, dg1sx = dg1sigma_dx(xhi00, xs_m, zs_m, da1_chain, da1x_chain, suma_g1,
                             suma_g1x, a1vdw_cteii)

    # gamma_c
    gc, dgcx = dgammac_dx(xhixm, dxhixm_dx, alphaii, tetha)

    # g2sigma
    suma_g2 = Cii2 * np.sum(da2ii[0] * x0_g2ii, axis=0)
    suma_g2x = da2x_ii[0]*x0_g2ii[0] + da2x_ii[1]*x0_g2ii[1]
    suma_g2x += da2x_ii[2]*x0_g2ii[2]
    suma_g2x *= Cii2

    dsuma2_chain = Cii2 * np.sum(da2ii * x0_a2ii, axis=1)

    dsuma2x_chain = da2x_ii[0] * x0_a2ii[0] + da2x_ii[1] * x0_a2ii[1]
    dsuma2x_chain += da2x_ii[2] * x0_a2ii[2]
    dsuma2x_chain *= Cii2

    dsuma2xxhi_chain = da2_xxhi00_ii[0] * x0_a2ii[0]
    dsuma2xxhi_chain += da2_xxhi00_ii[1] * x0_a2ii[1]
    dsuma2xxhi_chain += da2_xxhi00_ii[2] * x0_a2ii[2]
    dsuma2xxhi_chain *= Cii2

    da2new, da2newx = da2new_dx_dxhi00(khs, dkhs, dkhsx, dkhsxxhi,
                                       dsuma2_chain, dsuma2x_chain,
                                       dsuma2xxhi_chain, eps_ii)
    g2m, dg2mx = dg2mca_dx(xhi00, khs, dkhsx, xs_m, zs_m, da2new, da2newx,
                           suma_g2, suma_g2x, eps_ii, a1vdw_cteii)
    g2s = g2m * (1 + gc)
    dg2sx = dgcx*g2m + (1+gc)*dg2mx

    lng, dlngx = dlngmie_dx(ghs, g1s, g2s, dghsx, dg1sx, dg2sx, beps_ii,
                            beps_ii2)

    zs_m1 = (zs_m - 1.)
    xzs_m1 = x*zs_m1
    achain = - lng@xzs_m1
    dachainx = - dlngx@xzs_m1 - lng * zs_m1

    ares = amono + achain
    daresx = damonox + dachainx

    if self.asso_bool:
        nc = self.nc
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0

        # beta = temp_aux[0]
        # T_ad = 1/(self.eps_ij*beta)
        T_ad = temp_aux[29]
        aux1 = xs_k * sigma_kl3
        aux2 = np.dot(xs_k, aux1)
        sigma_x3 = np.sum(aux2)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad
        suma1 = 2*np.sum(dxsk_dx@aux1, axis=1)
        drhoad_dx = rho * (zs_m * sigma_x3 + xs_m * suma1)

        Iijklab = np.zeros([nc, nc])
        dIijklab = np.zeros([nc, nc])
        dIab(rho_ad, T_ad, Iijklab, dIijklab)
        dIijklab_dx = np.multiply.outer(drhoad_dx, dIijklab)

        diagasso = self.diagasso
        vki_asso = self.vki[self.group_asso_index]
        DIJ = self.DIJ
        xj_asso = x[self.molecule_id_index_sites]
        xjvk = xj_asso*vki_asso
        dxjasso_dx = self.dxjasso_dx

        # Fklab = np.exp(self.epsAB_kl * beta) - 1
        Fklab = temp_aux[30]
        Dijklab = self.kAB_kl * Fklab
        Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

        dDijklab_dx = np.stack(nc*[self.kAB_kl * Fklab])
        dDijklab_dx[:, self.indexABij[0], self.indexABij[1]] *= dIijklab_dx[:, self.indexAB_id[0], self.indexAB_id[1]]

        Xass = Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, xjvk, Xass, DIJ, Dijklab, diagasso)
        dXassx = dXass_dx(rho, xjvk, Xass, DIJ, Dijklab, dDijklab_dx,
                          dxjasso_dx, CIJ)

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        ares += np.dot(self.S*xjvk, aux1)
        daresx += (dxjasso_dx * aux1 + dXassx * xjvk * aux2)@self.S
    else:
        Xass = Xass0

    return ares, daresx, Xass


def dares_dx_drho(self, x, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3

    Sk = self.Sk
    vki = self.vki
    vk = self.vk
    dxkdx = self.dxkdx
    zs_m = self.zs_m
    x_k = x[self.groups_index]

    aux_Skvksvki = Sk*vki*vk
    xs_ki = x_k*aux_Skvksvki
    xs_m = np.sum(xs_ki)
    xs_k = xs_ki / xs_m

    dxk_dx_aux = aux_Skvksvki * dxkdx
    dxsk_dx = dxk_dx_aux * xs_m
    dxsk_dx -= np.outer(zs_m, xs_ki)
    dxsk_dx /= xs_m**2

    out = dxhi_dx_eval(xhi00, xs_k, xs_m, d_kk03, dxk_dx_aux)
    xhi, dxhi_dxhi00, dxhi_dx = out
    out = dxhix_dx_eval(xhi00, xs_k, dxsk_dx, xs_m, zs_m, d_kl3)
    xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00 = out
    out = dxhix_dx_eval(xhi00, xs_k, dxsk_dx, xs_m, zs_m, sigma_kl3)
    xhixm, dxhixm_dxhi00, dxhixm_dx, dxhixm_dx_dxhi00 = out

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3*xhix2, 4*xhix3],
                         [0., 2, 6*xhix, 12*xhix2]])

    out = d2kHS_dx_dxhi00(xhix, dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    khs, dkhs, d2khs, dkhsx, dkhsxxhi = out

    # monomer contribution calculation
    Ckl = self.Ckl
    Ckl2 = self.Ckl2
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

    out = da1sB_dx_dxhi00_eval(xhi00, xhix, xhix_vec, xs_m, zs_m,
                               I_lambdaskl, J_lambdaskl, ccteskl, a1vdwkl,
                               a1vdw_ctekl, dxhix_dxhi00, dxhix_dx)
    da1kl, da2kl, da1x_kl, da2x_kl = out

    # zero order pertubation
    aHS, daHSx = dahs_dxxhi(xhi, dxhi_dxhi00, dxhi_dx)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(da1kl * x0_a1, axis=1)
    suma1x_monomer = Ckl * (da1x_kl[0]*x0_a1[0] + da1x_kl[1]*x0_a1[1])

    a1m, da1mx = da1_dxxhi(xs_k, dxsk_dx, suma1_monomer, suma1x_monomer)

    # second order pertubation
    suma2_monomer = Ckl2 * np.sum(da2kl * x0_a2, axis=1)
    suma2x_monomer = da2x_kl[0]*x0_a2[0] + da2x_kl[1]*x0_a2[1]
    suma2x_monomer += da2x_kl[2]*x0_a2[2]
    suma2x_monomer *= Ckl2

    a2m, da2mx = da2_dxxhi(xs_k, dxsk_dx, khs, dkhs, dkhsx, xhixm,
                           dxhixm_dxhi00, dxhixm_dx, suma2_monomer,
                           suma2x_monomer, eps_kl, f1, f2, f3)

    # third order pertubation
    a3m, da3mx = da3_dxxhi(xs_k, dxsk_dx, xhixm, dxhixm_dxhi00, dxhixm_dx,
                           eps_kl, f4, f5, f6)

    beta2 = beta**2
    beta3 = beta2*beta
    am = aHS + beta * a1m + beta2 * a2m + beta3 * a3m
    damx = daHSx + beta * da1mx + beta2 * da2mx + beta3 * da3mx

    amono = xs_m * am
    damonox = self.zs_m * am[0] + xs_m * damx

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    out = da1sB_dx_d2xhi00_dxxhi_eval(xhi00, xhix, xhix_vec, xs_m, zs_m,
                                      I_lambdasii, J_lambdasii, cctesii,
                                      a1vdwii, a1vdw_cteii, dxhix_dxhi00,
                                      dxhix_dx, dxhix_dx_dxhi00)
    d2a1ii, d2a2ii, da1x_ii, da2x_ii, da1_xxhi00_ii, da2_xxhi00_ii = out

    # g hard sphere
    ghs, dghsx = dgdHS_dxxhi(x0i_matrix, xhix, dxhix_dxhi00, dxhix_dx)

    # g1sigma
    d2a1_chain = Cii * np.sum(d2a1ii[1:] * x0_a1ii, axis=1)
    # da1_chain = Cii * np.sum(da1ii[1] * x0_a1ii, axis=0)
    da1x_chain = Cii*(da1_xxhi00_ii[0]*x0_a1ii[0]+da1_xxhi00_ii[1]*x0_a1ii[1])
    dsuma_g1 = Cii * np.sum(d2a1ii[:2] * x0_g1ii, axis=1)
    # suma_g1 = Cii * np.sum(da1ii[0] * x0_g1ii, axis=0)
    suma_g1x = Cii*(da1x_ii[0] * x0_g1ii[0] + da1x_ii[1] * x0_g1ii[1])
    g1s, dg1sx = dg1sigma_dxxhi(xhi00, xs_m, zs_m, d2a1_chain, da1x_chain,
                                dsuma_g1, suma_g1x, a1vdw_cteii)

    # gamma_c
    gc, dgcx = dgammac_dxxhi(xhixm, dxhixm_dxhi00, dxhixm_dx, alphaii, tetha)

    # g2sigma
    dsuma_g2 = Cii2 * np.sum(d2a2ii[:2] * x0_g2ii, axis=1)
    suma_g2x = da2x_ii[0]*x0_g2ii[0] + da2x_ii[1]*x0_g2ii[1]
    suma_g2x += da2x_ii[2]*x0_g2ii[2]
    suma_g2x *= Cii2

    dsuma2x_chain = da2x_ii[0] * x0_a2ii[0] + da2x_ii[1] * x0_a2ii[1]
    dsuma2x_chain += da2x_ii[2] * x0_a2ii[2]
    dsuma2x_chain *= Cii2

    dsuma2xxhi_chain = da2_xxhi00_ii[0] * x0_a2ii[0]
    dsuma2xxhi_chain += da2_xxhi00_ii[1] * x0_a2ii[1]
    dsuma2xxhi_chain += da2_xxhi00_ii[2] * x0_a2ii[2]
    dsuma2xxhi_chain *= Cii2

    d2suma2_chain = Cii2 * np.sum(d2a2ii * x0_a2ii, axis=1)

    *d2a2new, da2newx = da2new_dxxhi_dxhi00(khs, dkhs, d2khs, dkhsx, dkhsxxhi,
                                            d2suma2_chain, dsuma2x_chain,
                                            dsuma2xxhi_chain, eps_ii)
    g2m, dg2mx = dg2mca_dxxhi(xhi00, khs, dkhs, dkhsx, xs_m, zs_m, d2a2new,
                              da2newx, dsuma_g2, suma_g2x, eps_ii, a1vdw_cteii)
    g2s = g2m * (1 + gc[0])
    g2s[1] += g2m[0] * gc[1]
    dg2sx = dgcx*g2m[0] + (1 + gc[0])*dg2mx

    lng, dlngx = dlngmie_dxxhi(ghs, g1s, g2s, dghsx, dg1sx, dg2sx, beps_ii,
                               beps_ii2)

    zs_m1 = (zs_m - 1.)
    xzs_m1 = x*zs_m1
    achain = - lng@xzs_m1
    dachainx = - dlngx@xzs_m1 - lng[0] * zs_m1

    ares = amono + achain
    ares *= self.dxhi00_1
    daresx = damonox + dachainx

    if self.asso_bool:
        nc = self.nc
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0

        # beta = temp_aux[0]
        # T_ad = 1/(self.eps_ij*beta)
        T_ad = temp_aux[29]
        aux1 = xs_k * sigma_kl3
        aux2 = np.dot(xs_k, aux1)
        sigma_x3 = np.sum(aux2)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad
        suma1 = 2*np.sum(dxsk_dx@aux1, axis=1)
        drhoad_dx = rho * (zs_m * sigma_x3 + xs_m * suma1)

        Iijklab = np.zeros([nc, nc])
        dIijklab = np.zeros([nc, nc])
        dIab(rho_ad, T_ad, Iijklab, dIijklab)
        dIijklab_dx = np.multiply.outer(drhoad_dx, dIijklab)
        dIijklab_drho = dIijklab*drho_ad

        diagasso = self.diagasso
        vki_asso = self.vki[self.group_asso_index]
        DIJ = self.DIJ
        xj_asso = x[self.molecule_id_index_sites]
        xjvk = xj_asso*vki_asso
        dxjasso_dx = self.dxjasso_dx

        # Fklab = np.exp(self.epsAB_kl * beta) - 1
        Fklab = temp_aux[30]
        Dijklab = self.kAB_kl * Fklab
        Dijklab[self.indexABij] *= Iijklab[self.indexAB_id]

        dDijklab_drho = self.kAB_kl * Fklab
        dDijklab_drho[self.indexABij] *= dIijklab_drho[self.indexAB_id]

        dDijklab_dx = np.stack(nc*[self.kAB_kl * Fklab])
        dDijklab_dx[:, self.indexABij[0], self.indexABij[1]] *= dIijklab_dx[:, self.indexAB_id[0], self.indexAB_id[1]]

        Xass = Xass_solver(rho, xjvk, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, xjvk, Xass, DIJ, Dijklab, diagasso)
        dXass = dXass_drho(rho, xjvk, Xass, DIJ, Dijklab, dDijklab_drho, CIJ)
        dXassx = dXass_dx(rho, xjvk, Xass, DIJ, Dijklab, dDijklab_dx,
                          dxjasso_dx, CIJ)

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2

        ares[0] += np.dot(self.S*xjvk, aux1)
        ares[1] += np.dot(self.S*xjvk, aux2 * dXass)
        daresx += (dxjasso_dx * aux1 + dXassx * xjvk * aux2)@self.S
    else:
        Xass = Xass0

    return ares, daresx, Xass
