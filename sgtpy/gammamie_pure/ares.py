from __future__ import division, print_function, absolute_import
import numpy as np
from .a1sB_monomer import a1sB_eval, da1sB_dxhi00_eval, d2a1sB_dxhi00_eval
from .a1sB_monomer import d3a1sB_dxhi00_eval
from .ahs_monomer import ahs, dahs_dxhi00, d2ahs_dxhi00
from .a3_monomer import a3, da3_dxhi00, d2a3_dxhi00
from .a2_monomer import a2, da2_dxhi00, d2a2_dxhi00
from .a1_monomer import a1, da1_dxhi00, d2a1_dxhi00
from .monomer_aux import kHS, dkHS_dxhi00, d2kHS_dxhi00, d3kHS_dxhi00
from .a2new_chain import da2new_dxhi00, d2a2new_dxhi00, d3a2new_dxhi00
from .gdHS_chain import gdHS, dgdHS_dxhi00, d2gdHS_dxhi00
from .gammac_chain import gammac, dgammac_dxhi00, d2gammac_dxhi00
from .g1sigma_chain import g1sigma, dg1sigma_dxhi00, d2g1sigma_dxhi00
from .g2mca_chain import g2mca, dg2mca_dxhi00, d2g2mca_dxhi00
from .lngmie_chain import lngmie, dlngmie_dxhi00, d2lngmie_dxhi00
from .association_aux import Xass_solver, CIJ_matrix, dXass_drho, d2Xass_drho
from .association_aux import Iab, dIab_drho, d2Iab_drho


# Eq. (14) Paper 2014
def xhi_eval(xhi00, xs_k, xs_m, d_kk03):
    dxhi_dxhi00 = xs_m * np.matmul(xs_k, d_kk03)
    xhi = xhi00 * dxhi_dxhi00
    return xhi, dxhi_dxhi00


# Eq (22) Paper 2014
def xhix_eval(xhi00, xs_k, xs_m, d_kl3):
    sum1 = np.matmul(np.matmul(xs_k, d_kl3), xs_k)
    dxhix_dxhi00 = xs_m * sum1
    xhix = xhi00 * dxhix_dxhi00
    return xhix, dxhix_dxhi00


def ares(self, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3
    xs_k = self.xs_k
    xs_m = self.xs_m
    Ckl = self.Ckl
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    ccteskl = self.ccteskl

    xhi, dxhi_dxhi00 = xhi_eval(xhi00, xs_k, xs_m, d_kk03)
    xhix, dxhix_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, d_kl3)
    xhixm, dxhixm_dxhi00 = xhix_eval(xhi00, xs_k, xs_m, sigma_kl3)

    xhix2 = xhix**2
    xhix3 = xhix2*xhix
    xhix4 = xhix3*xhix
    xhix_vec = np.array([[xhix, xhix2, xhix3, xhix4],
                         [1., 2 * xhix, 3*xhix2, 4*xhix3]])

    # monomer calculation
    a1kl, a2kl = a1sB_eval(xhi00, xhix, xhix_vec[0], xs_m, I_lambdaskl,
                           J_lambdaskl, ccteskl, a1vdwkl, a1vdw_ctekl)

    # zero order pertubation
    aHS = ahs(xhi)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(a1kl * x0_a1, axis=0)
    a1m = a1(xs_k, suma1_monomer)

    # second order pertubation
    khs = kHS(xhix)
    suma2_monomer = Ckl**2 * np.sum(a2kl * x0_a2, axis=0)
    a2m = a2(xs_k, khs, xhixm, suma2_monomer, eps_kl, f1, f2, f3)

    # third order pertubaton
    a3m = a3(xs_k, xhixm, eps_kl, f4, f5, f6)

    am = xs_m * (aHS + beta * a1m + beta2 * a2m + beta3 * a3m)

    # chain contribution calculation
    # lambdasii = self.lambdasii
    cctesii = self.cctesii
    alphaii = self.alphaii
    eps_ii = self.eps_ii
    Cii = self.Cii
    Cii2 = self.Cii2

    da1ii, da2ii = da1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xs_m,
                                     I_lambdasii, J_lambdasii, cctesii,
                                     a1vdwii, a1vdw_cteii, dxhix_dxhi00)

    # g hard sphere
    gHS = gdHS(x0i_matrix, xhix)

    # gamma_c
    gc = gammac(xhixm, alphaii, tetha)

    # g1sigma
    da1_chain = Cii * np.sum(da1ii[1] * x0_a1ii, axis=0)
    suma_g1 = Cii * np.sum(da1ii[0] * x0_g1ii, axis=0)
    g1s = g1sigma(xhi00, xs_m, da1_chain, suma_g1, a1vdw_cteii)

    # g2sigma
    khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    dsuma2_chain = Cii2 * np.sum(da2ii * x0_a2ii, axis=1)
    da2new = da2new_dxhi00(khs, dkhs, dsuma2_chain, eps_ii)
    suma_g2 = Cii2 * np.sum(da2ii[0] * x0_g2ii, axis=0)
    g2m = g2mca(xhi00, khs, xs_m, da2new, suma_g2, eps_ii, a1vdw_cteii)
    g2s = (1 + gc) * g2m

    lng = lngmie(gHS, g1s, g2s, beps_ii, beps_ii2)
    ach = - (xs_m - 1)*lng

    ares = am + ach

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        T_ad = temp_aux[29]
        Fklab = temp_aux[30]
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        rho_ad = rho * xs_m * sigma_x3

        Iijklab = Iab(rho_ad, T_ad)

        diagasso = self.diagasso
        vki_asso = self.vki[self.group_asso_index]
        DIJ = self.DIJ
        Dijklab = self.kAB_kl * Fklab * Iijklab

        Xass = Xass_solver(rho, vki_asso, DIJ, Dijklab, diagasso, Xass)
        ares += np.dot(self.S * vki_asso, (np.log(Xass) - Xass/2 + 1/2))
    else:
        Xass = None
    return ares, Xass


def dares_drho(self, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3
    xs_k = self.xs_k
    xs_m = self.xs_m
    Ckl = self.Ckl
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

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
    da1kl, da2kl = da1sB_dxhi00_eval(xhi00, xhix, xhix_vec[:2], xs_m,
                                     I_lambdaskl, J_lambdaskl, ccteskl,
                                     a1vdwkl, a1vdw_ctekl, dxhix_dxhi00)

    # zero order pertubation
    daHS = dahs_dxhi00(xhi, dxhi_dxhi00)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(da1kl * x0_a1, axis=1)
    da1m = da1_dxhi00(xs_k, suma1_monomer)

    # second order pertubation
    khs, dkhs = dkHS_dxhi00(xhix, dxhix_dxhi00)
    suma2_monomer = Ckl**2 * np.sum(da2kl * x0_a2, axis=1)
    da2m = da2_dxhi00(xs_k, khs, dkhs, xhixm, dxhixm_dxhi00, suma2_monomer,
                      eps_kl, f1, f2, f3)

    # third order pertubaton
    da3m = da3_dxhi00(xs_k, xhixm, dxhixm_dxhi00, eps_kl, f4, f5, f6)

    damono = xs_m * (daHS + beta * da1m + beta2 * da2m + beta3 * da3m)

    # chain contribution calculation
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
    khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    d2suma2_chain = Cii2 * np.sum(d2a2ii * x0_a2ii, axis=1)
    d2a2new = d2a2new_dxhi00(khs, dkhs, d2khs, d2suma2_chain, eps_ii)
    dsuma_g2 = Cii2 * np.sum(d2a2ii[:2] * x0_g2ii, axis=1)
    dg2m = dg2mca_dxhi00(xhi00, khs, dkhs, xs_m, d2a2new, dsuma_g2, eps_ii,
                         a1vdw_cteii)
    dg2s = dg2m * (1 + dgc[0])
    dg2s[1] += dg2m[0] * dgc[1]

    dlng = dlngmie_dxhi00(dgHS, dg1s, dg2s, beps_ii, beps_ii2)

    dachain = - (xs_m - 1.)*dlng

    ares = damono + dachain
    ares *= self.dxhi00_1

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        T_ad = temp_aux[29]
        Fklab = temp_aux[30]
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad

        Iijklab, dIijklab_drho = dIab_drho(rho_ad, T_ad, drho_ad)

        diagasso = self.diagasso
        vki_asso = self.vki[self.group_asso_index]
        DIJ = self.DIJ
        Dijklab = self.kAB_kl * Fklab * Iijklab
        dDijklab_drho = self.kAB_kl * Fklab * dIijklab_drho

        Xass = Xass_solver(rho, vki_asso, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, vki_asso, Xass, DIJ, Dijklab, diagasso)
        dXass = dXass_drho(rho, vki_asso, Xass, DIJ, Dijklab, dDijklab_drho,
                           CIJ)

        Svki = self.S * vki_asso
        ares[0] += np.dot(Svki, (np.log(Xass) - Xass/2 + 1/2))
        ares[1] += np.dot(Svki, (1/Xass - 1/2) * dXass)
    else:
        Xass = None
    return ares, Xass


def d2ares_drho(self, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, d_kk, d_kl, d_kl3, d_kk03, x0_kl = temp_aux[:8]
    a1vdw_ctekl, a1vdwkl, x0_a1, x0_a2, I_lambdaskl = temp_aux[8:13]
    J_lambdaskl, d_ii, d_ii3, x0_ii, a1vdw_cteii, a1vdwii = temp_aux[13:19]
    tetha, x0_a1ii, x0_a2ii, x0_g1ii, x0_g2ii, I_lambdasii = temp_aux[19:25]
    J_lambdasii, x0i_matrix, beps_ii, beps_ii2 = temp_aux[25:29]

    dxhi00_drho = self.dxhi00_drho
    xhi00 = rho*dxhi00_drho

    sigma_kl3 = self.sigma_kl3
    xs_k = self.xs_k
    xs_m = self.xs_m
    Ckl = self.Ckl
    eps_kl = self.eps_kl
    f1, f2, f3 = self.f1, self.f2, self.f3
    f4, f5, f6 = self.f4, self.f5, self.f6
    # lambdaskl = self.lambdaskl
    ccteskl = self.ccteskl

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
    d2a1kl, d2a2kl = d2a1sB_dxhi00_eval(xhi00, xhix, xhix_vec[:3], xs_m,
                                        I_lambdaskl, J_lambdaskl, ccteskl,
                                        a1vdwkl, a1vdw_ctekl, dxhix_dxhi00)

    # zero order pertubation
    d2aHS = d2ahs_dxhi00(xhi, dxhi_dxhi00)

    # first order pertubation
    suma1_monomer = Ckl * np.sum(d2a1kl * x0_a1, axis=1)
    d2a1m = d2a1_dxhi00(xs_k, suma1_monomer)

    # second order pertubation
    khs, dkhs, d2khs = d2kHS_dxhi00(xhix, dxhix_dxhi00)
    suma2_monomer = Ckl**2 * np.sum(d2a2kl * x0_a2, axis=1)
    d2a2m = d2a2_dxhi00(xs_k, khs, dkhs, d2khs, xhixm, dxhixm_dxhi00,
                        suma2_monomer, eps_kl, f1, f2, f3)

    # third order pertubaton
    d2a3m = d2a3_dxhi00(xs_k, xhixm, dxhixm_dxhi00, eps_kl, f4, f5, f6)

    d2amono = xs_m * (d2aHS + beta * d2a1m + beta2 * d2a2m + beta3 * d2a3m)

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
    khs, dkhs, d2khs, d3khs = d3kHS_dxhi00(xhix, dxhix_dxhi00)
    d3suma2_chain = Cii2 * np.sum(d3a2ii * x0_a2ii, axis=1)
    d3a2new = d3a2new_dxhi00(khs, dkhs, d2khs, d3khs, d3suma2_chain, eps_ii)
    d2suma_g2 = Cii2 * np.sum(d3a2ii[:3] * x0_g2ii, axis=1)
    d2g2m = d2g2mca_dxhi00(xhi00, khs, dkhs, d2khs, xs_m, d3a2new, d2suma_g2,
                           eps_ii, a1vdw_cteii)
    d2g2s = d2g2m * (1. + d2gc[0])
    d2g2s[1] += d2g2m[0] * d2gc[1]
    d2g2s[2] += 2. * d2g2m[1] * d2gc[1] + d2g2m[0] * d2gc[2]

    d2lng = d2lngmie_dxhi00(d2gHS, d2g1s, d2g2s, beps_ii, beps_ii2)

    d2achain = - (xs_m - 1.)*d2lng

    ares = d2amono + d2achain
    ares *= self.dxhi00_2

    if self.asso_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        T_ad = temp_aux[29]
        Fklab = temp_aux[30]
        sigma_x3 = np.matmul(np.matmul(sigma_kl3, xs_k), xs_k)
        drho_ad = xs_m * sigma_x3
        rho_ad = rho * drho_ad

        Iijklab, dIijklab_drho, d2Iijklab_drho = d2Iab_drho(rho_ad, T_ad,
                                                            drho_ad)

        diagasso = self.diagasso
        vki_asso = self.vki[self.group_asso_index]
        DIJ = self.DIJ
        Dijklab = self.kAB_kl * Fklab * Iijklab
        dDijklab_drho = self.kAB_kl * Fklab * dIijklab_drho
        d2Dijklab_drho = self.kAB_kl * Fklab * d2Iijklab_drho

        Xass = Xass_solver(rho, vki_asso, DIJ, Dijklab, diagasso, Xass)
        CIJ = CIJ_matrix(rho, vki_asso, Xass, DIJ, Dijklab, diagasso)
        dXass = dXass_drho(rho, vki_asso, Xass, DIJ, Dijklab, dDijklab_drho,
                           CIJ)
        d2Xass = d2Xass_drho(rho, vki_asso, Xass, dXass, DIJ, Dijklab,
                             dDijklab_drho, d2Dijklab_drho, CIJ)

        aux1 = np.log(Xass) - Xass/2 + 1/2
        aux2 = 1/Xass - 1/2
        Svki = self.S * vki_asso
        ares[0] += np.dot(Svki, aux1)
        ares[1] += np.dot(Svki, aux2 * dXass)
        ares[2] += np.dot(Svki, -(dXass/Xass)**2+d2Xass*aux2)
    else:
        Xass = None
    return ares, Xass
