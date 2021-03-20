import numpy as np
from .monomer_aux import Xi, dXi, d2Xi
from .monomer_aux import dkHS, d2kHS, d3kHS

from .a1sB_monomer import da1B_eval, d2a1B_eval, d3a1B_eval

from .aHS_monomer import ahs, dahs_deta, d2ahs_deta
from .a2m_monomer import a2m, da2m_deta, d2a2m_deta
from .a2m_monomer import da2m_new_deta, d2a2m_new_deta, d3a2m_new_deta
from .a3m_monomer import a3m, da3m_deta, d2a3m_deta

from .gdHS_chain import gdHS, dgdHS_drho, d2gdHS_drho
from .g1sigma_chain import g1sigma, dg1sigma_drho, d2g1sigma_drho
from .g2sigma_chain import g2MCA, dg2MCA_drho, d2g2MCA_drho
from .g2sigma_chain import gammac, dgammac_deta, d2gammac_deta
from .lngmie_chain import lngmie, dlngmie_drho, d2lngmie_drho

from .association_aux import Xass_solver, Iab, dIab_drho
from .association_aux import d2Iab_drho, dXass_drho, d2Xass_drho

from .polarGV import Apol, dApol_drho, d2Apol_drho


def ares(self, rho, temp_aux, Xass0=None):
    beta, beta2, beta3, dia, dia3, x0, x03, x0_a1, x0_a2 = temp_aux[:9]
    x0_a12, x0_a22, I_lambdas, J_lambdas, beps, beps2, tetha = temp_aux[9:16]
    x0_vector, cte_g1s, cte_g2s = temp_aux[16:19]

    eta, deta = self.eta_bh(rho, dia3)
    nsigma = self.eta_sigma(rho)
    # Parameters needed for evaluating the helmothlz contributions
    a1sb_a1, a1sb_a2 = da1B_eval(eta, I_lambdas, J_lambdas, self.lambdas,
                                 self.cctes, self.eps)
    dkhs = dkHS(eta)
    xi = Xi(x03, nsigma, self.f1, self.f2, self.f3)

    cte_a2m = self.cte_a2m
    eps3 = self.eps3

    # Monomer contribution
    ahs_eval = ahs(eta)
    a1m_eval = self.c*np.matmul(a1sb_a1, x0_a1)
    suma_a2 = np.matmul(a1sb_a2, x0_a2)
    a2m_eval = a2m(suma_a2[0], dkhs[0], xi, cte_a2m)
    a3m_eval = a3m(x03, nsigma, eps3, self.f4, self.f5, self.f6)

    a_mono = ahs_eval + beta*a1m_eval[0] + beta2*a2m_eval + beta3*a3m_eval
    a_mono *= self.ms

    # chain contribution calculation
    ghs = gdHS(x0_vector, eta)

    # g1sigma
    suma_g1 = self.c * np.dot(a1sb_a1[0], x0_a12)
    g1s = g1sigma(rho, suma_g1, a1m_eval[1], deta, cte_g1s)

    # g2sigma
    gc = gammac(x0, nsigma, self.alpha, tetha)
    da2m_new = da2m_new_deta(suma_a2, dkhs, cte_a2m)

    suma_g2 = self.c2*np.dot(a1sb_a2[0],  x0_a22)
    g2m = g2MCA(rho, suma_g2, da2m_new, dkhs[0], self.eps, cte_g2s, deta)
    g2s = (1+gc)*g2m

    lng = lngmie(ghs, g1s, g2s, beps, beps2)
    a_chain = - (self.ms - 1. + self.ring*eta)*lng

    # Total Helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:

        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        Fab = temp_aux[19]
        Kab = temp_aux[20]
        iab = Iab(Kab, eta)
        Dab = self.sigma3 * Fab * iab
        Dabij = np.zeros([self.nsites, self.nsites])
        Dabij[self.indexabij] = Dab
        KIJ = rho * (self.DIJ*Dabij)
        Xass = Xass_solver(self.nsites, KIJ, self.diagasso, Xass0)
        a += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[21]
        apolar = Apol(rho, eta, epsa, self.anij, self.bnij,
                      self.cnijk, self.mupolad2, self.npol, self.sigma3)
        a += apolar

    return a, Xass


def dares_drho(self, rho, temp_aux, Xass0=None):
    beta, beta2, beta3, dia, dia3, x0, x03, x0_a1, x0_a2 = temp_aux[:9]
    x0_a12, x0_a22, I_lambdas, J_lambdas, beps, beps2, tetha = temp_aux[9:16]
    x0_vector, cte_g1s, cte_g2s = temp_aux[16:19]

    eta, deta = self.eta_bh(rho, dia3)
    nsigma = self.eta_sigma(rho)

    drho = np.array([1., deta, deta**2])

    # Parameters needed for evaluating the helmothlz contributions
    a1sb_a1, a1sb_a2 = d2a1B_eval(eta, I_lambdas, J_lambdas, self.lambdas,
                                  self.cctes, self.eps)
    dkhs = d2kHS(eta)
    dxi = dXi(x03, nsigma, self.f1, self.f2, self.f3)

    cte_a2m = self.cte_a2m
    eps3 = self.eps3

    # monomer evaluation
    ahs_eval = dahs_deta(eta)
    a1m_eval = self.c*np.matmul(a1sb_a1, x0_a1)
    suma_a2 = np.matmul(a1sb_a2, x0_a2)
    a2m_eval = da2m_deta(suma_a2[:2], dkhs[:2], dxi, cte_a2m)
    a3m_eval = da3m_deta(x03, nsigma, eps3, self.f4, self.f5, self.f6)

    a_mono = ahs_eval + beta*a1m_eval[:2] + beta2*a2m_eval + beta3*a3m_eval
    a_mono *= self.ms * drho[:2]

    # chain contribution calculation
    dghs = dgdHS_drho(x0_vector, eta, drho)

    # g1sigma
    suma_g1 = self.c * np.dot(a1sb_a1[:2], x0_a12)
    suma_g1 *= drho[:2]
    d2a1m_drho = a1m_eval[1:]*drho[1:]
    dg1s = dg1sigma_drho(rho, suma_g1, d2a1m_drho, cte_g1s)

    # g2sigma
    dgc = dgammac_deta(x03, nsigma, self.alpha, tetha)
    dgc *= drho[:2]

    da2m_new = d2a2m_new_deta(suma_a2, dkhs, cte_a2m)
    da2m_new_drho = da2m_new*drho[1:]

    suma_g2 = self.c2*np.dot(a1sb_a2[:2],  x0_a22)
    suma_g2 *= drho[:2]

    dkhs_drho = dkhs[:2]*drho[:2]

    dg2m = dg2MCA_drho(rho, suma_g2, da2m_new_drho, dkhs_drho, self.eps,
                       cte_g2s)

    dg2s = dg2m * (1. + dgc[0])
    dg2s[1] += dgc[1] * dg2m[0]

    dlng = dlngmie_drho(dghs, dg1s, dg2s, beps, beps2)
    a_chain = - (self.ms - 1. + self.ring*eta)*dlng

    # Total helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        Fab = temp_aux[19]
        Kab = temp_aux[20]
        iab, diab = dIab_drho(Kab, eta, deta)
        # Fab = np.exp(beta * self.eABij) - 1
        Dab = self.sigma3 * Fab * iab
        dDab = self.sigma3 * Fab * diab
        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])

        Dabij[self.indexabij] = Dab
        dDabij_drho[self.indexabij] = dDab
        KIJ = rho * (self.DIJ*Dabij)
        Xass = Xass_solver(self.nsites, KIJ, self.diagasso, Xass0)

        CIJ = rho * np.tile(Xass**2, (self.nsites, 1)).T * Dabij * self.DIJ
        CIJ[self.diagasso] += 1.
        dXass = dXass_drho(rho, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
        a[0] += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
        a[1] += np.dot(self.S, (1/Xass - 1/2) * dXass)
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[21]
        dapolar = dApol_drho(rho, eta, deta, epsa, self.anij,
                             self.bnij, self.cnijk, self.mupolad2,
                             self.npol, self.sigma3)
        a += dapolar

    return a, Xass


def d2ares_drho(self, rho, temp_aux, Xass0=None):

    beta, beta2, beta3, dia, dia3, x0, x03, x0_a1, x0_a2 = temp_aux[:9]
    x0_a12, x0_a22, I_lambdas, J_lambdas, beps, beps2, tetha = temp_aux[9:16]
    x0_vector, cte_g1s, cte_g2s = temp_aux[16:19]

    eta, deta = self.eta_bh(rho, dia3)
    nsigma = self.eta_sigma(rho)

    drho = np.array([1., deta, deta**2, deta**3])

    # Parameters needed for evaluating the helmothlz contributions
    a1sb_a1, a1sb_a2 = d3a1B_eval(eta, I_lambdas, J_lambdas, self.lambdas,
                                  self.cctes, self.eps)

    dkhs = d3kHS(eta)
    dxi = d2Xi(x03, nsigma, self.f1, self.f2, self.f3)

    cte_a2m = self.cte_a2m
    eps3 = self.eps3

    # monomer evaluation
    ahs_eval = d2ahs_deta(eta)
    a1m_eval = self.c*np.matmul(a1sb_a1, x0_a1)
    suma_a2 = np.matmul(a1sb_a2, x0_a2)
    a2m_eval = d2a2m_deta(suma_a2[:3], dkhs[:3], dxi, cte_a2m)
    a3m_eval = d2a3m_deta(x03, nsigma, eps3, self.f4, self.f5, self.f6)

    a_mono = ahs_eval + beta*a1m_eval[:3] + beta2*a2m_eval + beta3*a3m_eval
    a_mono *= self.ms * drho[:3]

    # chain contribution calculation
    dghs = d2gdHS_drho(x0_vector, eta, drho)

    # g1sigma
    suma_g1 = self.c * np.dot(a1sb_a1[:3], x0_a12)
    suma_g1 *= drho[:3]
    d3a1m_drho = a1m_eval[1:]*drho[1:]
    dg1s = d2g1sigma_drho(rho, suma_g1, d3a1m_drho, cte_g1s)

    # g2sigma
    dgc = d2gammac_deta(x03, nsigma, self.alpha, tetha)
    dgc *= drho[:3]

    da2m_new = d3a2m_new_deta(suma_a2, dkhs, cte_a2m)
    da2m_new_drho = da2m_new*drho[1:]

    suma_g2 = self.c2*np.dot(a1sb_a2[:3],  x0_a22)
    suma_g2 *= drho[:3]

    dkhs_drho = dkhs[:3]*drho[:3]

    dg2m = d2g2MCA_drho(rho, suma_g2, da2m_new_drho, dkhs_drho, self.eps,
                        cte_g2s)

    dg2s = dg2m * (1. + dgc[0])
    dg2s[1] += dgc[1] * dg2m[0]
    dg2s[2] += dgc[2]*dg2m[0] + 2.*dgc[1]*dg2m[1]

    d2lng = d2lngmie_drho(dghs, dg1s, dg2s, beps, beps2)
    a_chain = - (self.ms - 1. + self.ring*eta)*d2lng

    # Total Helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        Fab = temp_aux[19]
        Kab = temp_aux[20]
        iab, diab, d2iab = d2Iab_drho(Kab, eta, deta)

        # Fab = np.exp(beta * self.eABij) - 1.
        Dab = self.sigma3 * Fab * iab
        dDab = self.sigma3 * Fab * diab
        d2Dab = self.sigma3 * Fab * d2iab

        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])
        d2Dabij_drho = np.zeros([self.nsites, self.nsites])
        Dabij[self.indexabij] = Dab
        dDabij_drho[self.indexabij] = dDab
        d2Dabij_drho[self.indexabij] = d2Dab

        KIJ = rho * (self.DIJ*Dabij)
        Xass = Xass_solver(self.nsites, KIJ, self.diagasso, Xass0)

        CIJ = rho * np.tile(Xass**2, (self.nsites, 1)).T * Dabij * self.DIJ
        CIJ[self.diagasso] += 1.
        dXass = dXass_drho(rho, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
        d2Xass = d2Xass_drho(rho, Xass, dXass, self.DIJ, Dabij, dDabij_drho,
                             d2Dabij_drho, CIJ)
        a[0] += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
        a[1] += np.dot(self.S, (1/Xass - 1/2) * dXass)
        a[2] += np.dot(self.S, - (dXass/Xass)**2 + d2Xass * (1/Xass - 1/2))
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[21]
        dapolar = d2Apol_drho(rho, eta, deta, epsa, self.anij,
                              self.bnij, self.cnijk, self.mupolad2,
                              self.npol, self.sigma3)

        a += dapolar

    return a, Xass
