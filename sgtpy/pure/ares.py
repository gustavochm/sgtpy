import numpy as np
from .monomer_aux import Xi, dXi, d2Xi
from .monomer_aux import dkHS, d2kHS, d3kHS

from .a1sB_monomer import da1B_eval, d2a1B_eval, d3a1B_eval
from .a1sB_monomer import x0lambda_eval

from .pertubaciones_eval import ahs, dahs_deta, d2ahs_deta
from .pertubaciones_eval import a1m
from .pertubaciones_eval import a2m,  da2m_deta, d2a2m_deta
from .pertubaciones_eval import da2m_new_deta, d2a2m_new_deta, d3a2m_new_deta
from .pertubaciones_eval import a3m, da3m_deta, d2a3m_deta

from .monomer import amono, damono_drho, d2amono_drho
from .chain import achain, dachain_drho, d2achain_drho

from .association_aux import Xass_solver, Iab, dIab_drho
from .association_aux import d2Iab_drho, dXass_drho, d2Xass_drho

from .polarGV import Apol, dApol_drho, d2Apol_drho


def ares(self, rho, temp_aux, Xass0=None):
    beta, dia, tetha, x0, x03 = temp_aux[0:5]

    eta, deta = self.eta_bh(rho, dia)
    nsigma = self.eta_sigma(rho)

    # Parameters needed for evaluating the helmothlz contributions
    x0_a1, x0_a2, x0_a12, x0_a22 = x0lambda_eval(x0, self.lambda_a,
                                                 self.lambda_r, self.lambda_ar)
    a1sb_a1, a1sb_a2 = da1B_eval(x0, eta, self.lambda_a, self.lambda_r,
                                 self.lambda_ar, self.cctes, self.eps)
    dkhs = dkHS(eta)
    xi = Xi(x03, nsigma, self.f1, self.f2, self.f3)
    da2m_new = da2m_new_deta(x0_a2, a1sb_a2,  dkhs, self.c, self.eps)

    # Monomer contribution
    ahs_eval = ahs(eta)
    a1m_eval = a1m(x0_a1, a1sb_a1, self.c)
    a2m_eval = a2m(x0_a2, a1sb_a2[0], dkhs[0], xi, self.c, self.eps)
    a3m_eval = a3m(x03, nsigma, self.eps, self.f4, self.f5, self.f6)
    a_mono = amono(ahs_eval, a1m_eval[0], a2m_eval, a3m_eval, beta,
                   self.ms)

    # Chain contribution
    a_chain = achain(x0, eta, x0_a12, a1sb_a1[0], a1m_eval[1], x03, nsigma,
                     self.alpha, tetha, x0_a22, a1sb_a2[0], da2m_new,
                     dkhs[0], dia, deta, rho, beta,  self.eps, self.c,
                     self.ms)

    # Total Helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:

        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0

        iab = Iab(dia, self.rcij, self.rdij, eta,  self.sigma3)
        Fab = temp_aux[5]
        Dab = self.sigma3 * Fab * iab
        Dabij = np.zeros([self.nsites, self.nsites])
        Dabij[self.indexabij] = Dab
        KIJ = rho * (self.DIJ*Dabij)
        Xass = Xass_solver(self.nsites, KIJ, self.diagasso, Xass0)
        a += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[6]
        apolar = Apol(rho, eta, epsa, self.anij, self.bnij,
                      self.cnijk, self.mupolad2, self.npol, self.sigma3)
        a += apolar

    return a, Xass


def dares_drho(self, rho, temp_aux, Xass0=None):

    beta, dia, tetha, x0, x03 = temp_aux[0:5]

    eta, deta = self.eta_bh(rho, dia)
    nsigma = self.eta_sigma(rho)

    drho = np.array([1., deta, deta**2])

    # Parameters needed for evaluating the helmothlz contributions
    x0_a1, x0_a2, x0_a12, x0_a22 = x0lambda_eval(x0, self.lambda_a,
                                                 self.lambda_r,
                                                 self.lambda_ar)
    a1sb_a1, a1sb_a2 = d2a1B_eval(x0, eta, self.lambda_a, self.lambda_r,
                                  self.lambda_ar, self.cctes, self.eps)
    dkhs = d2kHS(eta)
    dxi = dXi(x03, nsigma, self.f1, self.f2, self.f3)
    da2m_new = d2a2m_new_deta(x0_a2, a1sb_a2, dkhs, self.c, self.eps)

    # Monomer contribution
    ahs_eval = dahs_deta(eta)
    a1m_eval = a1m(x0_a1, a1sb_a1, self.c)
    a2m_eval = da2m_deta(x0_a2, a1sb_a2[:2], dkhs[:2], dxi, self.c,
                         self.eps)
    a3m_eval = da3m_deta(x03, nsigma, self.eps, self.f4, self.f5, self.f6)
    a_mono = damono_drho(ahs_eval, a1m_eval[:2], a2m_eval, a3m_eval, beta,
                         drho[:2], self.ms)

    # Chain contribution
    a_chain = dachain_drho(x0, eta, x0_a12, a1sb_a1[:2], a1m_eval, x03,
                           nsigma, self.alpha, tetha, x0_a22, a1sb_a2[:2],
                           da2m_new, dkhs[:2], dia, drho, rho, beta,
                           self.eps, self.c, self.ms)

    # Total helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        iab, diab = dIab_drho(dia, self.rcij, self.rdij, eta, deta,
                              self.sigma3)
        # Fab = np.exp(beta * self.eABij) - 1
        Fab = temp_aux[5]
        Dab = self.sigma3 * Fab * iab
        dDab = self.sigma3 * Fab * diab
        Dabij = np.zeros([self.nsites, self.nsites])
        dDabij_drho = np.zeros([self.nsites, self.nsites])

        Dabij[self.indexabij] = Dab
        dDabij_drho[self.indexabij] = dDab
        KIJ = rho * (self.DIJ*Dabij)
        Xass = Xass_solver(self.nsites, KIJ, self.diagasso, Xass0)
        CIJ = rho * Xass**2 * Dabij * self.DIJ

        CIJ[self.diagasso] += 1.
        CIJ = CIJ.T

        dXass = dXass_drho(rho, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
        a[0] += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
        a[1] += np.dot(self.S, (1/Xass - 1/2) * dXass)
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[6]
        dapolar = dApol_drho(rho, eta, deta, epsa, self.anij,
                             self.bnij, self.cnijk, self.mupolad2,
                             self.npol, self.sigma3)
        a += dapolar

    return a, Xass


def d2ares_drho(self, rho, temp_aux, Xass0=None):

    beta, dia, tetha, x0, x03 = temp_aux[0:5]

    eta, deta = self.eta_bh(rho, dia)
    nsigma = self.eta_sigma(rho)

    drho = np.array([1., deta, deta**2, deta**3])

    # Parameters needed for evaluating the helmothlz contributions
    x0_a1, x0_a2, x0_a12, x0_a22 = x0lambda_eval(x0, self.lambda_a,
                                                 self.lambda_r,
                                                 self.lambda_ar)
    a1sb_a1, a1sb_a2 = d3a1B_eval(x0, eta, self.lambda_a, self.lambda_r,
                                  self.lambda_ar, self.cctes, self.eps)

    dkhs = d3kHS(eta)
    dxi = d2Xi(x03, nsigma, self.f1, self.f2, self.f3)
    da2m_new = d3a2m_new_deta(x0_a2, a1sb_a2,  dkhs, self.c, self.eps)

    # Monomer contribution
    ahs_eval = d2ahs_deta(eta)
    a1m_eval = a1m(x0_a1, a1sb_a1, self.c)
    a2m_eval = d2a2m_deta(x0_a2, a1sb_a2[:3], dkhs[:3], dxi, self.c,
                          self.eps)
    a3m_eval = d2a3m_deta(x03, nsigma, self.eps, self.f4, self.f5, self.f6)
    a_mono = d2amono_drho(ahs_eval, a1m_eval[:3], a2m_eval, a3m_eval, beta,
                          drho[:3], self.ms)

    # Chain contribution
    a_chain = d2achain_drho(x0, eta, x0_a12, a1sb_a1[:3], a1m_eval, x03,
                            nsigma, self.alpha, tetha, x0_a22, a1sb_a2[:3],
                            da2m_new, dkhs[:3], dia, drho, rho, beta,
                            self.eps, self.c, self.ms)

    # Total Helmolthz
    a = a_mono + a_chain

    if self.assoc_bool:
        if Xass0 is None:
            Xass = 0.2 * np.ones(self.nsites)
        else:
            Xass = 1. * Xass0
        iab, diab, d2iab = d2Iab_drho(dia, self.rcij, self.rdij, eta, deta,
                                      self.sigma3)

        # Fab = np.exp(beta * self.eABij) - 1.
        Fab = temp_aux[5]
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
        CIJ = rho * Xass**2 * Dabij * self.DIJ

        CIJ[self.diagasso] += 1.
        CIJ = CIJ.T
        dXass = dXass_drho(rho, Xass, self.DIJ, Dabij, dDabij_drho, CIJ)
        d2Xass = d2Xass_drho(rho, Xass, dXass, self.DIJ, Dabij, dDabij_drho,
                             d2Dabij_drho, CIJ)
        a[0] += np.dot(self.S, (np.log(Xass) - Xass/2 + 1/2))
        a[1] += np.dot(self.S, (1/Xass - 1/2) * dXass)
        a[2] += np.dot(self.S, - (dXass/Xass)**2 + d2Xass * (1/Xass - 1/2))
    else:
        Xass = None

    if self.polar_bool:
        epsa = temp_aux[6]
        dapolar = d2Apol_drho(rho, eta, deta, epsa, self.anij,
                              self.bnij, self.cnijk, self.mupolad2,
                              self.npol, self.sigma3)

        a += dapolar

    return a, Xass
