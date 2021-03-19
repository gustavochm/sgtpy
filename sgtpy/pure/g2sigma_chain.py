import numpy as np


# Equation 65
def g2MCA(rho, suma_g2, da2m_new_deta, khs, eps, cte_g2s, deta_drho):

    da2 = da2m_new_deta * deta_drho
    g2 = (3.*da2 - eps*khs*suma_g2/rho) * cte_g2s

    return g2


def dg2MCA_drho(rho, suma_g2, d2a2m_new_drho, dkhs_drho, eps, cte_g2s):

    khs, dkhs = dkhs_drho
    d2a2 = d2a2m_new_drho
    suma1 = suma_g2 * eps

    dg2 = 3*d2a2 - suma1 * khs/rho
    dg2[1] += suma1[0] * khs/rho**2 - suma1[0] * dkhs / rho
    dg2 *= cte_g2s

    return dg2


def d2g2MCA_drho(rho, suma_g2, d3a2m_new_drho, dkhs_drho, eps, cte_g2s):

    khs, dkhs, d2khs = dkhs_drho

    d3a2 = d3a2m_new_drho

    suma1 = suma_g2 * eps
    sum1, dsum1, d2sum1 = suma1

    d2g2 = 3*d3a2 - suma1 * khs/rho
    d2g2[1] += sum1 * khs/rho**2 - sum1 * dkhs / rho
    d2g2[2] += -2*khs*sum1/rho**3 + 2*sum1*dkhs/rho**2 + 2*khs*dsum1/rho**2
    d2g2[2] += -2*dkhs*dsum1/rho - sum1*d2khs/rho
    d2g2 *= cte_g2s

    return d2g2


phi70 = 10.
phi71 = 10.
phi72 = 0.57
phi73 = -6.7
phi74 = -8


# Equation 63
def gammac(x0, nsigma, alpha, tetha):
    gc = phi70*(-np.tanh(phi71*(phi72-alpha))+1)
    gc *= nsigma*tetha*np.exp(phi73*nsigma + phi74*nsigma**2)
    return gc


def dgammac_deta(x03, nsigma, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte * np.exp(phi73*nsigma + phi74*nsigma**2)

    dg = np.array([g, g])
    dg[0] *= nsigma

    dg[1] *= (1. + nsigma*(phi73 + 2*phi74*nsigma))
    dg[1] *= x03

    return dg


def d2gammac_deta(x03, nsigma, alpha, tetha):

    cte = phi70*(-np.tanh(phi71*(phi72-alpha))+1)*tetha
    g = cte * np.exp(phi73*nsigma+phi74*nsigma**2)

    dg = np.array([g, g, g])
    dg[0] *= nsigma

    dg[1] *= (1. + nsigma*(phi73 + 2*phi74*nsigma))
    dg[1] *= x03

    aux1 = phi73**2 + 6*phi74+4*phi74*nsigma*(phi73 + phi74*nsigma)
    dg[2] *= (2*phi73 + nsigma * aux1) * x03**2


    return dg
