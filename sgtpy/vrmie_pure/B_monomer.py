from __future__ import division, print_function, absolute_import
import numpy as np
# from .monomer_aux import I_lam, J_lam


def B(eta, Ilam, Jlam, eps):
    # B calculation Eq 33
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)
    eta13 = (1-eta)**3
    ter1 = (1-eta/2)*Ilam/eta13 - 9.*eta*(1+eta)*Jlam/(2.*eta13)
    b = 12.*eta*eps*ter1
    return b


def dB(eta, Ilam, Jlam, eps):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    eta13 = (1-eta)**3.
    ter1 = (1.-eta/2.)*Ilam/eta13 - 9.*eta*(1+eta)*Jlam/(2.*eta13)
    b = 12.*eta*eps*ter1

    # first derivative
    ter1b1 = Ilam*(-2. + (eta-2.)*eta) + Jlam*18.*eta*(1.+2.*eta)
    ter2b1 = -6.*eps/(eta-1.)**4.
    db = ter1b1*ter2b1
    return np.hstack([b, db])


def d2B(eta, Ilam, Jlam, eps):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    eta13 = (1.-eta)**3.
    ter1 = (1.-eta/2.)*Ilam/eta13 - 9.*eta*(1.+eta)*Jlam/(2.*eta13)
    b = 12.*eta*eps*ter1

    # first derivative
    db = Ilam*(-2. + (eta-2)*eta) + Jlam*18.*eta*(1.+2.*eta)
    db *= (-6.*eps/(eta-1.)**4)

    # second derivative
    d2b = Ilam*(-5. + (eta-2)*eta) + Jlam*9.*(1 + eta*(7. + 4.*eta))
    d2b *= 12.*eps/(eta-1)**5
    return np.hstack([b, db, d2b])


def d3B(eta, Ilam, Jlam, eps):
    # I = I_lam(x0, lam)
    # J = J_lam(x0, lam)

    eta13 = (1.-eta)**3.
    ter1 = (1.-eta/2.)*Ilam/eta13 - 9.*eta*(1.+eta)*Jlam/(2.*eta13)
    b = 12.*eta*eps*ter1

    # first derivative
    db = Ilam*(-2. + (eta-2.)*eta) + Jlam*18.*eta*(1.+2.*eta)
    db *= (-6.*eps/(eta-1.)**4.)

    # second derivative
    d2b = Ilam*(-5. + (eta-2.)*eta) + Jlam*9.*(1. + eta*(7.+4.*eta))
    d2b *= 12.*eps/(eta-1.)**5.

    d3b = Ilam*(9. - (eta-2.)*eta) - Jlam*36.*(1. + eta*(3. + eta))
    d3b *= (36.*eps/(1.-eta)**6.)
    return np.hstack([b, db, d2b, d3b])
