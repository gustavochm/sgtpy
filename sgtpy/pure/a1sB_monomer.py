from __future__ import division, print_function, absolute_import
import numpy as np
from .B_monomer import B, dB, d2B, d3B
from .a1s_monomer import a1s, da1s, d2a1s, d3a1s


def a1sB(eta, Ilam, Jlam, lam, cctes, eps):
    b = B(eta, Ilam, Jlam, eps)
    a1 = a1s(eta, lam, cctes, eps)
    a1b = a1 + b
    return a1b


def da1sB(eta, Ilam, Jlam, lam, cctes, eps):
    db = dB(eta, Ilam, Jlam, eps)
    da1 = da1s(eta, lam, cctes, eps)

    da1b = da1 + db

    return da1b


def d2a1sB(eta, Ilam, Jlam, lam, cctes, eps):
    d2b = d2B(eta, Ilam, Jlam, eps)
    d2a1 = d2a1s(eta, lam, cctes, eps)
    d2a1b = d2a1 + d2b
    return d2a1b


def d3a1sB(eta, Ilam, Jlam, lam, cctes, eps):
    d3b = d3B(eta, Ilam, Jlam, eps)
    d3a1 = d3a1s(eta, lam, cctes, eps)
    d3a1b = d3a1 + d3b
    return d3a1b


# Function for creating arrays with a1s + B at all lambdas
def a1B_eval(eta, I_lambdas, J_lambdas, lambdas, cctes, eps):
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdas
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdas
    lambda_a, lambda_r, lambda_ar = lambdas
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes

    a1sb_a = a1sB(eta, I_la, J_la, lambda_a, cctes_la, eps)
    a1sb_r = a1sB(eta, I_lr, J_lr, lambda_r, cctes_lr, eps)
    a1sb_2a = a1sB(eta, I_2la, J_2la, 2*lambda_a, cctes_2la, eps)
    a1sb_2r = a1sB(eta, 2*lambda_r, cctes_2lr, eps)
    a1sb_ar = a1sB(eta, lambda_ar, cctes_lar, eps)

    a1sb_a1 = np.hstack([a1sb_a, a1sb_r])
    a1sb_a2 = np.hstack([a1sb_2a, a1sb_ar, a1sb_2r])
    return a1sb_a1, a1sb_a2


# Function for creating arrays with a1s + B and it's first derivative
# at all lambdas
def da1B_eval(eta, I_lambdas, J_lambdas, lambdas, cctes, eps):
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdas
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdas
    lambda_a, lambda_r, lambda_ar = lambdas
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes

    da1sb_a = da1sB(eta, I_la, J_la, lambda_a, cctes_la, eps)
    da1sb_r = da1sB(eta, I_lr, J_lr, lambda_r, cctes_lr, eps)
    da1sb_2a = da1sB(eta, I_2la, J_2la, 2*lambda_a, cctes_2la, eps)
    da1sb_2r = da1sB(eta, I_2lr, J_2lr, 2*lambda_r, cctes_2lr, eps)
    da1sb_ar = da1sB(eta, I_lar, J_lar, lambda_ar, cctes_lar, eps)

    a1sb_a1 = np.column_stack([da1sb_a, da1sb_r])
    a1sb_a2 = np.column_stack([da1sb_2a, da1sb_ar, da1sb_2r])
    return a1sb_a1, a1sb_a2


# Function for creating arrays with a1s + B and it's first
# and second derivative at all lambdas
def d2a1B_eval(eta, I_lambdas, J_lambdas, lambdas, cctes, eps):

    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdas
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdas
    lambda_a, lambda_r, lambda_ar = lambdas
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes

    d2a1sb_a = d2a1sB(eta, I_la, J_la, lambda_a, cctes_la, eps)
    d2a1sb_r = d2a1sB(eta, I_lr, J_lr, lambda_r, cctes_lr, eps)
    d2a1sb_2a = d2a1sB(eta, I_2la, J_2la, 2*lambda_a, cctes_2la, eps)
    d2a1sb_2r = d2a1sB(eta, I_2lr, J_2lr, 2*lambda_r, cctes_2lr, eps)
    d2a1sb_ar = d2a1sB(eta,  I_lar, J_lar, lambda_ar, cctes_lar, eps)

    a1sb_a1 = np.column_stack([d2a1sb_a, d2a1sb_r])
    a1sb_a2 = np.column_stack([d2a1sb_2a, d2a1sb_ar, d2a1sb_2r])

    return a1sb_a1, a1sb_a2


# Function for creating arrays with a1s + B and it's first, second and third
# derivative at all lambdas
def d3a1B_eval(eta, I_lambdas, J_lambdas, lambdas, cctes, eps):

    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdas
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdas
    lambda_a, lambda_r, lambda_ar = lambdas
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes

    d3a1sb_a = d3a1sB(eta, I_la, J_la, lambda_a, cctes_la, eps)
    d3a1sb_r = d3a1sB(eta, I_lr, J_lr, lambda_r, cctes_lr, eps)
    d3a1sb_2a = d3a1sB(eta, I_2la, J_2la, 2*lambda_a, cctes_2la, eps)
    d3a1sb_2r = d3a1sB(eta, I_2lr, J_2lr, 2*lambda_r, cctes_2lr, eps)
    d3a1sb_ar = d3a1sB(eta,  I_lar, J_lar, lambda_ar, cctes_lar, eps)

    a1sb_a1 = np.column_stack([d3a1sb_a, d3a1sb_r])
    a1sb_a2 = np.column_stack([d3a1sb_2a, d3a1sb_ar, d3a1sb_2r])

    return a1sb_a1, a1sb_a2


def x0lambda_eval(x0, lambda_a, lambda_r, lambda_ar):
    x0la = x0**lambda_a
    x0lr = x0**lambda_r
    x02la = x0**(2*lambda_a)
    x02lr = x0**(2*lambda_r)
    x0lar = x0**lambda_ar

    x0_a1 = np.hstack([x0la, -x0lr])
    x0_a2 = np.hstack([x02la, -2*x0lar, x02lr])

    x0_a12 = np.hstack([lambda_a*x0la, -lambda_r*x0lr])
    x0_a22 = np.hstack([lambda_a*x02la, -lambda_ar*x0lar, lambda_r*x02lr])
    return x0_a1, x0_a2, x0_a12, x0_a22
