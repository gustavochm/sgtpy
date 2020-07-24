import numpy as np
from ..sgt import sgt_mix
from scipy.optimize import minimize_scalar


def fobj_beta(beta, iftexp, rho1, rho2, T, P, eos):
    bij = np.array([[0, beta], [beta, 0]])
    eos.beta_sgt(bij)
    tenb = np.zeros_like(iftexp)
    n = len(iftexp)
    for i in range(n):
        tenb[i] = sgt_mix(rho1[i], rho2[i], T[i], P[i], eos)
    fo = np.mean((1 - tenb/iftexp)**2)
    return fo


def fit_beta(beta0, ExpTension, EquilibriumInfo, eos):
    rho1, rho2, T, P = EquilibriumInfo
    args = (ExpTension, rho1, rho2, T, P, eos)
    opti = minimize_scalar(fobj_beta, beta0, args=args)
    return opti
