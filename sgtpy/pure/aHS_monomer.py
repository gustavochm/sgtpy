import numpy as np


# Hard sphere Eq 11
def ahs(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    return a


def dahs_deta(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    da = 2*(-2+eta)/(-1+eta)**3
    return np.array([a, da])


def d2ahs_deta(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    da = 2*(-2+eta)/(-1+eta)**3
    d2a = (10-4*eta)/(-1+eta)**4
    return np.array([a, da, d2a])
