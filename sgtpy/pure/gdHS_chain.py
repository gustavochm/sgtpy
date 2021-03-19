import numpy as np


# Eq 47 to 50

def ki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)

    ks = np.array([k0, k1, k2, k3])
    return ks


def dki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)

    dk0 = (24 + eta*(-6+eta*(3-7*eta+eta**2)))/(eta-1)**4/3
    dk1 = - (12 + eta*(2+eta)*(6-6*eta+eta2))/(eta-1)**4/2
    dk2 = 3*eta/(4*(-1+eta)**3)
    dk3 = (3+eta*(12+eta*(eta-3)*(eta-1)))/(6*(-1+eta)**4)
    ks = np.array([[k0, k1, k2, k3],
                    [dk0, dk1, dk2, dk3]])
    return ks


def d2ki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)

    dk0 = (24 + eta*(-6+eta*(3-7*eta+eta**2)))/(eta-1)**4/3
    dk1 = - (12 + eta*(2+eta)*(6-6*eta+eta2))/(eta-1)**4/2
    dk2 = 3*eta/(4*(-1+eta)**3)
    dk3 = (3+eta*(12+eta*(eta-3)*(eta-1)))/(6*(-1+eta)**4)

    d2k0 = (-30 + eta*(1+eta)*(4+eta))/(eta-1)**5
    d2k1 = 6*(5-2*eta*(eta-1))/(eta-1)**5
    d2k2 = -3*(1+2*eta)/(4*(eta-1)**4)
    d2k3 = (-4+eta*(eta-7))/((-1+eta)**5)

    ks = np.array([[k0, k1, k2, k3],
                  [dk0, dk1, dk2, dk3],
                  [d2k0, d2k1, d2k2, d2k3]])
    return ks


# Eq 46
def gdHS(x0_vector, eta):
    ks = ki_chain(eta)
    # xs = np.array([1, x0, x0**2, x0**3])
    g = np.exp(np.dot(x0_vector, ks))
    return g


def dgdHS_drho(x0_vector, eta, deta_drho):
    dks = dki_chain(eta)
    # xs = np.array([1., x0, x0**2, x0**3])
    dg = np.matmul(dks, x0_vector)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
    dg *= deta_drho[:2]
    return dg


def d2gdHS_drho(x0_vector, eta, deta_drho):
    d2ks = d2ki_chain(eta)
    # xs = np.array([1., x0, x0**2, x0**3])
    d2g = np.matmul(d2ks, x0_vector)
    d2g[0] = np.exp(d2g[0])
    d2g[2] += d2g[1]**2
    d2g[2] *= d2g[0]
    d2g[1] *= d2g[0]
    d2g *= deta_drho[:3]
    return d2g
