import numpy as np


# Equation 28
def I_lam(x0, lam):
    lam3 = 3. - lam
    I = (x0**lam3 - 1.) / lam3
    return I

# Equation 29
def J_lam(x0, lam):
    lam3 = 3. - lam
    lam4 = 4. - lam
    J = (lam3*x0**lam4 - lam4*x0**lam3 + 1.)/(lam3 * lam4)
    return J


# Equation 16
def kHS(eta):
    num = (1-eta)**4
    den = 1 + 4*eta+4*eta**2-4*eta**3 + eta**4
    return num/den


def dkHS(eta):
    den = 1 + 4*eta+4*eta**2-4*eta**3 + eta**4
    num1 = (1-eta)**4
    num2 = -4*(-1 + eta)**3*(-2 + (-5 + eta)*eta)

    khs = num1/den
    dkhs = num2/den**2

    return khs, dkhs


def d2kHS(eta):
    den = 1 + 4*eta+4*eta**2-4*eta**3 + eta**4
    num1 = (1-eta)**4
    num2 = -4*(-1 + eta)**3*(-2 + (-5 + eta)*eta)

    num3 = 4*(-1 + eta)**2*(17 + eta*(82 + eta*(39 +
           eta*(-80 + eta*(77 + 3*(-10 + eta)*eta)))))
    khs = num1/den
    dkhs = num2/den**2
    d2khs = num3/den**3
    return khs, dkhs, d2khs


def d3kHS(eta):
    den = 1 + 4*eta+4*eta**2-4*eta**3 + eta**4

    num1 = (1-eta)**4
    num2 = -4*(-1 + eta)**3*(-2 + (-5 + eta)*eta)

    num3 = 4*(-1 + eta)**2*(17 + eta*(82 + eta*(39 +
           eta*(-80 + eta*(77 + 3*(-10 + eta)*eta)))))

    num4 = -624. - 4032. * eta - 576. * eta**2 + 16656. * eta**3-11424.*eta**4
    num4 += -16896. * eta**5 + 34752. * eta**6 - 27936. * eta**7+13776.*eta**8
    num4 += - 4416. * eta**9 + 768. * eta**10 - 48. * eta**11
    khs = num1/den
    dkhs = num2/den**2
    d2khs = num3/den**3
    d3khs = num4/den**4
    return khs, dkhs, d2khs, d3khs


def eta_eff(eta, ci):
    eta_vec = np.array([eta, eta**2, eta**3, eta**4])
    neff = np.dot(ci, eta_vec)
    return neff


def deta_eff(eta, ci):

    eta_vec = np.array([[eta, eta**2, eta**3, eta**4],
                       [1., 2*eta, 3*eta**2, 4*eta**3]])

    dneff = np.matmul(eta_vec, ci)
    return dneff


def d2eta_eff(eta, ci):

    eta_vec = np.array([[eta, eta**2, eta**3, eta**4],
                       [1., 2*eta, 3*eta**2, 4*eta**3],
                       [0., 2., 6.*eta, 12.*eta**2]])
    d2neff = np.matmul(eta_vec, ci)

    return d2neff


def d3eta_eff(eta, ci):
    eta_vec = np.array([[eta, eta**2, eta**3, eta**4],
                       [1., 2*eta, 3*eta**2, 4*eta**3],
                       [0., 2., 6.*eta, 12.*eta**2],
                       [0., 0., 6., 24.*eta]])

    d3neff = np.matmul(eta_vec, ci)

    return d3neff


# Equation 17
def Xi(x03, nsigma, f1, f2, f3):
    x = f1*nsigma + f2*nsigma**5 + f3*nsigma**8
    return x


def dXi(x03, nsigma, f1, f2, f3):
    x = f1*nsigma + f2*nsigma**5 + f3*nsigma**8
    dx = x03*(f1 + 5*f2 * nsigma**4 + 8*f3*nsigma**7)
    return x, dx


def d2Xi(x03, nsigma, f1, f2, f3):
    x = f1*nsigma + f2*nsigma**5 + f3*nsigma**8
    dx = x03*(f1 + 5*f2 * nsigma**4 + 8*f3*nsigma**7)
    d2x = x03**2*(20.*f2 * nsigma**3 + 56.*f3*nsigma**6)
    return x, dx, d2x
