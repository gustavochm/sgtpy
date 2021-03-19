

# Equation 64
def g1sigma(rho, suma_g1, da1m_deta, deta_drho, cte_g1s):

    da1 = da1m_deta * deta_drho
    g1 = cte_g1s * (3.*da1 - suma_g1/rho)

    return g1


def dg1sigma_drho(rho, suma_g1, d2a1m_drho, cte_g1s):
    dg1 = 3.*d2a1m_drho - suma_g1/rho
    dg1[1] += suma_g1[0]/rho**2
    dg1 *= cte_g1s

    return dg1


def d2g1sigma_drho(rho, suma_g1, d3a1m_drho, cte_g1s):

    rho2 = rho**2
    rho3 = rho2*rho
    d2g1 = 3*d3a1m_drho - suma_g1/rho
    d2g1[1] += suma_g1[0]/rho2
    d2g1[2] += -2*suma_g1[0]/rho3 + 2*suma_g1[1]/rho2
    d2g1 *= cte_g1s

    return d2g1
