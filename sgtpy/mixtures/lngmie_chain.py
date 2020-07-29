import numpy as np


def lngmie(ghs, g1, g2, beta, eps):
    be = beta * eps
    lng = np.log(ghs) + (be*g1+be**2*g2)/ghs
    return lng


def dlngmie_dxhi00(ghs, g1, g2, beta, eps):
    be = beta * eps
    ghs, dghs = ghs

    g1, dg1 = g1
    g2, dg2 = g2

    lng = np.log(ghs) + (be*g1+be**2*g2)/ghs

    dlng = be * ghs * (dg1 + be * dg2)
    dlng += dghs * (ghs - be * (g1 + be * g2))
    dlng /= ghs**2

    return np.array([lng, dlng])


def d2lngmie_dxhi00(ghs, g1, g2, beta, eps):
    be = beta * eps

    ghs, dghs, d2ghs = ghs
    g1, dg1, d2g1 = g1
    g2, dg2, d2g2 = g2

    lng = np.log(ghs) + (be*g1+be**2*g2)/ghs

    dlng = be * ghs * (dg1 + be * dg2)
    dlng += dghs * (ghs - be * (g1 + be * g2))
    dlng /= ghs**2

    d2lng = 2*be * dghs**2*(g1+be*g2)
    d2lng += ghs**2*(d2ghs + be * (d2g1+be*d2g2))
    d2lng += ghs * (-dghs*(2*be*(dg1+be*dg2)+dghs)-be*(g1+be*g2)*d2ghs)
    d2lng /= ghs**3
    return np.array([lng, dlng, d2lng])


def dlngmie_dx(ghs, g1, g2, dghsx, dg1x, dg2x, beta, eps):
    be = beta * eps

    aux = be*g1+be**2*g2
    lng = np.log(ghs) + aux/ghs

    dlng = be * ghs * (dg1x + be * dg2x)
    dlng += dghsx * (ghs - aux)
    dlng /= ghs**2

    return lng, dlng


def dlngmie_dxxhi(ghs, g1, g2, dghsx, dg1x, dg2x, beta, eps):
    be = beta * eps
    ghs, dghs = ghs

    g1, dg1 = g1
    g2, dg2 = g2

    aux = be*g1+be**2*g2
    lng = np.log(ghs) + aux/ghs

    dlng = be * ghs * (dg1 + be * dg2)
    dlng += dghs * (ghs - aux)
    dlng /= ghs**2

    dlngx = be * ghs * (dg1x + be * dg2x)
    dlngx += dghsx * (ghs - aux)
    dlngx /= ghs**2

    return np.array([lng, dlng]), dlngx
