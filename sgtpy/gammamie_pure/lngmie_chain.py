from __future__ import division, print_function, absolute_import
import numpy as np


def lngmie(ghs, g1, g2, be, be2):
    # be = beta * eps
    lng = np.log(ghs) + (be*g1+be2*g2)/ghs
    return lng


def dlngmie_dxhi00(ghs, g1, g2, be, be2):
    # be = beta * eps
    ghs, dghs = ghs

    g1, dg1 = g1
    g2, dg2 = g2

    aux = (be*g1+be2*g2)

    lng = np.log(ghs) + aux/ghs

    dlng = be * ghs * (dg1 + be * dg2)
    dlng += dghs * (ghs - aux)
    dlng /= ghs**2

    return np.array([lng, dlng])


def d2lngmie_dxhi00(ghs, g1, g2, be, be2):
    # be = beta * eps

    ghs, dghs, d2ghs = ghs
    g1, dg1, d2g1 = g1
    g2, dg2, d2g2 = g2

    # be2 = be**2
    ghs2 = ghs**2
    ghs3 = ghs2 * ghs
    dghs2 = dghs**2
    aux1 = be*g1+be2*g2
    aux2 = be*dg1+be2*dg2

    lng = np.log(ghs) + aux1/ghs

    dlng = ghs * aux2
    dlng += dghs * (ghs - aux1)
    dlng /= ghs2

    d2lng = 2*dghs2*aux1
    d2lng += ghs2*(d2ghs + be * (d2g1+be*d2g2))
    d2lng += ghs*(-2*dghs*aux2-dghs2 - aux1*d2ghs)
    d2lng /= ghs3

    return np.array([lng, dlng, d2lng])
