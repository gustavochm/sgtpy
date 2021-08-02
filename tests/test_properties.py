import pytest
import numpy as np
from SGTPy import mixture, component
from SGTPy import saftvrmie


def setup(T=298.15):
    water = component(
        "water",
        ms=1.7311,
        sigma=2.4539,
        eps=110.85,
        lambda_r=8.308,
        lambda_a=6.0,
        eAB=1991.07,
        rcAB=0.5624,
        rdAB=0.4,
        sites=[0, 2, 2],
        cii=1.5371939421515458e-20,
    )

    octane = component(
        "octane", ms=3.0, sigma=4.227, eps=333.7, lambda_r=16.14, lambda_a=6.0
    )
    mix = mixture(water, octane)
    z = (-3.5450933661557516e-06, 2.6819855363061289e-03, -4.3239556892743974e-01)
    p = np.poly1d(z)
    # To be optimized from experimental LLE
    kij = p(T)

    Kij = np.array([[0.0, kij], [kij, 0.0]])

    # setting interactions corrections
    mix.kij_saft(Kij)

    eos = saftvrmie(mix)
    return eos


class TestGetLNGamma:
    def test_binary(self):
        mix = setup()

        _ = mix.get_lngamma(0.1, 298.15, 101325)
