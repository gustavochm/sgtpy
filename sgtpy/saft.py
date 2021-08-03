from __future__ import division, print_function, absolute_import
from .vrmie_mixtures.saftvrmiemix import saftvrmie_mix
from .vrmie_pure.saftvrmie import saftvrmie_pure
from .gammamie_mixtures.saftgammamie_mixture import saftgammamie_mix
from .gammamie_pure.saftgammamie_pure import saftgammamie_pure


def saftvrmie(mix_or_component):
    '''
    Returns SAFT-VR-Mie EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`SGTPy.mixture` or :class:`SGTPy.component` object
    Returns
    -------
    eos : object
        SAFT-VR-Mie EoS object
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = saftvrmie_pure(mix_or_component)
    else:
        eos = saftvrmie_mix(mix_or_component)
    return eos




def saftgammamie(mix_or_component):
    '''
    Returns SAFT-Gamma-Mie EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`saftgammamie.mixture` or :class:`saftgammamie.component` object
    Returns
    -------
    eos : object
        SAFT-Gamma-Mie EoS object
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = saftgammamie_pure(mix_or_component)
    else:
        eos = saftgammamie_mix(mix_or_component)
    return eos
