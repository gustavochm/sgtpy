from __future__ import division, print_function, absolute_import
from .vrmie_mixtures.saftvrmiemix import saftvrmie_mix
from .vrmie_pure.saftvrmie import saftvrmie_pure
from .gammamie_mixtures.saftgammamie_mixture import saftgammamie_mix
from .gammamie_pure.saftgammamie_pure import saftgammamie_pure


def saftvrmie(mix_or_component, compute_critical=True):
    '''
    Returns SAFT-VR-Mie EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`SGTPy.mixture` or :class:`SGTPy.component` object
    compute_critical: bool
        If True the critical point of the fluid will attempt to be computed
        (it might fail for some fluids).
    Returns
    -------
    eos : object
        SAFT-VR-Mie EoS object
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = saftvrmie_pure(mix_or_component, compute_critical)
    else:
        eos = saftvrmie_mix(mix_or_component, compute_critical)
    return eos


def saftgammamie(mix_or_component, compute_critical=True):
    '''
    Returns SAFT-Gamma-Mie EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`saftgammamie.mixture` or :class:`saftgammamie.component` object
    compute_critical: bool
        If True the critical point of the fluid will attempt to be computed
        (it might fail for some fluids).
    Returns
    -------
    eos : object
        SAFT-Gamma-Mie EoS object
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = saftgammamie_pure(mix_or_component, compute_critical)
    else:
        eos = saftgammamie_mix(mix_or_component, compute_critical)
    return eos
