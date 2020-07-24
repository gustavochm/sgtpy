from __future__ import division, print_function, absolute_import
from .mixtures.saftvrmiemix import saftvrmie_mix
from .pure.saftvrmie import saftvrmie_pure


def saftvrmie(mix_or_component):
    nc = mix_or_component.nc
    if nc == 1:
        eos = saftvrmie_pure(mix_or_component)
    else:
        eos = saftvrmie_mix(mix_or_component)
    return eos
