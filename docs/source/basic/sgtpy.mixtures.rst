SGTPy.mixture
=============

:class:`SGTPy.mixture` object stores both pure component and mixture related information and interaction parameters needed for equilibria and interfacial properties computation.


Two pure components are required to create a base mixture:

>>> from SGTPy import component, mixture
>>> ethanol = component('ethanol2C', ms=1.7728, sigma=3.5592 , eps=224.50,
...                    lambda_r=11.319, lambda_a=6., eAB = 3018.05, rcAB=0.3547,
...                    rdAB=0.4, sites=[1,0,1], cii=5.3141080872882285e-20)
>>> hexane = component('hexane', ms=1.96720036, sigma=4.54762477, eps=377.60127994,
...                    lambda_r=18.41193194, cii=3.581510586936205e-19 )
>>> # creating mixture
>>> mix = mixture(hexane, ethanol)

Additional components can be added to the mixture with the :func:`SGTPy.mixture.add_component` method.

>>> cpme = component('cpme', ms=2.32521144, sigma=4.13606074, eps=343.91193798, lambda_r=14.15484877,
...                  lambda_a=6.0, npol=1.91990385, mupol=1.27, sites=[0,0,1],
...                  cii=3.5213681817448466e-19)
>>> # adding a component
>>> mix.add_component(cpme)


Once all components have been added to the mixture, the interaction parameters must be supplied.
SGTPy considers :math:`k_{ij}`  binary correction for the Mie potential interaction energy and :math:`l_{ij}` for the cross-association energy correction.


.. math::
   \epsilon_{ij} = (1-k_{ij}) \frac{\sqrt{\sigma_i^3 \sigma_j^3}}{\sigma_{ij}^3} \sqrt{\epsilon_i \epsilon_j}


.. math::
   \epsilon_{ij}^{AB} = (1 - l_{ij})\sqrt{\epsilon_{ii}^{AB} \epsilon_{jj}^{AB}}

The method :func:`SGTPy.mixture.kij_saft` sets the binary correction :math:`k_{ij}` for the Mie potential interaction energy.

>>> from SGTPy import component, mixture
>>> methane = component('methane', ms = 1.0, sigma = 3.752 , eps = 170.75,
...                    lambda_r = 16.39, lambda_a = 6.)
>>> dodecane = component('dodecane', ms = 4.0, sigma = 4.351 , eps = 378.56,
...                    lambda_r = 18.41, lambda_a = 6.)
>>> mix = mixture(methane, dodecane)
>>> # interaction parameter optimized from phase equilibria data
>>> kij = -0.02199102576365056
>>> Kij = np.array([[0, kij], [kij, 0]])
>>> # setting kij correction
>>> mix.kij_saft(Kij)


The method :func:`SGTPy.mixture.lij_saft` sets the binary correction :math:`l_{ij}` for the cross-association energy.


>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515458e-20)
>>> butanol = component('butanol2C', ms = 1.9651, sigma = 4.1077 , eps = 277.892,
...                    lambda_r = 10.6689, lambda_a = 6., eAB = 3300.0, rcAB = 0.2615,
...                    rdAB = 0.4, sites = [1,0,1], npol = 1.45, mupol = 1.6609,
...                    cii  = 1.5018715324070352e-19)
>>> # optimized from experimental LLE
>>> kij, lij = np.array([-0.00736075, -0.00737153])
>>> Kij = np.array([[0, kij], [kij, 0]])
>>> Lij = np.array([[0, lij], [lij, 0]])
>>> # setting interactions corrections
>>> mix.kij_saft(Kij)
>>> mix.lij_saft(Lij)





.. warning:: User is required to supply the necessary parameters for methods


.. autoclass:: SGTPy.mixture
    :members:
