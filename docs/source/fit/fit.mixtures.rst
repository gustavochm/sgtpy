Equilibrium data
================

SGTPy includes functions to the interaction parameters for binary mixtures, the fitted parameters by pairs can be used then with mixtures containing more components.
Depending on the mixture there might be more of one type of phase equilibria. Supposing that the parameters model are represented by :math:`\underline{\xi}`, the SGTPy functions will fit the given experimental data with the followings objectives functions:

If there is Vapor-Liquid Equilibria (VLE):

.. math::
	FO_{VLE}(\underline{\xi}) = \sum_{j=1}^{Np} \left[ \sum_{i=1}^c (y_{i,j}^{cal} - y_{i,j}^{exp})^2 + \left( \frac{P_{j}^{cal}}{P_{j}^{exp}} - 1\right)^2 \right]

If there is Liquid-Liquid Equilibria (LLE):

.. math::
	FO_{LLE}(\underline{\xi}) = \sum_{j=1}^{Np} \sum_{i=1}^c  \left[(x_{i,j} - x_{i,j}^{exp})^2 +  (w_{i,j} - w_{i,j}^{exp})^2 \right]

If there is Vapor-Liquid-Liquid Equilibria (VLLE):

.. math::
	FO_{VLLE}(\underline{\xi}) = \sum_{j=1}^{Np} \left[ \sum_{i=1}^c  \left[ (x_{i,j}^{cal} - x_{i,j}^{exp})^2 +  (w_{i,j}^{cal} - w_{i,j}^{exp})^2 +  (y_{i,j}^{cal} - y_{i,j}^{exp})^2 \right] + \left( \frac{P_{j}^{cal}}{P_{j}^{exp}} - 1\right)^2 \right]

If there is more than one type of phase equilibria, SGTPy will sum the errors of each one.

As an example the parameters for the system of hexane and ethanol will be fitted, first, the experimental data has to be loaded and then set up as a tuple as if shown in the following code block.

>>> import numpy as np
>>> # Experimental temperature saturation in K
>>> Texp = np.array([351.45, 349.15, 346.35, 340.55, 339.05, 334.95, 332.55, 331.85,
...       331.5 , 331.25, 331.15, 331.4 , 331.6 , 332.3 , 333.35, 336.65,
...       339.85, 341.85])
>>> # Experimental pressure in Pa
>>> Pexp = np.array([101330., 101330., 101330., 101330., 101330., 101330., 101330.,
...       101330., 101330., 101330., 101330., 101330., 101330., 101330.,
...       101330., 101330., 101330., 101330.])
>>> # Experimental liquid composition
>>> Xexp = np.array([[0.   , 0.01 , 0.02 , 0.06 , 0.08 , 0.152, 0.245, 0.333, 0.452,
...        0.588, 0.67 , 0.725, 0.765, 0.898, 0.955, 0.99 , 0.994, 1.   ],
....       [1.   , 0.99 , 0.98 , 0.94 , 0.92 , 0.848, 0.755, 0.667, 0.548,
...        0.412, 0.33 , 0.275, 0.235, 0.102, 0.045, 0.01 , 0.006, 0.   ]])
>>> # Experimental vapor composition
>>> Yexp = np.array([[0.   , 0.095, 0.193, 0.365, 0.42 , 0.532, 0.605, 0.63 , 0.64 ,
...        0.65 , 0.66 , 0.67 , 0.675, 0.71 , 0.745, 0.84 , 0.935, 1.   ],
...       [1.   , 0.905, 0.807, 0.635, 0.58 , 0.468, 0.395, 0.37 , 0.36 ,
...        0.35 , 0.34 , 0.33 , 0.325, 0.29 , 0.255, 0.16 , 0.065, 0.   ]])
>>> datavle = (Xexp, Yexp, Texp, Pexp)

If the system exhibits any other type phase equilibria the necessary tuples would have the following form:

>>> datalle = (Xexp, Wexp, Texp, Pexp)
>>> datavlle = (Xexp, Wexp, Yexp, Texp, Pexp)

Here, ``Xexp``, ``Wexp`` and ``Yexp`` are experimental mole fractions for liquid, liquid and vapor phase, respectively. ``Texp`` and ``Pexp`` are experimental temperature and pressure, respectively.

Fitting :math:`k_{ij}`
----------------------

For the mixture of hexane and ethanol, the hexane is modeled as a non-associating fluid and the ethanol is modeled as a self-associating fluid.
As there is no cross-association only the :math:`k_{ij}` binary correction can be optimized. First, the mixture is created.

>>> from SGTPy import component
>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                    lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                    rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> hexane = component('hexane', ms = 1.96720036, sigma = 4.54762477, eps = 377.60127994,
...                   lambda_r = 18.41193194, cii = 3.581510586936205e-19)
>>>mix = mixture(hexane, ethanol)

As a scalar is been fitted, SciPy recommends to give a certain interval where the minimum could be found, the function ``fit_kij`` handles this optimization as follows:


>>> from SGTPy.fit import fit_kij
>>> # bounds for kij
>>> kij_bounds = (-0.01, 0.01)
>>> fit_kij(kij_bounds, mix, datavle = datavle)
>>> # x: 0.011818496388350879

.. automodule:: SGTPy.fit.fitbinary
    :members: fit_kij
    :undoc-members:
    :show-inheritance:
    :noindex:


Fitting :math:`k_{ij}` and :math:`l_{ij}`
-----------------------------------------

There are other mixtures where there is cross-association, as in the mixture of water and ethanol. In this case both :math:`k_{ij}` and :math:`l_{ij}` can be optimized. First, the mixture is created.

>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6.,  eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515455e-20)
>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                    lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                    rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> mix = mixture(ethanol, water)

The parameters are optimized simultaneously using the ``fit_asso`` function.

>>> from SGTPy.fit import fit_asso
>>> #initial guess for kij and lij
>>> x0 = np.array([0, 0])
>>> min_options = {'method':'Nelder-Mead'}
>>> fit_asso(x0, mix, datavle = datavle, minimize_options=min_options)

.. automodule:: SGTPy.fit.fitbinary
    :members: fit_asso
    :undoc-members:
    :show-inheritance:
    :noindex:


Fitting :math:`k_{ij}` and :math:`r_{c}^{ABij}`
-----------------------------------------------

Finally, in the case of mixtures that exhibit induced association, both :math:`k_{ij}` and :math:`r_{c}^{ABij}` should be optimized.
This optimization is illustrated for a mixture of ethanol and CPME.

>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                    lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                    rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> cpme = component('cpme', ms =  2.32521144, sigma = 4.13606074, eps = 343.91193798, lambda_r = 14.15484877,
...                 lambda_a = 6.0, npol = 1.91990385,mupol = 1.27, sites =[0,0,1], cii = 3.5213681817448466e-19)
>>> mix = mixture(ethanol, cpme)

The parameters are optimized simultaneously using the ``fit_cross`` function.

>>> from SGTPy.fit import fit_cross
>>> #initial guesses for kij and rcij
>>> x0 = [0.01015194, 2.23153033]
>>> fit_cross(x0, mix, assoc=0, datavle=datavle)

.. automodule:: SGTPy.fit.fitbinary
    :members: fit_cross
    :undoc-members:
    :show-inheritance:
    :noindex:
