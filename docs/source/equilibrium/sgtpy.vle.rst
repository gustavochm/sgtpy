Vapor liquid equilibrium
=========================

The two-phase flash can be used to compute vapor-liquid equilibria at fixed temperature and pressure. When dealing with saturated liquids or vapor four types of problems arise, solution methods and routines are related to a simplification of the Radford-Rice mass balance, as the phase fraction is already known.

Another option for these situations is to solve the following system of equations:

.. math::

	f_i &= \ln K_i + \ln \hat{\phi}_i^v(\underline{y}, T, P) -\ln \hat{\phi}_i^l(\underline{x}, T, P) \quad i = 1,...,c \\
	f_{c+1} &= \sum_{i=1}^c (y_i-x_i)

bubble points
#############

In this case, a saturated liquid of known composition is forming a differential size bubble. If the pressure is specified, the temperature must be found. Similarly, when the temperature is specified, equilibrium pressure has to be calculated.

The usual approach for solving this problem consists of a combined quasi-Newton for solving for temperature or pressure and successive substitution for composition with the following simplification of the Radford-Rice equation:

.. math::

	FO = \sum_{i=1}^c x_i (K_i-1) = \sum_{i=1}^c y_i -1 = 0

In the case of having a good initial value of the true equilibrium values, the full multidimensional system of equations can be solved.


In the following code block and example from this computation, it is shown.

>>> from SGTPy import component, mixture, saftvrmie
>>> from SGTPy.equilibrium import bubbleTy
>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                      lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                      rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> hexane = component('hexane', ms = 1.96720036, sigma = 4.54762477, eps = 377.60127994,
...                     lambda_r = 18.41193194, cii = 3.581510586936205e-19)
>>> mix = mixture(hexane, ethanol)
>>> # fitted to experimental data
>>> kij = 0.011818492037463553
>>> Kij = np.array([[0, kij], [kij, 0]])
>>> mix.kij_saft(Kij)
>>> eos = saftvrmie(mix)
>>> # bubble point conditions
>>> P = 1.01325e5  # Pa
>>> x = np.array([0.2, 0.8])
>>> # initial guess for temperature and vapor composition
>>> T0 = 320.
>>> y0 = np.array([0.8, 0.2])
>>> bubbleTy(y0, T0, x, P, eos)
>>> # vapor composition and equilibrium temperature
... (array([0.58026242, 0.41973758]), 333.45268641828727)

.. automodule:: SGTPy.equilibrium.bubble
    :members: bubbleTy
    :undoc-members:
    :show-inheritance:
    :noindex:



In the following case, a saturated liquid of known composition and temperature is forming a differential size bubble. We need to find the composition and equilibrium pressure.

>>> from SGTPy.equilibrium import bubblePy
>>> # bubble point conditions
>>> T = 350.  # K
>>> x = np.array([0.2, 0.8])
>>> # initial guess for tempetature and vapor composition
>>> P0 = 1e5 # Pa
>>> y0 = np.array([0.8, 0.2])
>>> bubblePy(y0, P0, x, T, eos)
>>> vapor composition and equilibrium pressure
... (array([0.52007469, 0.47992531]), 178461.90299494946)

.. automodule:: SGTPy.equilibrium.bubble
    :members: bubblePy
    :undoc-members:
    :show-inheritance:
    :noindex:


dew points
##########

In this case, a saturated vapor of known composition and temperature is forming a differential size dew. We need to find the composition and equilibrium pressure.

The usual approach for solving this problem consists of a combined quasi-Newton for solving for Pressure and successive substitution for composition with the following simplification of the Radford-Rice equation:

.. math::
	FO = 1 - \sum_{i=1}^c \frac{y_i}{K_i} = 1 - \sum_{i=1}^c x_i = 0

In the case of having a good initial value of the true equilibrium values, a full multidimensional system can be solved.

In the following code block and example from this computation, it is shown for composition and equilibrium pressure.


>>> from SGTPy.equilibrium import dewPx
>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                    lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                    rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> cpme = component('cpme', ms =  2.32521144, sigma = 4.13606074, eps = 343.91193798, lambda_r = 14.15484877,
...                 lambda_a = 6.0, npol = 1.91990385,mupol = 1.27, sites =[0,0,1], cii = 3.5213681817448466e-19)
>>> mix = mixture(ethanol, cpme)
>>> kij = 0.01015194
>> Kij = np.array([[0, kij], [kij, 0]])
>>> mix.kij_saft(Kij)
>>> eos = saftvrmie(mix)
>>> # induced association set up
>>> rc = 2.23153033 # Angstrom
>>> eos.eABij[0,1] = ethanol.eAB / 2
>>> eos.eABij[1,0] = ethanol.eAB / 2
>>> eos.rcij[0,1] = rc * 1e-10
>>> eos.rcij[1,0] = rc * 1e-10
>>> # dew point conditions
>>> T = 350. # K
>>> y = np.array([0.4, 0.6])
>>> # initial guess for temperature and liquid composition
>>> P0 = 1e5 # Pa
>>> x0 = np.array([0.2, 0.8])
>>> dewPx(x0, P0, y, T, eos)
>>> # vapor composition and equilibrium pressure
... (array([0.10431595, 0.89568405]), 62927.01280107427)

.. automodule:: SGTPy.equilibrium.dew
    :members: dewPx
    :undoc-members:
    :show-inheritance:
    :noindex:


Similarly, the calculation can be carried out for equilibria composition and temperature:

>>> from SGTPy.equilibrium import dewTx
>>> # dew point conditions
>>> P = 1.01325e5  # Pa
>>> y = np.array([0.4, 0.6])
>>> # initial guess for temperature and liquid composition
>>> T0 = 350.
>>> x0 = np.array([0.2, 0.8])
>>> dewTx(x0, T0, y, P, eos)
>>> # vapor composition and equilibrium temperature
... (array([0.10611092, 0.89388908]), 364.3596395673508)


.. automodule:: SGTPy.equilibrium.dew
    :members: dewTx
    :undoc-members:
    :show-inheritance:
    :noindex:
