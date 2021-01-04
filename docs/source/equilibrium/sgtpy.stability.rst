Stability
=========

Stability analysis plays a fundamental role during phase equilibria computation. Most stability tests are based on the fact that a consistent equilibrium must minimize the Gibbs free energy of the system at a given temperature and pressure. Within this idea Michelsen proposed the tangent plane distance function which allows testing the relative stability of a mixture at given composition (:math:`z`), temperature (:math:`T`),  and pressure(:math:`P`).

.. math::

	tpd(w) =  \sum_{i=1}^c w_i \left[\ln w_i +  \ln \hat{\phi}_i(w) - \ln z_i - \ln \hat{\phi}_i(z) \right]

The tpd function is evaluated for a trial composition (:math:`w`) and if the tpd takes a negative value it implies that the energy of the system decreased with the formation of the new phase, i.e. the original phase was unstable. In order to test the stability of a mixture, the usual method is to find a minimum of the function a verify the sign of the tpd function at the minimum. Minimization recommendations for this purpose were given by Michelsen and they are included in Phasepy's implementation.

Minimization of tpd function
----------------------------

As this is an iterative process, in order to find a minimum an initial guess of it has to be supplied. In the following code block, the stability of a liquid mixture is tested against the formation of another liquid.

>>> import numpy as np
>>> from SGTPy import component, mixture, saftvrmie
>>> from SGTPy.equilibrium import lle
>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515458e-20)
>>> butanol = component('butanol2C', ms = 1.9651, sigma = 4.1077 , eps = 277.892,
...                     lambda_r = 10.6689, lambda_a = 6., eAB = 3300.0, rcAB = 0.2615,
...                     rdAB = 0.4, sites = [1,0,1], npol = 1.45, mupol = 1.6609,
...                     cii  = 1.5018715324070352e-19)
>>> mix = mixture(water, butanol)
>>> # optimized from experimental LLE
>>> kij, lij = np.array([-0.00736075, -0.00737153])
>>> Kij = np.array([[0, kij], [kij, 0]])
>>> Lij = np.array([[0., lij], [lij, 0]])
>>> # setting interactions corrections
>>> mix.kij_saft(Kij)
>>> mix.lij_saft(Lij)
>>> # creating eos model
>>> eos = saftvrmie(mix)
>>> T = 320 # K
>>> P = 1.01e5 # Pa
>>> z = np.array([0.8, 0.2])
>>> #Search for trial phase
>>> w = np.array([0.99, 0.01])
>>> tpd_min(w, z, T, P, eos, stateW = 'L', stateZ = 'L')
>>> #composition of minimum found and tpd value
... (array([0.95593129, 0.04406871]), -0.011057873031562693)

As the tpd value at the minimum is negative it means that the phase is unstable at will split into two liquids. Similarly, the relative stability can be tested against a vapor phase, in which case is found that the original phase was more stable than the vapor.

>>> w = np.array([0.99, 0.01])
>>> tpd_min(w, z, T, P, eos, stateW = 'V', stateZ = 'L')
>>> #composition of minimum found and tpd value
... #(array([0.82414873, 0.17585127]), 0.8662934867235452)

.. automodule:: SGTPy.equilibrium.stability
    :members: tpd_min
    :undoc-members:
    :show-inheritance:
    :noindex:



Finding all minimums
--------------------

Sometimes might be useful to find all the minimums of a given mixture, for that case phasepy will try to find them using different initial guesses until the number of requested minimums is found. In the next example, three minimums were requested.

>>> from SGTPy.equilibrium import tpd_minimas
>>> T = 320 # K
>>> P = 1.01e5 # Pa
>>> z = np.array([0.8, 0.2])
>>> nmin = 3
>>> tpd_minimas(nmin, z, T, P, eos, 'L', 'L')
>>> #composition of minimuns found and tpd values
...((array([0.95593125, 0.04406875]),   array([0.55571917, 0.44428083]),  array([0.95593125, 0.04406875])),
... array([-0.01105787, -0.01083626, -0.01105787]))

Similar to the first example, all the minimums in vapor phase can be found, in this case there only one minimum.

>>> tpd_minimas(nmin , z, T, P, model, 'V', 'L')
>>> #composition of minimuns found and tpd values
... ((array([0.82414939, 0.17585061]),  array([0.82414939, 0.17585061]),  array([0.82414939, 0.17585061])),
...  array([0.86629349, 0.86629349, 0.86629349]))


.. automodule:: SGTPy.equilibrium.stability
    :members: tpd_minimas
    :undoc-members:
    :show-inheritance:
    :noindex:


Liquid liquid equilibrium initiation
------------------------------------

Using the same principles stated above, tpd function can be used to generate initial guesses for liquid-liquid equilibria, the function ell_init allows to find two minimums of the mixture.

>>> from SGTPy.equilibrium import lle_init
>>> T = 320 # K
>>> P = 1.01e5 # Pa
>>> z = np.array([0.8, 0.2])
>>> lle_init(z, T, P, eos)
>>> #initial values for lle computation
... (array([0.95593125, 0.04406875]), array([0.55571917, 0.44428083]))


.. automodule:: SGTPy.equilibrium.stability
    :members: lle_init
    :undoc-members:
    :show-inheritance:
    :noindex:
