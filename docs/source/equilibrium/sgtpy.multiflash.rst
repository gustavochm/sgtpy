Multiphase Flash
================

Meta-stable solutions of the isofugacity method is an important concern when dealing with more than two liquid phases. Stability verification during the equilibria computation must be performed. In SGTPy liquid-liquid equilibria and vapor-liquid-liquid equilibria are solved similarly with a modified Radford-Rice mass balance system of equations that allows verifying the stability and equilibria of the phases simultaneously.

.. math::
	 \sum_{i=1}^c \frac{z_i (K_{ik} \exp{\theta_k}-1)}{1+ \sum\limits^{\pi}_{\substack{j=1 \\ j \neq r}}{\psi_j (K_{ij}} \exp{\theta_j} -1)} = 0 \qquad k = 1,..., \pi,  k \neq r

This system of equations was proposed by Gupta et al, and it is a modified Radford-Rice mass balance that introduces stability variables :math:`\theta`. This allows to solve the mass balance for phase fraction and stability variables and then update composition similar to a regular two-phase flash. The stability variable gives information about the phase if it takes a positive value the phase is unstable, on the other hand, if it is zero then the phase is stable. The algorithm of successive substitution and Newton method can be slow in some cases, in that situation the function will attempt to minimize the Gibbs free energy of the system.

.. math::
	 min \, {G} = \sum_{k=1}^\pi \sum_{i=1}^c F_{ik} \ln \hat{f}_{ik}


Liquid Liquid Equilibrium
#########################

The two-phase flash can be used for solving liquid-liquid equilibrium, but it is important to consider the stability of the phases. For that reason, an algorithm that can compute stability and equilibrium simultaneously was implemented in this package.


In the following code block and an example of how to solve this problem, it is shown.

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
>>> # LLE conditions
>>> T = 298.15 # K
>>> P = 1.01325e5 # Pa
>>> # global composition
>>> z = np.array([0.8, 0.2])
>>> # initial composition of the liquid phases
>>> x0 = np.array([0.9, 0.1])
>>> w0 = np.array([0.6, 0.4])
>>> # LLE is performed as a flash that search stable phases
>>> lle(x0, w0, z, T, P, eos, full_output=False)
... #(array([0.96022175, 0.03977825]), array([0.53375333, 0.46624667]), 0.37569428828046253)

.. automodule:: SGTPy.equilibrium.ell
    :members: lle
    :undoc-members:
    :show-inheritance:


Three phase equilibrium
#######################



Binary mixtures
***************

For degrees of freedom's restriction, a system of equations has to be solved for three phase equilibrium of binary mixtures. In the following code block, an example of how to do it is shown.

>>> from SGTPy.equilibrium import vlleb
>>> P = 1.01325e5 # Pa
>>> # initial guesses
>>> x0 = np.array([0.96, 0.06])
>>> w0 = np.array([0.53, 0.47])
>>> y0 = np.array([0.8, 0.2])
>>> T0 = 350. # K
>>> vlleb(x0, w0, y0, T0, P, 'P', eos,full_output=False)
... # (array([0.94267074, 0.05732926]), array([0.61296229, 0.38703771]),
... # array([0.77529982, 0.22470018]), array([367.2045658]))

.. automodule:: SGTPy.equilibrium.hazb
    :members: vlleb
    :undoc-members:
    :show-inheritance:



Multicomponent mixtures
***********************

When working with multicomponent mixtures (3 or more) a multiflash has to be performed in order to compute three-phase equilibrium. This algorithm ensures that stable phases are computed.

>>> import numpy as np
>>> from SGTPy import component, mixture, saftvrmie
>>> from SGTPy.equilibrium import vlle
>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6.,  eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515455e-20)
>>> butanol = component('butanol2C', ms = 1.9651, sigma = 4.1077 , eps = 277.892,
...                      lambda_r = 10.6689, lambda_a = 6., eAB = 3300.0, rcAB = 0.2615,
...                      rdAB = 0.4, sites = [1,0,1], npol = 1.45, mupol = 1.6609,
...                      cii  = 1.5018715324070352e-19)
>>> mtbe = component('mtbe', ms =2.17847383,  sigma=  4.19140014, eps =  306.52083841,
...                   lambda_r = 14.74135198, lambda_a = 6.0, npol = 2.95094686,
...                   mupol = 1.3611, sites = [0,0,1], cii =3.5779968517655445e-19 )
>>> mix = mixture(water, butanol)
>>> mix.add_component(mtbe)
>>> #butanol water
>>> k12, l12 = np.array([-0.00736075, -0.00737153])
>>> # mtbe butanol
>>> k23 = -0.0029995
>>> l23 = 0.
>>> rc23 =  1.90982649
>>> # mtbe water
>>> k13 = -0.07331438
>>> l13 = 0.
>>> rc13 = 2.84367922
>>> # setting up interaction corrections
>>> Kij = np.array([[0., k12, k13], [k12, 0., k23], [k13, k23, 0.]])
>>> Lij = np.array([[0., l12, l13], [l12, 0., l23], [l13, l23, 0.]])
>>> mix.kij_saft(Kij)
>>> mix.lij_saft(Lij)
>>> eos = saftvrmie(mix)
>>> # setting up induced association
>>> # mtbe water
>>> eos.eABij[0,2] = water.eAB / 2
>>> eos.eABij[2,0] = water.eAB / 2
>>> eos.rcij[0,2] = rc13 * 1e-10
>>> eos.rcij[2,0] = rc13 * 1e-10
>>> # mtbe butanol
>>> eos.eABij[2,1] = butanol.eAB / 2
>>> eos.eABij[1,2] = butanol.eAB / 2
>>> eos.rcij[2,1] = rc23 * 1e-10
>>> eos.rcij[1,2] = rc23 * 1e-10
>>> # three phase equilibria conditions
>>> T = 345. #K
>>> P = 1.01325e5 # Pa
>>> # global composition
>>> z = np.array([0.5, 0.3, 0.2])
>>> # initial guesses
>>> x0 = np.array([0.9, 0.05, 0.05])
>>> w0 = np.array([0.45, 0.45, 0.1])
>>> y0 = np.array([0.3, 0.1, 0.6])
>>> vlle(x0, w0, y0, z, T, P, eos, full_output = False)
>>> # phase compositions
... (array([0.96430196, 0.03056118, 0.00513686]), array([0.44365858, 0.40405065, 0.15229077]),
...  array([0.32687062, 0.06433222, 0.60879716]))

.. automodule:: SGTPy.equilibrium.hazt
    :members: vlle
    :undoc-members:
    :show-inheritance:
