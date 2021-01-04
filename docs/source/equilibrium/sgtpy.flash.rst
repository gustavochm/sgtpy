Two phase Flash
===============

This is the most classical calculation of phase equilibria. Temperature, pressure and global composition of a system are known. If the mixture is unstable it will split into two o more phases. When trying to compute two-phase separation the flash algorithm can be used.  The usual approach to solve this problem is to solve the following mass balance and then update composition by successive substitution.

.. math::
	FO = \sum_{i=1}^c \left( x_i^\beta - x_i^\alpha \right) = \sum_{i=1}^c \frac{z_i (K_i-1)}{1+\psi (K_i-1)}

Where, :math:`z`, is the global composition of component :math:`K =  x^\beta / x^\alpha` are the equilibrium constant and :math:`\psi` is the phase fraction of phase :math:`\beta`. Subscript refers to component index and superscript refers to the phase index.
This method can be slow at high pressures, and in those cases, the algorithm changes to a second-order minimization of the Gibbs free energy of the system:

.. math::
	min \, {G(\underline{F}^\alpha, \underline{F}^\beta)} = \sum_{i=1}^c (F_i^\alpha \ln \hat{f}_i^\alpha + F_i^\beta \ln \hat{f}_i^\beta)

Where, :math:`F` is the mole number and :math:`\hat{f}` is the effective fugacity.

In the following code block a flash calculation for vapor-liquid equilibria is  shown:


>>> from SGTPy import component, mixture, saftvrmie
>>> from SGTPy.equilibrium import flash
>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6.,  eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515455e-20)
>>> ethanol = component('ethanol2C', ms = 1.7728, sigma = 3.5592 , eps = 224.50,
...                     lambda_r = 11.319, lambda_a = 6., eAB = 3018.05, rcAB = 0.3547,
...                     rdAB = 0.4, sites = [1,0,1], cii= 5.3141080872882285e-20)
>>> mix = mixture(ethanol, water)
>>> kij, lij = np.array([-0.0069751 , -0.01521566])
>>> Kij = np.array([[0, kij], [kij, 0]])
>>> Lij = np.array([[0., lij], [lij, 0]])
>>> # setting interactions corrections
>>> mix.kij_saft(Kij)
>>> mix.lij_saft(Lij)
>>> # creating eos model
>>> eos = saftvrmie(mix)
>>> # flash conditions
>>> T = 355. # K
>>> P = 1e5 # Pa
>>> z = np.array([0.4, 0.6]) # global composition
>>> # initial guesses
>>> x0 = np.array([0.1, 0.9])
>>> y0 = np.array([0.8, 0.2])
>>> flash(x0, y0, 'LV', z, T, P, eos)
>>> # x, y, psi
... (array([0.2636008, 0.7363992]), array([0.55807925, 0.44192075]), 0.4631890692698939)

The same algorithm can be applied for liquid-liquid equilibria, as can be seen:


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
>>> # LLE is performed as a flash
>>> flash(x0, w0, 'LL', z, T, P, eos)
... (array([0.96022175, 0.03977825]), array([0.53375333, 0.46624667]), 0.3756942886430804)

.. automodule:: SGTPy.equilibrium.flash
    :members: flash
    :undoc-members:
    :show-inheritance:
