SAFT-VR-Mie for fluid mixtures
==============================

The set up of fluid mixtures is shown :doc:`here <./sgtpy.mixtures>`. SAFT-VR-Mie EoS bases its unlike interactions using the following mixing rules to evaluate the size parameter (:math:`\sigma_{ij}`), well depth (:math:`\epsilon_{ij}`) and attractive and repulsive exponents (:math:`\lambda_{ij}`).

.. math::
   \sigma_{ij}=\frac{\sigma_{ii}+\sigma_{jj}}{2}
.. math::
   \epsilon_{ij} = (1-k_{ij}) \frac{\sqrt{\sigma_i^3 \sigma_j^3}}{\sigma_{ij}^3} \sqrt{\epsilon_i \epsilon_j}

.. math::
   \left(\lambda_{k,ij}-3\right)=\sqrt{\left(\lambda_{k,ii}-3\right)\left(\lambda_{k,jj}-3\right)}\quad;\quad k=a,r

Here only the Mie potential interaction energy admits a binary interaction parameter :math:`k_{ij}`, this is optimized from experimental phase equilibria data.

To use SAFT-VR-Mie EoS with fluid mixtures, a created mixture object is passed to the :func:`SGTPy.saftvrmie` function. This function returns the SAFT-VR-Mie EoS object and includes the needed method to compute properties such as density, pressure, residual heat capacities, entropy and enthalpy, chemical potential and fugacity coefficients.
The set up the ``eos`` object is straightforward, as shown below:

>>> from SGTPy import component, mixture, saftvrmie
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
>>> eos = saftvrmie(mix)

As for the case of :doc:`pure fluids <./sgtpy.puresaft>`, for mixtures of CG fluids, the influence parameter can be correlated from the molecular parameters [1]_.

.. math::
   \sqrt{\frac{c_{ii}}{N_{av}^2 \epsilon_{ii} \sigma_{ii}^5}} = m_{s,i} (0.12008 + 2.21989\alpha_{ii})
.. math::
   \alpha_{ii} = \frac{\lambda_{r, ii}}{3(\lambda_{r, ii} -3 )} \left( \frac{\lambda_{r, ii}}{6} \right)^{6/\lambda_{r, ii}-6}

>>> eos.cii_correlation(overwrite = True)
... array([1.92075094e-20, 1.27211926e-18])  # influence parameter in J m5/mol2



For the case of cross-association the unlike site geometry (:math:`r^{ABij}`) and energy (:math:`\epsilon^{AB}_{ij}`) are obtained with the following mixing rules:

.. math::
  r_{c}^{ABij}=\frac{r_{c}^{ABii}+r_{c}^{ABjj}}{2}
.. math::
  r_{d}^{ABij}=\frac{r_{d}^{ABii}+r_{d}^{ABjj}}{2}
.. math::
   \epsilon_{ij}^{AB} = (1 - l_{ij})\sqrt{\epsilon_{ii}^{AB} \epsilon_{jj}^{AB}}

The association energy admits a binary interaction parameter (:math:`l_{ij}`), this parameter can be optimized alongside the :math:`k_{ij}` parameter from experimental equilibria data.


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
>>> eos = saftvrmie(mix)

Another possible scenario is when a fluid in the mixture doesn't self-associate but can associate with other components in the mixture. This phenomenon occurs with mixtures of ethers and alcohols. A simple workaround of this induced association is to consider the cross association energy as half of the association energy of the self associating fluid and to optimize the geometry of the site. As in SAFT-VR-Mie the distance :math:`r_d^{AB}` is usually set to :math:`0.4\sigma` this implies that only :math:`r_c^{AB}` is optimized.

.. math::
   \epsilon_{ij}^{AB} = \frac{\epsilon^{AB} (self-associating)}{2}; \qquad  r_{c}^{ABij} (fitted)

The ternary mixture of hexane, ethanol and CPME exhibit induced association. The mixture is created as usual.

>>> from SGTPy import component, mixture
>>> ethanol = component('ethanol2C', ms=1.7728, sigma=3.5592 , eps=224.50,
...                    lambda_r=11.319, lambda_a=6., eAB = 3018.05, rcAB=0.3547,
...                    rdAB=0.4, sites=[1,0,1], cii=5.3141080872882285e-20, Mw=46.07)
>>> hexane = component('hexane', ms=1.96720036, sigma=4.54762477, eps=377.60127994,
...                    lambda_r=18.41193194, cii=3.581510586936205e-19, Mw=100.16)
>>> cpme = component('cpme', ms=2.32521144, sigma=4.13606074, eps=343.91193798, lambda_r=14.15484877,
...                  lambda_a=6.0, npol=1.91990385, mupol=1.27, sites=[0,0,1],
...                  cii=3.5213681817448466e-19, Mw=86.18)
>>> # creating mixture
>>> mix = mixture(hexane, ethanol)
>>> # adding a component
>>> mix.add_component(cpme)

The :math:`k_{ij}` binary correction corrections are set by pairs and the induced association is set by manually modifying the association energy matrix (``eos.eABij``) and the association range matrix (``eos.rcij``).

>>> # setting kij corrections
>>> k12 = 0.011818492037463553
>>> k13 = 0.0008700151297528677
>>> k23 = 0.01015194
>>> Kij = np.array([[0., k12, k13], [k12, 0., k23], [k13, k23, 0.]])
>>> mix.kij_saft(Kij)
>>> eos = saftvrmie(mix)
>>> # induced association set up
>>> rc = 2.23153033 # Angstrom
>>> eos.eABij[1,2] = ethanol.eAB / 2
>>> eos.eABij[2,1] = ethanol.eAB / 2
>>> eos.rcij[1,2] = rc * 1e-10
>>> eos.rcij[2,1] = rc * 1e-10

The ``eos`` object can be used for some basic properties calculations.
The density of the mixture is computed with the :func:`density <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.density>` method at given composition, temperature and pressure.
This method includes automatic initialization with Topliss's method [2]_, on the other hand, when an initial guess is available it uses Newton's method.

>>> T = 350.  # K
>>> P = 1e5  # Pa
>>> x = np.array([0.1, 0.3, 0.6])
>>> eos.density(x, T, P, 'L')
... 9311.834138400469  # liquid density in mol/m3
>>> eos.density(x, T, P, 'V')
... 35.458148587066376 # vapor density in mol/m3

The ``eos`` object also can compute the pressure of the mixture at given composition, density and temperature using the :func:`pressure <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.pressure>` method.
In the following code block, the previously computed volume roots are verified.


>>> T = 350.  # K
>>> x = np.array([0.1, 0.3, 0.6])
>>> rhol = 9311.834138400469  # mol/m3
>>> eos.pressure(x, rhol, T)
... 99999.99999504909  # Pa
>>> rhov = 35.458148587066376  # mol/m3
>>> eos.pressure(x, rhov, T)
... 99999.99999968924  # Pa

The ``eos`` object is used for phase equilibria calculation through fugacity coefficient, this is donde with the :func:`logfugef <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.logfugef>` method.
This function requires the composition, temperature and pressure and returns the computed fugacity coefficient and the compute volume root.

>>> T = 350.  # K
>>> P = 1e5  # Pa
>>> x = np.array([0.1, 0.3, 0.6])
>>> eos.logfugef(x, T, P, 'L')
>>> # fugacity coefficients and computed liquid volume root
... (array([ 0.77982905,  0.47877663, -0.79012744]), 0.00010739022894277775)
>>> eos.logfugef(x, T, P, 'V')
>>> # fugacity coefficients and computed vapor volume root
... (array([-0.02400476, -0.03146375, -0.03088407]), 0.02820226209906395)

For the calculation of interfacial properties using SGT, the ``eos`` object includes the calculation of the chemical potential using the :func:`muad <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.muad>` method.
This function requires the density vector and temperature as inputs and return the chemical potentials.

>>> T = 350.  # K
>>> x = np.array([0.1, 0.3, 0.6])
>>> rhol = 9311.834138400469  # mol/m3
>>> eos.muad(rhol*x, T)
... array([-1.2536145 , -0.45605463, -1.03181152])
>>> rhov = 35.458148587066376  # mol/m3
>>> eos.muad(rhov*x, T)
... array([-2.0574483 , -0.96629501, -0.27256815])


The ``eos`` object is also capable of computing thermal derived properties.
In the following code block the residual entropy is computed with the :func:`EntropyP <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.EntropyR>` method,
the residual enthalpy is computed with the :func:`EnthalpyR <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.EnthalpyR>` method and the residual
heat capacity is computed with the :func:`CpR <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.Cpr>` method. Finally, the speed of sound requires
the total isobaric and isochoric heat capacities, for simplicity in this example the ideal gas contributions are considered as ``5R/2`` and ``3R/2``, respectively.
Then the speed of sound is computed with the  :func:`speed_sound <SGTPy.mixtures.saftvrmiemix.saftvrmie_mix.speed_sound>` method.



>>> T = 350.  # K
>>> P = 1e5  # Pa
>>> x = np.array([0.1, 0.3, 0.6])
>>> eos.EntropyR(x, T, P, 'L')
... -96.33127812255216  # Residual entropy in J/mol K
>>> eos.EnthalpyR(x, T, P, 'L')
...-34450.62328681776 # Residual enthalpy in J/mol
>>> eos.CpR(x, T, P, 'L')
... 58.66381738561176  # Residual heat capacity in J/mol K
>>> R = 8.314 # J / mol K
>>> CvId = 3*R/2
>>> CpId = 5*R/2
>>> eos.speed_sound(x, T, P, 'L', CvId=CvId, CpId=CpId)
... 1189.1342143755487 # speed of sound in m/s


.. autofunction:: SGTPy.saftvrmie
   :noindex:

.. automodule:: SGTPy.mixtures.saftvrmiemix
   :members: saftvrmie_mix
   :undoc-members:
   :show-inheritance:

.. [1] `AIChE Journal, 62(5), 1781–1794 (2016). <https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/aic.15190>`_
.. [2] `Computers & Chemical Engineering, 12(5), 483–489 (1988). <https://www.sciencedirect.com/science/article/abs/pii/0098135488850671>`_
