SAFT-VR-Mie for pure fluids
===========================

The set up of pure components is shown :doc:`here <./sgtpy.component>`. The created object is passed to the :func:`SGTPy.saftvrmie` function, this function returns the SAFT-VR-Mie EoS object which is ready to do some basic computation such as, density, pressure, fugacity coefficient, chemical potential, as well as, some thermal derived properties such as residual heat capacities, entropy and enthalpy.

The set up the ``eos`` is straightforward, as shown bellow:

>>> from SGTPy import component, saftvrmie
>>> methane = component('methane', ms = 1.0, sigma = 3.752 , eps = 170.75,
...                    lambda_r = 16.39, lambda_a = 6.)
>>> eos = saftvrmie(methane)

In the case of CG fluids the influence parameter can be correlated from the molecular parameters [1]_ according to the following expression:

.. math::
   \sqrt{\frac{c_{ii}}{N_{av}^2 \epsilon \sigma^5}} = m_s (0.12008 + 2.21989\alpha)
.. math::
   \alpha = \frac{\lambda_r}{3(\lambda_r -3 )} \left( \frac{\lambda_r}{6} \right)^{6/\lambda_r-6}

>>> # correlated influence parameter for methane
>>> eos.cii_correlation(overwrite=True)
... 1.9207509420744775e-20  # influence parameter in J m5 / mol2


The :func:`SGTPy.saftvrmie` function can identify the contributions present in the fluid. For a self-associating fluid:

>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], Mw = 18.01528, cii = 1.5371939421515458e-20)
>>> eos = saftvrmie(water)

The ``eos`` object includes useful calculations using SAFT-VR-Mie EoS. These methods work with the international system of units, that is, the temperature in Kelvin, pressure in Pascal, density in mol/m :math:`^3`. The :func:`density <SGTPy.pure.saftvrmie.saftvrmie_pure.density>` method can compute volume roots of liquid and vapor phases at given temperature and pressure. The method includes automatic initiation using Topliss's method [2]_, if an initial value is available the volume is computed with Newton's method.

>>> T = 300.  # K
>>> P = 1e5  # Pa
>>> eos.density(T, P, 'L')
... 56938.970481192526  # mol/m3
>>> eos.density(T, P, 'V')
... 44.962772347755454  # mol/m3

The :func:`pressure <SGTPy.pure.saftvrmie.saftvrmie_pure.pressure>` method computes the pressure at given density and temperature.  In the following codeblock, the previously computed roots are verified.

>>> T = 300.  # K
>>> rhov = 44.962772347754836  # mol/m3
>>> eos.pressure(rhov, T)
... 99999.99999999964  # Pa
>>> rhol = 56938.970481192526  # mol/m3
>>> eos.pressure(rhol, T)
... 100000.00000008068  # Pa

The :func:`psat <SGTPy.pure.saftvrmie.saftvrmie_pure.psat>` method allows to compute the saturation pressure at given temperature. It requires either an initial guess for the saturation pressure ``P0`` or for the volume of the phases ``v0``.
>>> T = 300.  # K
>>> P0 = 1e3  # Pa
>>> eos.psat(T, P0=P0)
>>> # equilibrium pressure (Pa), liquid volume (mol/m3), vapor volume (mol/m3)
... (3640.841209122654, 1.756342718267362e-05, 0.6824190076059896)

The phase equilibria calculation can be verified through chemical potentials (:func:`muad <SGTPy.pure.saftvrmie.saftvrmie_pure.muad>`) or through fugacity coefficients (:func:`logfug <SGTPy.pure.saftvrmie.saftvrmie_pure.logfug>`)


>>> Psat = 3640.841209122654  # Pa
>>> vl = 1.756342718267362e-05  # m3/mol
>>> vv = 0.6824190076059896  # m3/mol
>>> np.allclose(eos.muad(1/vl, T) , eos.muad(1/vv, T))
... True
>>> np.allclose(eos.logfug(T, Psat, 'L', v0=vl)[0], eos.logfug(T, Psat, 'V', v0=vv)[0])
... True

The ``eos`` object is also capable of computing other properties such as residual entropies. This is done with the :func:`EntropyR <SGTPy.pure.saftvrmie.saftvrmie_pure.EntropyR>` method.

>>> Sl = eos.EntropyR(T, Psat, 'L', v0=vl)
>>> Sv = eos.EntropyR(T, Psat, 'V', v0=vv)
>>> Sl - Sv
... -142.8199794928061 # J / mol K
>>> # NIST value = -146.3584 J / mol K

The calculation of residual enthalpies is done in the same manner using the :func:`EnthalpyR <SGTPy.pure.saftvrmie.saftvrmie_pure.EnthalpyR>` method.

>>> Hl = eos.EnthalpyR(T, Psat, 'L')
>>> Hv = eos.EnthalpyR(T, Psat, 'V')
>>> (Hl - Hv) / 1000.
... -42.84599384780775 # kJ / mol
>>> # NIST value = -43.9081 kJ / mol

For the calculation of heat capacities, only the residual contribution is computed from the SAFT-VR-Mie EoS using the :func:`CpR <SGTPy.pure.saftvrmie.saftvrmie_pure.CpR>` method. The ideal contribution can be obtained from correlations or data banks, such as DIPPR 801.

>>> R = 8.314  # J/mol K
>>> # Ideal Gas Heat Capacity by DIPPR
>>> k1=33363
>>> k2=26790
>>> k3=2610.5
>>> k4=8896
>>> k5=1169
>>> CpId = k1 + k2 * ((k3/T) /np.sinh(k3/T))**2
>>> CpId += k4 * ((k5/T) /np.cosh(k5/T))**2
>>> CpId /= 1000. # J / mol K
>>> CvId = CpId - R # J / mol K
>>> eos.CpR(T, Psat, 'L', v0=vl) + CpId
... 61.79557814825187 # J / mol K
>>> # NIST value = 75.320 J / mol K
>>> eos.CpR(T, Psat, 'V', v0=vv) + CpId
... 35.34264072509833 # J / mol K
>>> # NIST value = 34.483 J / mol K

Finally, the speed of sound of the phases can be computed using the :func:`speed_sound <SGTPy.pure.saftvrmie.saftvrmie_pure.speed_sound>` method. As this calculation requires the total isobaric and isochoric heat capacities, the ideal contribution must be supplied manually.

>>> eos.speed_sound(T, Psat, 'L', v0=vl, CvId=CvId, CpId=CpId)
... 1542.8100435020717 # m/s
>>> # NIST value = 1501.4 m/s
>>> eos.speed_sound(T, Psat, 'V', v0=vv, CvId=CvId, CpId=CpId)
... 427.1691887269907 # m/s
>>> # NIST value = 427.89 m/s

.. autofunction:: SGTPy.saftvrmie

.. automodule:: SGTPy.pure.saftvrmie
   :members: saftvrmie_pure
   :undoc-members:
   :show-inheritance:

.. [1] `AIChE Journal, 62(5), 1781–1794 (2016). <https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/aic.15190>`_
.. [2]  `Computers & Chemical Engineering, 12(5), 483–489 (1988). <https://www.sciencedirect.com/science/article/abs/pii/0098135488850671>`_
