SGT for pure components
=======================

When working with pure components, SGT implementation is direct as there is a continuous path from the vapor to the liquid phase. SGT can be reformulated using the density as an independent variable.

.. math::
	\sigma = \sqrt{2} \int_{\rho^\alpha}^{\rho_\beta} \sqrt{c_i \Delta \Omega} d\rho

Here, :math:`\Delta \Omega` represents the grand thermodynamic potential difference, obtained from:

.. math::
	\Delta \Omega = a_0 - \rho \mu^0 + P^0

Where :math:`P^0` is the equilibrium pressure.

In SGTPy this integration is done using orthogonal collocation, which reduces the number of nodes needed for a desired error. This calculation is done with the ``sgt_pure`` function and it requires the equilibrium densities, temperature and pressure as inputs.

>>> from SGTPy import component, saftvrmie
>>> # The pure component is defined with the influence parameter
>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
...                     rdAB = 0.4, sites = [0,2,2], cii = 1.5371939422641703e-20)
>>> eos = saftvrmie(water)

First, vapor-liquid equilibria have to be computed. This is done with the ``psat`` method from the EoS, which returns the pressure and densities at equilibrium. Then the interfacial can be computed as it is shown.
>>> from SGTPy.sgt import sgt_pure
>>> T = 373. # K
>>> P0 = 1e5 # Pa
>>> P, vl, vv = eos.psat(T, P0=P0)
>>> rhol = 1/vl
>>> rhov = 1/vv
>>> sgt_pure(rhol, rhov, T, P, eos)
... array([58.84475645])

Optionally, ``full_output`` allows getting all the computation information as the density profile, interfacial length and grand thermodynamic potential.

>>> solution = sgt_pure(rhol, rhov, T, Psat, eos, full_output = True)
>>> solution.z # interfacial lenght array
>>> solution.rho # density path array
>>> solution.tension # IFT computed value


.. automodule:: SGTPy.sgt.sgtpuros
    :members: sgt_pure
    :undoc-members:
    :show-inheritance:
