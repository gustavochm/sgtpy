SGTPy
=====

SGTPy is an object-oriented python software for phase equilibria and interfacial properties calculation using SAFT-VR-Mie EoS. The SAFT-VR-Mie Equation of State describes the Helmholtz free energy as follows:

.. math::
   a=a^{IG}+a^{MONO}+a^{CHARI}+a^{ASSOC}+a^{POL}

Where, :math:`a` is the total Helmholtz free energy, :math:`a^{IG}` is the ideal gas reference, :math:`a^{MONO}` represents monomer (unbounded) contribution, :math:`a^{CHARI}` accounts for the formation of chain and ring molecule while :math:`a^{ASSOC}` accounts for the intermolecular association contribution and :math:`a^{POL}` represents the polar contribution.
For further information about each contribution, we recommend the original references [1]_, [2]_, [3]_.


The coded equation of state was tested to pass the following molar partial property test and Gibbs-Duhem consistency:


 .. math::
 	\ln \phi - \sum_{i=1}^c x_i \ln \hat{\phi_i}  = 0 \\
 	\frac{d \ln \phi}{dP} - \frac{Z - 1}{P} = 0 \\
 	\sum_{i=1}^c x_i d \ln \hat{\phi_i} = 0

Here, :math:`\phi` is the fugacity coefficient of the mixture,  :math:`x_i` and :math:`\hat{\phi_i}` is the mole fraction and fugacity coefficient of component :math:`i`, :math:`P` refers to pressure and :math:`Z` to the compressibility factor.

To use SGTPy, first, it is required to create components and mixtures, and then combine them with the equation of state to create a final model object, which can be used to carry out the desired calculations.

.. toctree::
   ./basic/sgtpy.component
   ./basic/sgtpy.mixtures

With the class :class:`SGTPy.component`, only pure component info is saved. These parameters are required to evaluate SAFT-VR-Mie EoS. This includes numbers of segments (:math:`m_s`), well-depth of Mie potential (:math:`\epsilon`) in K units, size parameter of Mie potential (:math:`\sigma`) in Å (:math:`10^{-10}` m)  , attractive (:math:`\lambda_a`) and repulsive (:math:`\lambda_r`) exponents of Mie Potential. If the fluid is modeled as a ring it requires a geometric factor (:math:`\chi`) [2]_. For the case of pure self-associating fluid, three extra parameters are needed:  the association energy (:math:`\epsilon^{AB}`) in K units, the association range (:math:`r_c^{AB}/\sigma`) and association center position (:math:`r_d^{AB}/\sigma`). Additionally, the association scheme is characterized by the triple [B, P, N], which indicates the number of bipolar, positive and negative association sites, respectively. The polar contribution requires the definition of a dipolar moment (:math:`\mu`)  in Debye units, and the number of polar sites (:math:`n_p`).
Finally, the influence parameter (:math:`c_{ii}`) in J m :math:`^5` / mol :math:`^2` is required to study the interfacial behavior using Square Gradient Theory.

>>> from SGTPy import component
>>> water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
...                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
...                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515458e-20)
>>> water.sigma
... 2.4539e-10  # stored in meters
>>> water.eps
... 1.5304491948e-21  # stored in Joules
>>> butanol = component('butanol2C', ms = 1.9651, sigma = 4.1077 , eps = 277.892,
...                    lambda_r = 10.6689, lambda_a = 6., eAB = 3300.0, rcAB = 0.2615,
...                    rdAB = 0.4, sites = [1,0,1], npol = 1.45, mupol = 1.6609,
...                    cii  = 1.5018715324070352e-19)
>>> butanol.rcAB
... 1.0741635500000002e-10  # stored in meters
>>> butanol.eAB
... 4.55614104e-20  # stored in Joules

A mixture can be created from two components and the :class:`SGTPy.mixture` class:


>>> from SGTPy import mixture
>>> mix = mixture(water, butanol)
>>> mix.sigma
... [2.4539e-10, 4.1077000000000007e-10]
>>> mix.mupol
... [0, 1.6609]  # dipolar moment in Debye

The ``mix`` object stores pure component molecular parameters and allows to set interaction corrections for the cross-association energy and the Mie potential interaction energy. Both pure component or fluid mixture can be modeled with SAFT-VR-Mie EoS, examples of how to use obtain properties from the EoS are shown in the following sections.

.. toctree::
  :maxdepth: 1

  ./basic/sgtpy.puresaft
  ./basic/sgtpy.mixsaft


A complete list of available calculations in SGTPy is found :doc:`here <./basic/sgtpy.guidelines>`.

.. toctree::
  :hidden:

  ./basic/sgtpy.guidelines




.. [1] `Journal of Chemical Physics, 139(15), 1–37 (2013). <https://aip.scitation.org/doi/citedby/10.1063/1.4819786>`_
.. [2] `Langmuir, 33(42), 11518–11529 (2017). <https://pubs.acs.org/doi/abs/10.1021/acs.langmuir.7b00976>`_
.. [3] `AIChE Journal, 52(3), 1194–1204. (2006). <https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.10683>`_
