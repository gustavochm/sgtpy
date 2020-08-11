=====
SGTPy
=====

What is SGTPy?
--------------

SGTPy is an open-source python package of SAFT-VR-Mie Equation of State (EOS).
SGTPy allows to work with pure fluids and fluid mixtures, additionally the fluids
can be modeled considering association, cross-association and polar contributions.
SGTPy was built on top of phasepy's phase equilibrium and Square
Gradient Theory (SGT) functions. These functions were updated to speed-up
calculations of associative mixtures.

SGTPy relies on Numpy, SciPy and PhasePy.


Installation Prerequisites
--------------------------
- numpy
- scipy

Installation
------------

Get the latest version of SGTPy from
https://pypi.python.org/pypi/SGTPy/


If you have an installation of Python with pip, simple install it with:

    $ pip install SGTPy

To get the git version, run:

    $ git clone https://github.com/gustavochm/SGTPy

Getting Started
---------------

.. code-block:: python
      >>> import numpy as np
      >>> from SGTPy import component, mixture, saftvrmie
      >>>> ethanol = component('ethanol2C', ms=1.7728, sigma=3.5592 , eps=224.50,
                    lambda_r=11.319, lambda_a=6., eAB=3018.05, rcAB=0.3547,
                    rdAB=0.4, sites=[1,0,1], cii=5.3141080872882285e-20)
      >>> hexane = component('hexane', ms=1.96720036, sigma=4.54762477, eps=377.60127994,
                   lambda_r=18.41193194, cii=3.581510586936205e-19)
       >>> mix = mixture(hexane, ethanol)
       >>> # fitted to experimental data
       >>> kij = 0.011818492037463553
       >>> Kij = np.array([[0, kij], [kij, 0]])
       >>> mix.kij_saft(Kij)
       >>> eos = saftvrmie(mix)

Latest source code
------------------

The latest development version of SGTPy's sources can be obtained at

    git clone https://github.com/gustavochm/SGTPy

Bug reports
-----------

To report bugs, please use the SGTPy's Bug Tracker at:

    https://github.com/gustavochm/SGTPy/issues


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the SGTPy license, if it is convenient for you,
please cite SGTPy if used in your work. Please also consider contributing
any changes you make back, and benefit the community.
