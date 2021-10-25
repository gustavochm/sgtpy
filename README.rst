=====
sgtpy
=====

What is sgtpy?
--------------

sgtpy is an open-source python package of SAFT-VR-Mie and SAFT-Gamma-Mie Equations of State (EOS).
sgtpy allows to work with pure fluids and fluid mixtures, additionally the fluids
can be modeled considering association, cross-association and polar contributions.
sgtpy was built on top of phasepy's phase equilibrium and Square
Gradient Theory (SGT) functions. These functions were updated to speed-up
calculations of associative mixtures.

sgtpy relies on numpy, scipy and phasepy.

**New in v0.0.12: Group contribution SAFT-Gamma-Mie**

**Note in v0.0.13:** Due to PEP-8 standard's the module name has change from `SGTPy` to `sgtpy`.


Installation Prerequisites
--------------------------
- numpy
- scipy
- cython
- pandas
- numba

Installation
------------

Get the latest version of sgtpy from
https://pypi.python.org/pypi/sgtpy/


If you have an installation of Python with pip, simple install it with:

    $ pip install sgtpy

To get the git version, run:

    $ git clone https://github.com/gustavochm/sgtpy

**Note for Apple Silicon users:** It is recommended to install python and sgtpy dependencies (numpy, scipy, cython, pandas and numba) through conda miniforge, then you can install sgtpy running ``pip install sgtpy``.

Documentation
-------------

SGTPy's documentation is available on the web (under development):

    https://sgtpy.readthedocs.io/en/latest/


Getting Started
---------------

sgtpy easily allows you to perform phase equilibria and interfacial properties
calculations using SAFT-VR-Mie or SAFT-gamma-Mie EoS. First, components are defined with their
molecular parameters, then a mixture can be created with them.

.. code-block:: python

      >>> import numpy as np
      >>> from sgtpy import component, mixture, saftvrmie
      >>> ethanol = component('ethanol2C', ms=1.7728, sigma=3.5592 , eps=224.50,
                    lambda_r=11.319, lambda_a=6., eAB=3018.05, rcAB=0.3547,
                    rdAB=0.4, sites=[1,0,1], cii=5.3141080872882285e-20)
      >>> hexane = component('hexane', ms=1.96720036, sigma=4.54762477,
                               eps=377.60127994, lambda_r=18.41193194,
                               cii=3.581510586936205e-19)
      >>> mix = mixture(hexane, ethanol)
      >>> # fitted to experimental data
      >>> kij = 0.011818492037463553
      >>> Kij = np.array([[0, kij], [kij, 0]])
      >>> mix.kij_saft(Kij)
      >>> eos = saftvrmie(mix)

The eos object can be used to compute phase equilibria.

.. code-block:: python

      >>> from sgtpy.equilibrium import bubblePy
      >>> # computing bubble point
      >>> T = 298.15 # K
      >>> x = np.array([0.3, 0.7])
      >>> # initial guesses for vapor compotision and pressure
      >>> y0 = 1.*x
      >>> P0 = 8000. # Pa
      >>> sol = bubblePy(y0, P0, x, T, eos, full_output=True)

Finally, the equilibria results can be used to model the interfacial behavior of
the mixture using SGT.

.. code-block:: python

      >>> from sgtpy.sgt import sgt_mix
      >>> # reading solution object
      >>> y, P = sol.Y, sol.P
      >>> vl, vv = sol.v1, sol.v2
      >>> #density vector of each phase
      >>> rhox = x/vl
      >>> rhoy = y/vv
      >>> bij = 0.05719272059410664
      >>> beta = np.array([[0, bij], [bij, 0]])
      >>> eos.beta_sgt(beta)
      >>> #solving BVP of SGT with 25 colocation points
      >>> solsgt = sgt_mix(rhoy, rhox, T, P, eos, n = 25, full_output = True)

For more examples, please have a look at the Jupyter Notebook files
located in the *examples* folder of the sources or
`view examples in github <https://github.com/gustavochm/SGTPy/tree/master/Examples>`_.



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

This package is part of the article *SGTPy: A Python open-source code for
calculating the interfacial properties of fluids based on the Square Gradient
Theory using the SAFT-VR Mie equation of state* by Andrés Mejía,
Erich A. Müller and Gustavo Chaparro. J. Chem. Inf. Model., 2021,
`<https://doi.org/10.1021/acs.jcim.0c01324>`_.

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the sgtpy license, if it is convenient for you,
please cite sgtpy if used in your work. Please also consider contributing
any changes you make back, and benefit the community.
