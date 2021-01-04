SGTPy.fit
===========

In order to compute phase equilibria and interfacial properties it is necessary to count with the molecular parameters of pure component parameters as: the numbers of segments (:math:`m_s`), well-depth of Mie potential (:math:`\epsilon`), size parameter of Mie potential (:math:`\sigma`), attractive (:math:`\lambda_a`) and repulsive (:math:`\lambda_r`) exponents of Mie Potential. For the case of pure self-associating fluid, three extra parameters are needed:  the association energy (:math:`\epsilon^{AB}`) in K units, the association range (:math:`r_c^{AB}/\sigma`) and association center position (:math:`r_d^{AB}/\sigma`). In the case of polar fluids, this contribution requires the number of polar sites (:math:`n_p`). Additionally, to model the fluid with SGT the influence parameter is required.

In the case of mixtures, the :math:`k_{ij}` and  :math:`l_{ij}` binary interaction parameters are required as the optimized :math:`r_{c}^{ABij}` for induced associating mixtures.

SGTPy includes several functions that rely on equilibria routines included in the package and in SciPy optimization tools for fitting model parameters. These functions are explained in the following sections for pure components and for mixtures.

.. toctree::
   ./fit/fit.pure
   ./fit/fit.mixtures
