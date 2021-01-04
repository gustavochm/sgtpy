SGTPy.equilibrium
===================


Phase equilibrium conditions are obtained from a differential entropy balance of a multiphase system. The following equations must be solved:

.. math::
	T^\alpha = T^\beta = ... &= T^\pi\\
	P^\alpha = P^\beta = ... &= P^\pi\\
	\mu_i^\alpha = \mu_i^\beta = ... &= \mu_i^\pi \quad i = 	1,...,c

Where :math:`T`, :math:`P` and :math:`\mu` are the temperature, pressure and chemical potential. When working with EoS usually equilibrium is guaranteed by fugacity coefficients:

.. math::
	x_i^\alpha\hat{\phi}_i^\alpha = x_i^\beta \hat{\phi}_i^\beta = ... = x_i^\pi \hat{\phi}_i^\pi \quad i = 1,...,c


Usual equilibrium calculations include vapor-liquid equilibrium (flash, bubble point, dew point), liquid-liquid equilibrium (flash and stability test) and vapor-liquid-liquid equilibrium (multiflash and stability test). Those algorithms are described in the following sections:

.. toctree::
	 :maxdepth: 1

	 ./equilibrium/sgtpy.flash
	 ./equilibrium/sgtpy.vle
	 ./equilibrium/sgtpy.multiflash
	 ./equilibrium/sgtpy.stability
