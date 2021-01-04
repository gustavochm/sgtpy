Available calculations in SGTPy
-------------------------------

- Property calculation from SAFT-VR-Mie EoS

==================================  ====================
Property                            Available?
==================================  ====================
Density                             |:heavy_check_mark:|
Pressure                            |:heavy_check_mark:|
Helmholtz free energy               |:heavy_check_mark:|
Chemical Potential                  |:heavy_check_mark:|
Fugacity coefficient                |:heavy_check_mark:|
Helmholtz Free Energy               |:heavy_check_mark:|
Residual Entropy*                   |:heavy_check_mark:|
Residual Enthalpy*                  |:heavy_check_mark:|
Residual isochoric heat capacity*   |:heavy_check_mark:|
Residual isobaric heat capacity*    |:heavy_check_mark:|
Speed of sound*                     |:heavy_check_mark:|
==================================  ====================


.. warning:: *Temperature derivatives of Helmholtz free energy are computed numerically



- Phase Equilibria

============================================   ==================
Phase equilibria                               Available?
============================================   ==================
Phase stability (tpd minimization)             |:heavy_check_mark:|
TP flash                                       |:heavy_check_mark:|
Bubble points                                  |:heavy_check_mark:|
Dew points                                     |:heavy_check_mark:|
Liquid-Liquid Equilibria (multiplash)          |:heavy_check_mark:|
Vapor-Liquid-Liquid Equilibria (multiplash)    |:heavy_check_mark:|
HP flash
SP flash
============================================   ==================

- Square Gradient Theory solution methods

============================================   ====================
 Calculation method                            Available?
============================================   ====================
Pure fluid                                     |:heavy_check_mark:|
Reference component (mixtures)                 |:heavy_check_mark:|
Path function (mixtures)                       |:heavy_check_mark:|
Orthogonal collocation (mixtures)              |:heavy_check_mark:|
Stabilized BVP (mixtures)                      |:heavy_check_mark:|
============================================   ====================



Guidelines to reuse SGTPy functions
===================================

SGTPy includes functions to compute phase equilibria and interfacial properties using Square Gradient Theory.
Those functions can be used with the included SAFT-VR-Mie EoS, but they will also work with any user-defined model as long as it
meets the following requirements. The model (``eos`` ) should be an object with attributes and methods.



Model attributes
================


==================   ===========   =============================================
Attribute             Type         Description
==================   ===========   =============================================
eos.nc                integrer     Number of components in the mixture
eos.secondordersgt    bool         | whether derivatives of the chemical potential
                                   | are available through the eos.dmuad method
==================   ===========   =============================================


Pure fluid methods
==================



- **eos.temperature_aux(T)**

Method that compute all the temperature dependent parameters and return them as a tuple.


*Input*: Temperature

*Return*: tuple (temp_aux)


- **eos.logfug_aux(temp_aux, P, state, v0, Xass0)**

Method that computes the natural logarithm of the fugacity coefficient for pure fluids (nc=1).


*Input*: temperature dependent parameters (temp_aux), pressure (P), state ('L' for liquid phase and 'V' for vapor phase), v0 is used as initial guess to compute the volume root, if ``None`` it should automatically initiate the calculation. Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*:  natural logarithm of the fugacity coefficient (lnphi), computed volume root (v) and computed fraction of nonbonded sites (Xass).

- **eos.muad_aux(rho, temp_aux, Xass0)**

Method that computes chemical potential for pure fluids  (nc=1).

*Input*: density (rho), temperature dependent parameters (temp_aux). Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*: chemical potential (mu) and computed fraction of nonbonded sites (Xass).

- **eos.a0ad_aux(rho, temp_aux, Xass0)**

Method that computes the Helmtholtz density free energy for pure fluids (nc=1).

*Input*: density (rho), temperature dependent parameters (temp_aux). Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*: Helmtholtz density free energy (a0ad) and computed fraction of nonbonded sites (Xass).


- **eos.dOm_aux(rho, temp_aux, mu0, P, Xass0)**

Method that computes the delta of the Grand Thermodynamic Potential for pure fluids (nc=1).

*Input*: density (rho), temperature dependent parameters (temp_aux), equilibrium chemical potential at given temperature (mu0), equilibrium pressure (P), Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*: Grand Thermodynamic Potential (dOm) and computed fraction of nonbonded sites (Xass).

- **eos.sgt_adim(T)**

Method that computes the factor to make SGT calculations consistent.

*Input*: Temperature

*Return*: temperature factor (Tfactor), pressure factor (Pfactor), density factor (rofactor), interfacial tension factor (tenfactor), interfacial lenght factor (zfactor)




Fluid mixtures methods
======================

- **eos.temperature_aux(T)**

Method that compute all the temperature dependent parameters and return them as a tuple.

*Input*: Temperature

*Return*: tuple (temp_aux)


- **eos.logfugef_aux(x, temp_aux, P, state, v0, Xass0)**

Method that computes the natural logarithm of the effective fugacity coefficient for mixtures (nc>=2).

*Input*: composition (x), temperature dependent parameters (temp_aux), pressure (P), state ('L' for liquid phase and 'V' for vapor phase), v0 is used as initial guess to compute the volume root, if ``None`` it should automatically initiate the calculation. Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*:  natural logarithm of the fugacity coefficient (lnphi), computed volume root (v) and computed fraction of nonbonded sites (Xass).


- **eos.muad_aux(rhoi, temp_aux, Xass0)**

Method that computes the chemical potential for mixtures (nc>=2).

*Input*: density vector (rhoi), temperature dependent parameters (temp_aux). Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*:  chemical potential (mu) and computed fraction of nonbonded sites (Xass).

- **eos.dmuad_aux(rhoi, temp_aux, Xass0)**

Method that computes the chemical potential and its derivatives matrix (d mu_i / d rho_j) for mixtures (nc>=2).

*Input*: density vector (rhoi), temperature dependent parameters (temp_aux). Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*:  chemical potential (mu), its derivatives matrix (dmu) and computed fraction of nonbonded sites (Xass).

- **eos.a0ad_aux(rhoi, temp_aux, Xass0)**

Method that computes the Helmtholtz density free energy for pure for mixtures (nc>=2).

*Input*: density vector (rhoi), temperature dependent parameters (temp_aux). Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*: Helmtholtz density free energy (a0ad) and computed fraction of nonbonded sites (Xass).


- **eos.dOm_aux(rhoi, temp_aux, mu0, P, Xass0)**

Method that computes the delta of the Grand Thermodynamic Potential for pure fluids (nc=1).

*Input*: density vector (rhoi), temperature dependent parameters (temp_aux), equilibrium chemical potential at given temperature (mu0), equilibrium pressure (P), Xass0 is used as initial guess to compute the fraction of nonbonded sites, ``None`` should be used for non-associating mixtures or to automatically initiate the calculation.

*Return*: Grand Thermodynamic Potential (dOm) and computed fraction of nonbonded sites (Xass).

- **eos.sgt_adim(T)**

Method that computes the factor to make SGT calculations consistent.

*Input*: Temperature

*Return*: temperature factor (Tfactor), pressure factor (Pfactor), density factor (rofactor), interfacial tension factor (tenfactor), interfacial lenght factor (zfactor)

- **eos.ci(T)**

Method that computes the influence parameter matrix for mixtures.

*Input*: Temperature

*Return*: influence parameter matrix (cij)
