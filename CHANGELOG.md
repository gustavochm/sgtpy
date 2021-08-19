# SGTPy Changelog

## v0.0.13

* New `eos.set_induced_asso` method in SAFT-VR-Mie to set up induced association (solvation).
* New `mixture.set_kijsaft` and `mixture.set_lijsaft` methods to set kij and lij in SAFT-VR-Mie.
* Now you can set temperature dependent kij and lij for SAFT-VR-Mie.
* New `eos.set_betaijsgt` method to set betaij in SGT in SAFT-VR-Mie and SAFT-gamma-Mie.
* Now you can set temperature dependent betaij in SGT calculations.
* Now you set temperature dependent polynomial for influence parameters (`cii`) in SAFT-VR-Mie and SAFT-gamma-Mie.
* Module renamed from `SGTPy` to `sgtpy` (PEP-8 standard)

## v0.0.12

* Now you can add (+) pure components to create mixtures.
* Bugfix in full_output in Psat/Tsat for pure fluids.
* Group contribution SAFT-gamma-Mie added.
* Included forcefield for ring-like molecular parameters calculation.
* Bugfix in sgt_mix ift calculation without jacobian.

## v0.0.11

* Included critical point calculation for pure fluids.
* First Changelog!
