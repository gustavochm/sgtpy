# SGTPy Changelog


## v0.0.19
* The SAFT-gamma-Mie database has been updated to include recent parameters of the following articles: [Febra et al. (2021)](https://doi.org/10.1016/j.fluid.2021.113002), [Wehbe et al. (2022)](https://doi.org/10.1080/00268976.2023.2197712), [Perdomo et al. (2023)](https://doi.org/10.1016/j.fluid.2022.113635), [Valsecchi et al. (2024)](https://doi.org/10.1016/j.fluid.2023.113952) and [Bernet et al. (2024)](https://doi.org/10.1021/acs.jced.3c00358).
* The SAFT-gamma-Mie database now includes a `author_key` and `doi` to identify where the parameters come from.
* The `bubbleTy` and `bubblePy` functions now include an attribute `not_in_y_list` in which the user can provide the index (or indices) of a compononent not present in the vapor phase. This is useful is there are very heavy non-volatile components in a mixture.  
* The `dewTx` and `dewPx` functions now include an attribute `not_in_x_list` in which the user can provide the index (or indices) of a compononent not present in the liquid phase. This is useful is there are light non-condensables components in a mixture.
* The `flash` function now includes the attributes  `not_in_x_list` and `not_in_y_list`, where the user can provide the indices of the components not present in the phase `x` or `y`. This is useful in VLE if there are non-volatile/non-condensables components in the mixture.
* The `flash` function now allows choosing between two minimization approaches: `gibbs` or `helmholtz`. The  new  `helmholtz` minimizes the Gibbs free energy using a Helmholtz approach. See [Nagaranjan et al. (1991)](https://doi.org/10.1016/0378-3812(91)80011-J) for further details.
* `saftgammamie` and `saftvrmie` EoS now include the method `association_solver` and `association_check` to compute and to check the non-bonded sites results.

## v0.0.18

* Fixed bug in assosiation configuration for SAFT-gamma-mie. The bug only affected association sites of the same type in differenent groups (e.g. 'e1' - 'e1' and 'H'-'H' association.)

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
