# AblationModel
This code will determine the thickness needed for a thermal ablative subject to certain temperature conditions inside the Genesis rocket engine. It uses 1D Charring Material Ablative (CMA) logic.

The wall is broken up into discrete nodes(i=1,2,3,...N) and the temperature at each node is calculated based on the heat flux from the hot gases and the thermal properties of the material. From there, the physics of pyrolyzing ablatives are applied to the problem. An Arrehenius-type expression is used to specifiy the decomposition rate of the phenolic resin matrix and a "Char Fraction" is defined to interpolate across virgin and char values for variables like k and cp. Changes in the total thermal conductivity due to density and property changes is then fed back into the thermal model, which simulated the material pyrolizing over time. A correction for spatially varying conductivity is also applied because "k" changes, and the standard 1D heat conduction equation treats the matieral as isotropic.

Then, the main pyrolysis effects are added into the model, namely, the pyrolysis heat sink term, mass loss and gas generation, transpiration cooling, blowing cofrrection, Boltzmann re-radiation, and more that has yet to be introduced in the current model. Development is ongoing.

Boundary conditions are supplied by data from RPA and known values.

Theres also a file that calculates mass flow rates for a Cold Gas Thruster but that is not important.
