# AblationModel
This code will determine the thickness needed for a thermal ablative subject to certain temperature conditions inside the Genesis rocket engine. It uses 1D Charring Material Ablative logic.

The wall is broken up into discrete nodes(i=1,2,3,...N) and the temperature at each node is calculated based on the heat flux from the hot gases and the thermal properties of the material.

Boundary conditions are supplied by data from RPA and known values.\

Theres also a file that calculates mass flow rates for a Cold Gas Thruster but that is not important
