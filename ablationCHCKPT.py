import numpy as np
import matplotlib.pyplot as plt

#Author: Luke Tocco
#Date: 6/17/2024

#This code will determine the thickness needed for a thermal ablative subject to certain temperature conditions inside the Genesis rocket engine. It uses 1D Charring Material Ablative logic.
#The wall is broken up into discrete nodes(i=1,2,3,...N) and the temperature at each node is calculated based on the heat flux from the hot gases and the thermal properties of the material.
#Boundary conditions are supplied by data from RPA and known values.

#Material Constants (For Grade CE Canvas Phenolic)
rho_virgin = 1.38*10**3 #kg/m^3
rho_char = 800 #kg/m^3 
k_virgin = 0.30   #W/m-K
k_char = 0.15 #W/m-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
cp_virgin = 1500 #J/kg-K
cp_char = 1150 #J/kg-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
burn_time = 8 #seconds, approximate burn time of engine
delta_h_pyro = 5e5 #J/kg, heat of pyrolysis, PLACEHOLDER I DONT KNOW WHAT THIS IS

#Arrenhius Parameters for Charring Rate
B = 1e6 #PLACEHOLDER
E = 1.2e5 #PLACEHOLDER
m = 2 #2nd order reaction
R = 8.314 #J/mol-K, universal gas constant

#1D Geometry and Time Parameters Setup
L = 0.0127 #m (this is 0.5 inches, the conservative guess for a thickness of the phenolic wall)
N = 200
dx = L / (N-1)
dt = 0.001
time_steps = int(burn_time / dt)

#RPA Convection Coefficient Data (from the under the hood Bartz Correlaton calculations)
hg = 1346.9 #W/m^2-K
q_rad = 449.8813 * 1000 #W/m^2

#Setup Arrays
T_current = np.ones(N) * 300  #initial temperature array
T_next = np.zeros(N) #array to hold next time step temperatures
rho_current = np.ones(N) * rho_virgin
rho_next = np.zeros(N)

#Boundary Conditions
T_gas = 2494 #K, Adiabatic wall temperature...due to boundary layer effects, the temperature responsible for conduction is only a fraction of the total stagnation temperature (RPA)
T_current[-1] = 300 #K, ambient temperature at the outer surface

#Arrehenius Function
def drho_dt(rho, T):
    #Calculate the density change using Arrhenius equation
    if rho <= rho_char:
        return 0
    else:
       reaction_term = ((rho - rho_char) / rho_virgin)**m
       rhodot = -B * np.exp(-E / (R * T)) * rho_virgin * reaction_term

    return rhodot

def get_nodal_properties(rho_node):

    gamma = (rho_virgin - rho_node) / (rho_virgin - rho_char)

    gamma = np.clip(gamma, 0, 1) #0 means virgin, 1 means charred

    k_node = k_virgin * (1-gamma) + gamma * k_char

    cp_node = cp_virgin * (1-gamma) + gamma * cp_char

    alpha_node = k_node / (cp_node * rho_node)

    return k_node, cp_node, alpha_node

#Main Time Loop
for t in range(time_steps):
        
    #Get the nodal properties at the inner wall
    k_surface, cp_surface, alpha_surface = get_nodal_properties(rho_current[0])
    rhodot_surface = drho_dt(rho_current[0], T_current[0])
    pyro_sink_surface = (delta_h_pyro * abs(rhodot_surface)) / (rho_current[0] * cp_surface)

    #Calculate the heat coming in from the hot wall using RPA Convection Coefficient Data
    T_next[0] = T_current[0] + ((2)/(rho_current[0]*cp_surface*dx)) * (q_rad + hg * (T_gas - T_current[0]) - k_surface * (T_current[0] - T_current[1]) / dx - pyro_sink_surface) * dt

    rho_new = rho_current[0] + rhodot_surface * dt
    rho_next[0] = max(rho_char, rho_new)

    for i in range(1, N-1):

        #Get the nodal properties of the current node
        k_node, cp_node, alpha_node = get_nodal_properties(rho_current[i])
        rhodot = drho_dt(rho_current[i], T_current[i])
        pyro_sink = (delta_h_pyro * abs(rhodot)) / (rho_current[i] * cp_node)

        #Calculate the temperature at the next node for nodes 2-98
        T_next[i] = T_current[i] + (alpha_node * (T_current[i+1] - 2*T_current[i] + T_current[i-1]) / dx**2 - pyro_sink) * dt

        rho_new = (rho_current[i] + rhodot * dt)
        rho_next[i] = max(rho_char, rho_new)


    #Get the nodal properties at the outer wall
    k_outer, cp_outer, alpha_outer = get_nodal_properties(rho_current[-1])
    rhodot_outer = drho_dt(rho_current[-1], T_current[-1])
    pyro_sink_outer = (delta_h_pyro * abs(rhodot_outer)) / (rho_current[-1] * cp_outer)

    #Calculate the temp at the outer wall which is conducting heat to the outer air
    T_next[-1] = T_current[-1] + (2/(rho_current[-1]*cp_outer*dx)) * (k_outer * (T_current[-2] - T_current[-1]) / dx - pyro_sink_outer) * dt

    rho_new = (rho_current[-1] + rhodot_outer * dt)
    rho_next[-1] = max(rho_char, rho_new)

    #Update arrays
    T_current = T_next.copy()
    rho_current = rho_next.copy()

print("Inner wall temp:", T_current[0], "K")
print("Minimum density:", np.min(rho_current), "kg/m^3")

if t % 1000 == 0:
    print(T_current[0], np.min(rho_current))


#------Verification Plotting------#

rhodot_profile = np.zeros(N)

for i in range(N):
    rhodot_profile[i] = abs(
    drho_dt(rho_current[i], T_current[i])
    )

gamma_profile = (rho_virgin - rho_current) / ( rho_virgin - rho_char)
gamma_profile = np.clip(gamma_profile, 0, 1)

x_axis = np.linspace(0, L*1000, N)
fig, ax1 = plt.subplots(figsize=(10,5))

# Temperature plot
ax1.plot(x_axis, T_current, linewidth=2, label="Temperature")
ax1.set_xlabel("Distance from inner wall (mm)")
ax1.set_ylabel("Temperature (K)")
ax1.grid()

# Create second y-axis
ax2 = ax1.twinx()

# Density plot
ax2.plot(x_axis, rho_current, linewidth=2, linestyle='--', label="Density")
ax2.set_ylabel("Density (kg/m^3)")

#combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title("Temperature and Density Profiles")
plt.show()

# Plot the char fraction and pyrolysis rate profiles
plt.figure(figsize=(10,5))
plt.plot(x_axis, gamma_profile, linewidth=2)
plt.xlabel("Distance from inner wall (mm)")
plt.ylabel("Char Fraction")
plt.title("Char Fraction Profile")
plt.grid()

plt.figure(figsize=(10,5))
plt.plot(x_axis, rhodot_profile, linewidth=2)
plt.xlabel("Distance from inner wall (mm)")
plt.ylabel("Pyrolysis Rate (kg/m³-s)")
plt.title("Pyrolysis Rate Profile")
plt.grid()

plt.show()




