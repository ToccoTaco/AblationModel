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
k_char = 0.80 #W/m-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
cp_virgin = 1500 #J/kg-K
cp_char = 1100 #J/kg-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
burn_time = 8 #seconds, approximate burn time of engine
delta_h_pyro = 2e6 #J/kg, heat of pyrolysis, PLACEHOLDER I DONT KNOW WHAT THIS IS
cp_pyrolysis_gas = 2500 #J/kg-K, specific heat of the pyrolysis gas, PLACEHOLDER I DONT KNOW WHAT THIS IS

#Arrenhius Parameters for Charring Rate
B = 3e7 #PLACEHOLDER
E = 1.4e5 #PLACEHOLDER
m = 2 #2nd order reaction
R = 8.314 #J/mol-K, universal gas constant

#1D Geometry and Time Parameters Setup
L = 0.0127 #m (this is 0.5 inches, the conservative guess for a thickness of the phenolic wall)
N = 300
dx = L / (N-1)
dt = 0.0005
time_steps = int(burn_time / dt)

#RPA Convection Coefficient Data (from the under the hood Bartz Correlaton calculations)
hg_unblown = 1346.9 #W/m^2-K
q_rad = 449.8813 * 1000 #W/m^2

#Radiation constants
epsilon = 0.85 #emissivity of the surface of carbonaceous char, PLACEHOLDER
sigma = 5.67e-8 #Stefan-Boltzmann constant, W/m^2-K^4


#Setup Arrays
T_current = np.ones(N) * 300  #initial temperature array
T_next = np.zeros(N) #array to hold next time step temperatures
rho_current = np.ones(N) * rho_virgin
rho_next = np.zeros(N)
rhodot_array = np.zeros(N) #array to hold the pyrolysis rate at each node for plotting

#Mass loss tracking setup
mass_lost = 0.0 #Stores the integrated mass lost due to pyrolysis
mdot_history = [] # Stores the mass loss rate at each time step for plotting
mass_lost_history = [] # Stores the cumulative mass lost at each time step for plotting
time_history = []  # Stores the time at each time step for plotting

#recession tracking setup
recession_depth = 0.0
recession_history = []
recession_rate_history = []


#Boundary Conditions
T_gas = 2494 #K, Adiabatic wall temperature...due to boundary layer effects, the temperature responsible for conduction is only a fraction of the total stagnation temperature (RPA)
T_current[-1] = 300 #K, ambient temperature at the outer surface

#Arrehenius Function
def drho_dt(rho, T, dt):
    #Calculate the density change using Arrhenius equation
    if rho <= rho_char:
        return 0
    else:
       reaction_term = ((rho - rho_char) / rho_virgin)**m
       rhodot = -B * np.exp(-E / (R * T)) * rho_virgin * reaction_term

       max_rhodot = -(rho - rho_char) / dt 
        
       # Clip the rate (both are negative, so we take the max, meaning the closest to zero)
       rhodot = max(rhodot, max_rhodot)

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

    #Pre calculate the density change for all nodes
    for i in range(N):
        rhodot_array[i] = drho_dt(rho_current[i], T_current[i], dt)

# Calculate cumulative gas mass flux (mdot_g) flowing from back to front
    mdot_g = np.zeros(N)
    # The back wall only has its own generated gas flowing through it
    mdot_g[-1] = abs(rhodot_array[-1]) * dx 
    
    # Sweep from the back wall (N-2) towards the hot inner wall (0)
    for i in range(N-2, -1, -1):
        mdot_g[i] = mdot_g[i+1] + abs(rhodot_array[i]) * dx
        
    mdot_total = mdot_g[0] # The total gas leaving the surface is the cumulative flow at node 0
        
    #Get the nodal properties at the inner wall
    k_surface, cp_surface, alpha_surface = get_nodal_properties(rho_current[0])
    rhodot_surface = rhodot_array[0] #drho_dt(rho_current[0], T_current[0], dt)

    mdot_surface = abs(rhodot_surface) * dx
    recession_rate_surface = mdot_surface / rho_current[0]  # Recession rate in m/s
    recession_depth += recession_rate_surface * dt  # Update recession depth

    pyro_sink_surface = (delta_h_pyro * abs(rhodot_surface)) / (rho_current[0] * cp_surface)

    #Calculate the blowing parameter Bc
    Bc = (mdot_total *cp_pyrolysis_gas) / hg_unblown

    # Apply the blowing correction to hg (+protection against division by zero)
    if Bc > 0.001:
        hg_blown = hg_unblown * (np.log(1 + Bc) / Bc)
    else:
        hg_blown = hg_unblown

    #Surface re-radiation term
    # Calculate heat radiating away from the glowing char
    q_rerad = epsilon * sigma * (T_current[0]**4)

    #Calculate the heat coming in from the hot wall using RPA Convection Coefficient Data
    T_next[0] = T_current[0] + ((2)/(rho_current[0]*cp_surface*dx)) * (q_rad + hg_blown * (T_gas - T_current[0]) - q_rerad - k_surface * (T_current[0] - T_current[1]) / dx - pyro_sink_surface) * dt
    T_next[0] = max(300, T_next[0]) #prevents numerical instability

    rho_new = rho_current[0] + rhodot_surface * dt
    rho_next[0] = max(rho_char, rho_new)

    for i in range(1, N-1):

        #Get the nodal properties of the current node
        k_node, cp_node, alpha_node = get_nodal_properties(rho_current[i])

        k_im1, _, _ = get_nodal_properties(rho_current[i-1])
        k_ip1, _, _ = get_nodal_properties(rho_current[i+1])

        k_left = 0.5 * (k_node + k_im1)
        k_right = 0.5 * (k_node + k_ip1)

        #Conduction term
        conduction_term = (k_right * (T_current[i+1] - T_current[i]) - k_left * (T_current[i] - T_current[i-1])) / dx**2

        #Pyrolysis sink term
        rhodot = rhodot_array[i] #drho_dt(rho_current[i], T_current[i], dt)
        pyro_sink = (delta_h_pyro * abs(rhodot)) / (rho_current[i] * cp_node)

        #Transpiration Term
        #Energy absorbed by gas flowing from i+1 to i
        gas_convection = (mdot_g[i] * cp_pyrolysis_gas * (T_current[i+1] - T_current[i]) / dx) / (rho_current[i] * cp_node)

        #Calculate the temperature at the next node for nodes 2-98
        T_dot = (conduction_term / (rho_current[i] * cp_node) - pyro_sink + gas_convection)
        T_next[i] = T_current[i] + T_dot * dt
        T_next[i] = max(300, T_next[i]) #prevents numerical instability

        rho_new = (rho_current[i] + rhodot * dt)
        rho_next[i] = max(rho_char, rho_new)


    #Get the nodal properties at the outer wall
    k_outer, cp_outer, alpha_outer = get_nodal_properties(rho_current[-1])
    rhodot_outer = rhodot_array[-1] #drho_dt(rho_current[-1], T_current[-1])
    pyro_sink_outer = (delta_h_pyro * abs(rhodot_outer)) / (rho_current[-1] * cp_outer)

    #Calculate the temp at the outer wall which is conducting heat to the outer air
    T_next[-1] = T_current[-1] + (2/(rho_current[-1]*cp_outer*dx)) * (k_outer * (T_current[-2] - T_current[-1]) / dx - pyro_sink_outer) * dt
    T_next[-1] = max(300, T_next[-1]) #prevents numerical instability

    rho_new = (rho_current[-1] + rhodot_outer * dt)
    rho_next[-1] = max(rho_char, rho_new)

    #Update mass loss tracking
    mass_lost += mdot_total * dt
    mdot_history.append(mdot_total)
    mass_lost_history.append(mass_lost)
    time_history.append(t * dt)

    #update recession tracking
    recession_history.append(recession_depth)
    recession_rate_history.append(mdot_total / rho_current[0])
    
    #Update arrays
    T_current = T_next.copy()
    rho_current = rho_next.copy()

print("Inner wall temp:", T_current[0], "K")
print("Minimum density:", np.min(rho_current), "kg/m^3")
print("Total mass lost:", mass_lost, "kg/m^2")
print("Final gas generation rate:", mdot_total, "kg/m^2-s")
print("Total recession:", recession_depth*1000, "mm")

if t % 1000 == 0:
    print(T_current[0], np.min(rho_current))


#------Verification Plotting------#

rhodot_profile = np.zeros(N)

for i in range(N):
    rhodot_profile[i] = abs(
    drho_dt(rho_current[i], T_current[i], dt)
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

# Plot mass loss history
plt.figure(figsize=(10,5))
plt.plot(time_history, mass_lost_history, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Mass Lost (kg/m²)")
plt.title("Cumulative Mass Loss")
plt.grid()
plt.show()

#Plot gas generation rate
plt.figure(figsize=(10,5))
plt.plot(time_history, mdot_history, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Gas Generation Rate (kg/m²-s)")
plt.title("Pyrolysis Gas Generation Rate")
plt.grid()
plt.show()

#Plot recession depth vs. time
plt.figure(figsize=(10,5))
plt.plot(time_history, recession_history, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Recession Depth (m)")
plt.title("Recession Depth Over Time")
plt.grid()
plt.show()

#Plot recession rate vs. time
plt.figure(figsize=(10,5))
plt.plot(time_history, recession_rate_history, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Recession Rate (m/s)")
plt.title("Recession Rate Over Time")
plt.grid()
plt.show()




