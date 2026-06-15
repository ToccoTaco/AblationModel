import numpy as np
import matplotlib.pyplot as plt

#Author: Luke Tocco

#--Material Constants (For Grade CE Canvas Phenolic) --
rho_virgin = 1.38*10**3 #kg/m^3
rho_char = 800 #kg/m^3 
k_virgin = 0.30   #W/m-K
k_char = 0.80 #W/m-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
cp_virgin = 1500 #J/kg-K
cp_char = 1100 #J/kg-K, PLACEHOLDER I DONT KNOW WHAT THIS IS
burn_time = 8 #seconds, approximate burn time of engine
delta_h_pyro = 2e6 #J/kg, heat of pyrolysis, PLACEHOLDER I DONT KNOW WHAT THIS IS
cp_pyrolysis_gas = 2500 #J/kg-K, specific heat of the pyrolysis gas, PLACEHOLDER I DONT KNOW WHAT THIS IS
B_prime_c = 0.08 #Approximate mass fraction of oxidizing species (O2, H20, CO2) in exhaust, currently an approximation but will be changed once RPA values are available.

#Arrenhius Parameters for Charring Rate
B = 3e7 #PLACEHOLDER
E = 1.4e5 #PLACEHOLDER
m = 2 #2nd order reaction
R = 8.314 #J/mol-K, universal gas constant

#1D Geometry and Time Parameters Setup
L_initial = 0.0127 #m (this is 0.5 inches, the conservative guess for a thickness of the phenolic wall)
L_current = L_initial
N = 300
dt = 0.0005
time_steps = int(burn_time / dt)

#Boundary Layer Data (Grabbed from RPA)
hg_unblown = 1346.9 #W/m^2-K
q_rad = 449.8813 * 1000 #W/m^2
T_gas = 2494 #K, Adiabatic wall temperature...due to boundary layer effects, the temperature responsible for conduction is only a fraction of the total stagnation temperature (RPA)

#Radiation constants
epsilon = 0.85 #emissivity of the surface of carbonaceous char, PLACEHOLDER
sigma = 5.67e-8 #Stefan-Boltzmann constant, W/m^2-K^4

#Setup SimulationArrays
T_current = np.ones(N) * 300  #initial temperature array
T_next = np.zeros(N) #array to hold next time step temperatures
rho_current = np.ones(N) * rho_virgin
rho_next = np.zeros(N)
rhodot_array = np.zeros(N) #array to hold the pyrolysis rate at each node for plotting

#History Tracking
mass_lost = 0.0 #Stores the integrated mass lost due to pyrolysis
mdot_history = [] # Stores the mass loss rate at each time step for plotting
mass_lost_history = [] # Stores the cumulative mass lost at each time step for plotting
time_history = []  # Stores the time at each time step for plotting
recession_depth = 0.0 #Stores the cumulative recession depth at each time step for plotting
recession_history = [] # Stores the recession depth at each time step for plotting
recession_rate_history = [] # Stores the recession rate at each time step for plotting

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

    L_current = L_initial - recession_depth
    dx_current = L_current / (N-1)

    #Pre calculate the density change for all nodes
    for i in range(N):
        rhodot_array[i] = drho_dt(rho_current[i], T_current[i], dt)

# Calculate cumulative gas mass flux (mdot_g) flowing from back to front
    mdot_g = np.zeros(N) #initalize array to hold the cumulative gas mass flux

    # The back wall only has its own generated gas flowing through it
    mdot_g[-1] = abs(rhodot_array[-1]) * dx_current
    
    # Sweep from the back wall (N-2) towards the hot inner wall (0)
    for i in range(N-2, -1, -1):
        mdot_g[i] = mdot_g[i+1] + abs(rhodot_array[i]) * dx_current
        
    mdot_total = mdot_g[0] # The total gas leaving the surface is the cumulative flow at node 0

    #-------Surface Node 0----------

    #Get the nodal properties at the inner wall
    k_surface, cp_surface, alpha_surface = get_nodal_properties(rho_current[0])
    rhodot_surface = rhodot_array[0] #drho_dt(rho_current[0], T_current[0], dt)

    #Thermochemical Recession
    #Calculate how much char mass is oxidized into gas

    mdot_char_oxidation = (hg_unblown / cp_pyrolysis_gas) * B_prime_c
    recession_rate_surface = mdot_char_oxidation / rho_char
    recession_depth += recession_rate_surface * dt

    #Add the oxidized char mass to the toal gass blowing into the boundary layer
    mdot_total += mdot_char_oxidation #We add this to the mdot total because the carbon turning into gas adds mass to the boundary layer and improves the blowing correction!


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

    pyro_sink_vol_surface = (delta_h_pyro * abs(rhodot_surface)) #K/s
    conduction_out_surface = k_surface * (T_current[0] - T_current[1]) / dx_current #W/m^2

    #Calculate the heat coming in from the hot wall using RPA Convection Coefficient Data
    dT0 = (2.0 / (rho_current[0] * cp_surface * dx_current)) * (
        q_rad
        + hg_blown * (T_gas - T_current[0])
        - q_rerad
        - conduction_out_surface
        - pyro_sink_vol_surface * dx_current / 2.0   # half-cell volume
    ) * dt
    T_next[0] = max(300.0, T_current[0] + dT0)
    
    rho_next[0] = max(rho_char, rho_current[0] + rhodot_surface * dt)

    #------Interior Nodes 1...N-2-----------

    for i in range(1, N-1):

        #Get the nodal properties of the current node
        k_node, cp_node, alpha_node = get_nodal_properties(rho_current[i])

        k_im1, _, _ = get_nodal_properties(rho_current[i-1])
        k_ip1, _, _ = get_nodal_properties(rho_current[i+1])

        k_left = 0.5 * (k_node + k_im1)
        k_right = 0.5 * (k_node + k_ip1)

        #Conduction term
        conduction_term = (k_right * (T_current[i+1] - T_current[i]) - k_left * (T_current[i] - T_current[i-1])) / dx_current**2

        #Pyrolysis sink term
        rhodot = rhodot_array[i] #drho_dt(rho_current[i], T_current[i], dt)
        pyro_sink = (delta_h_pyro * abs(rhodot)) / (rho_current[i] * cp_node)

        #Transpiration Term
        #Energy absorbed by gas flowing from i+1 to i
        gas_convection = (mdot_g[i] * cp_pyrolysis_gas * (T_current[i+1] - T_current[i]) / dx_current) / (rho_current[i] * cp_node)

        #Landau Transformation grid velocity correction
        eta = i / (N-1)
        dT_dx = (T_current[i+1] - T_current[i-1]) / (2 * dx_current)
        grid_velocity_term = recession_rate_surface * (1.0 - eta) * dT_dx

        #Calculate the temperature at the next node for nodes 2-98
        T_dot = (conduction_term / (rho_current[i] * cp_node) - pyro_sink + gas_convection + grid_velocity_term)
        T_next[i] = max(300.0, T_current[i] + T_dot * dt)
 
        rho_next[i] = max(rho_char, rho_current[i] + rhodot * dt)

    #------Node N-1 Outer Wall----------

    # Get the nodal properties at the outer wall
    k_outer, cp_outer, alpha_outer = get_nodal_properties(rho_current[-1])
    rhodot_outer = rhodot_array[-1] #drho_dt(rho_current[-1], T_current[-1])
    pyro_sink_vol_outer = (delta_h_pyro * abs(rhodot_outer))

    #Calculate the temp at the outer wall which is conducting heat to the outer air
    dT_outer = (2.0 / (rho_current[-1] * cp_outer * dx_current)) * (
        k_outer * (T_current[-2] - T_current[-1]) / dx_current
        - pyro_sink_vol_outer * dx_current / 2.0
    ) * dt

    T_next[-1] = max(300.0, T_current[-1] + dT_outer)
 
    rho_next[-1] = max(rho_char, rho_current[-1] + rhodot_outer * dt)

    #Update mass loss tracking
    mass_lost += mdot_total * dt
    mdot_history.append(mdot_total)
    mass_lost_history.append(mass_lost)
    time_history.append(t * dt)
    recession_history.append(recession_depth)
    recession_rate_history.append(recession_rate_surface)
    
    #Update arrays
    T_current = T_next.copy()
    rho_current = rho_next.copy()


# ── Final diagnostics ─────────────────────────────────────────────────────────
print("=" * 60)
print("ENGINE GENESIS – 1-D ABLATION SIMULATION (PATCHED)")
print("=" * 60)
print(f"Inner wall temperature  : {T_current[0]:.1f} K")
print(f"Minimum density         : {np.min(rho_current):.1f} kg/m³")
print(f"Total mass outgassed    : {mass_lost:.4f} kg/m²")
print(f"Final surface gas flux  : {mdot_total:.5f} kg/m²-s")
print(f"Total surface recession : {recession_depth*1000:.3f} mm")
print(f"Remaining wall thickness: {(L_initial - recession_depth)*1000:.3f} mm")
print("=" * 60)
 
# ── Spatial profiles at end of burn ───────────────────────────────────────────
rhodot_profile = np.array([abs(drho_dt(rho_current[i], T_current[i], dt)) for i in range(N)])
gamma_profile  = np.clip((rho_virgin - rho_current) / (rho_virgin - rho_char), 0, 1)
dx_final = (L_initial - recession_depth) / (N - 1)
x_axis   = np.linspace(0, (L_initial - recession_depth) * 1000, N)   # mm
 
# ── Dashboard Figure ──────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(
    "Engine Genesis — 1-D Transient Ablation Dashboard (8 s Burn)\n",
    fontsize=15, fontweight="bold", y=0.96,
)
 
# 1: Temperature & Density
ax1 = axs[0, 0]
ax1.plot(x_axis, T_current, color="crimson", lw=2, label="Temperature")
ax1.set_xlabel("Distance from Hot Wall (mm)")
ax1.set_ylabel("Temperature (K)", color="crimson")
ax1.tick_params(axis="y", labelcolor="crimson")
ax1.grid(True, ls=":", alpha=0.6)
ax1_twin = ax1.twinx()
ax1_twin.plot(x_axis, rho_current, color="navy", lw=2, ls="--", label="Density")
ax1_twin.set_ylabel("Density (kg/m³)", color="navy")
ax1_twin.tick_params(axis="y", labelcolor="navy")
ax1.set_title("Final Temperature & Density State")
 
# 2: Char fraction
axs[0, 1].plot(x_axis, gamma_profile, color="darkorange", lw=2)
axs[0, 1].set_xlabel("Distance from Hot Wall (mm)")
axs[0, 1].set_ylabel("Char Fraction (0=Virgin, 1=Char)")
axs[0, 1].set_title("Final Material Degradation State")
axs[0, 1].grid(True, ls=":", alpha=0.6)
 
# 3: Pyrolysis rate
axs[1, 0].plot(x_axis, rhodot_profile, color="purple", lw=2)
axs[1, 0].set_xlabel("Distance from Hot Wall (mm)")
axs[1, 0].set_ylabel("Pyrolysis Rate (kg/m³-s)")
axs[1, 0].set_title("Final Volatile Reaction Zone")
axs[1, 0].grid(True, ls=":", alpha=0.6)
 
# 4: Cumulative mass lost
axs[1, 1].plot(time_history, mass_lost_history, color="forestgreen", lw=2)
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Mass Outflow (kg/m²)")
axs[1, 1].set_title("Total Outgassing Mass Over Time")
axs[1, 1].grid(True, ls=":", alpha=0.6)
 
# 5: Gas flux history
axs[2, 0].plot(time_history, mdot_history, color="teal", lw=2)
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Gas Flux at Surface (kg/m²-s)")
axs[2, 0].set_title("Boundary-Layer Blowing Mass Flux")
axs[2, 0].grid(True, ls=":", alpha=0.6)
 
# 6: Recession depth & rate
ax6 = axs[2, 1]
ax6.plot(time_history, np.array(recession_history) * 1000, color="black", lw=2)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Recession Depth (mm)", color="black")
ax6.grid(True, ls=":", alpha=0.6)
ax6_twin = ax6.twinx()
ax6_twin.plot(time_history, np.array(recession_rate_history) * 1000,
              color="royalblue", lw=1.5, ls=":")
ax6_twin.set_ylabel("Erosion Rate (mm/s)", color="royalblue")
ax6_twin.tick_params(axis="y", labelcolor="royalblue")
ax6.set_title("Surface Ablation History")
 
plt.tight_layout()
fig.subplots_adjust(hspace=0.45, wspace=0.30, top=0.91, bottom=0.05)
plt.show()
 