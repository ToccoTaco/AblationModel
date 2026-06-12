import math

#This code calculates the thrust, exit velocity, and mass flow rate of the Cold Gas thryuster and checks if its feasible based on the mass flow rate that is needed, and the mass flow rate
#that the lines can provide. It also checks the Mach number in the lines to see if there will be significant pressure losses.

#Author: Luke Tocco
#Date : 6/17/2024

def run_propulsion_analysis():
    
    # 1. INPUTS
    p_tank_psi = 100  # Tank Pressure in PSI (Gauge)
    p_amb_psi = 14.7 # Ambient Pressure in PSI 
    
    #throat and line dimensions
    d_throat_in = 0.40  # Throat Diameter in Inches
    d_line_in = 0.492 #3/8" NPT Inner Diameter in inches
    
    gamma = 1.4
    R = 287 
    temp_k = 293
    
    #conversions
    p_c_pa = (p_tank_psi + p_amb_psi) * 6894.76
    p_amb_pa = p_amb_psi * 6894.76
    area_throat = math.pi * ((d_throat_in * 0.0254) / 2)**2
    area_line = math.pi * ((d_line_in * 0.0254) / 2)**2
    
    # mass flow calculations for demand
    # the Choked Flow constant for Air is ~0.0404
    choked_const = math.sqrt(gamma / R * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))
    m_dot_demand = (area_throat * p_c_pa / math.sqrt(temp_k)) * choked_const
    
    # exit velocity calculation
    v_exit = math.sqrt(
        ((2 * gamma * R * temp_k) / (gamma - 1)) * (1 - (p_amb_pa / p_c_pa)**((gamma - 1) / gamma))
    )
    
    #thrust calculation
    thrust_newtons = m_dot_demand * v_exit
    thrust_lbs = thrust_newtons / 4.448
    
    # Max Supply of the line (if the line itself choked)
    m_dot_supply_max = (area_line * p_c_pa / math.sqrt(temp_k)) * choked_const
    
    # Calculate Line Mach Number based on Area Ratio
    area_ratio = area_line / area_throat

    # Simplified Mach estimation for subsonic flow in lines

    line_mach = 0
    if area_ratio > 1:
      
        line_mach = 1 / area_ratio #linear approximation for Mach < 0.5
    else:
        line_mach = 1.0 #if this then choked

    #outputs
    print("\n" + "="*30)
    print(f"RESULTS FOR {d_throat_in}\" THROAT @ {p_tank_psi} PSI")
    print("="*30)
    print(f"Mass Flow Rate:    {m_dot_demand:.4f} kg/s")
    print(f"Exit Velocity:      {v_exit:.1f} m/s ({v_exit*2.237:.1f} mph)")
    print(f"Calculated Thrust:  {thrust_lbs:.2f} lbs")
    
    print("\n--- Plumbing Check ---")
    print(f"Line Area Ratio:    {area_ratio:.2f}")
    print(f"Est. Line Mach:     {line_mach:.2f}")
    
    if line_mach > 0.4:
        print("WARNING: Air speed in lines is high (> Mach 0.4). Expect significant pressure drop/losses.")
    elif line_mach > 1.0 or area_ratio < 1.0:
        print("CRITICAL: Line is smaller than throat! The line will choke and kill your thrust.")
    else:
        print("Plumbing Status:  GOOD. Line size is sufficient for this throat.")
    print("="*30)

if __name__ == "__main__":
    run_propulsion_analysis()