"""
Cantera-based pyrolysis gas properties.
 
Output:
    Prints the cp_g_T and cp_g_val arrays for ablation.py Also generates a plot of cp(T).

"""

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

#Elemental composition of the canvas phenolic pyrolysis gas

C_mass = 0.567
H_mass = 0.060
O_mass = 0.373

#Convert to molar ratios
C_mol = C_mass / 12.011
H_mol = H_mass / 1.008
O_mol = O_mass / 15.999
total_mol = C_mol + H_mol + O_mol

#Normalize the molar ratios
C_mf = C_mol / total_mol
H_mf = H_mol / total_mol
O_mf = O_mol / total_mol

print("CANVAS PHENOLIC PYROLYSIS GAS — CANTERA EQUILIBRIUM")
print(f"Input mass fractions:  C={C_mass:.3f}  H={H_mass:.3f}  O={O_mass:.3f}")
print(f"Element mole fracs:    C={C_mf:.4f}  H={H_mf:.4f}  O={O_mf:.4f}")
print(f"Molar C:H:O ratio:     1 : {H_mol/C_mol:.3f} : {O_mol/C_mol:.3f}")

#Solve system to get a starting species
a = O_mf / 2.0                  # CO2 moles
b = (C_mf - a) / 2.0            # C2H2 moles
c = (H_mf - 2.0*b) / 2.0       # H2 moles
 
total_sp = a + b + c
X_start = {
    'CO2':  a / total_sp,
    'C2H2': b / total_sp,
    'H2':   c / total_sp,
}
 
print(f"\nStarting species (before equilibration):")
for sp, x in X_start.items():
    print(f"  {sp:6s}: {x:.4f}")

gas = ct.Solution('gri30.yaml')
 
# Verify element balance
gas.TPX = 1500, 101325, X_start   # arbitrary T and P just to check
C_check = gas.elemental_mole_fraction('C')
H_check = gas.elemental_mole_fraction('H')
O_check = gas.elemental_mole_fraction('O')
 
print(f"\nElement balance verification:")
print(f"  C: got {C_check:.4f}  target {C_mf:.4f}  error {abs(C_check-C_mf)/C_mf*100:.3f}%")
print(f"  H: got {H_check:.4f}  target {H_mf:.4f}  error {abs(H_check-H_mf)/H_mf*100:.3f}%")
print(f"  O: got {O_check:.4f}  target {O_mf:.4f}  error {abs(O_check-O_mf)/O_mf*100:.3f}%")

#Solver Setup

P_psi     = 400.0
P_Pa      = P_psi * 6894.76       # convert psi to Pa
P_atm     = P_psi / 14.696        # 27.22 atm
 
T_sweep = np.array([300, 400, 500, 600, 700, 800, 900,
                    1000, 1100, 1200, 1400, 1600, 1800,
                    2000, 2200, 2400])
 
print(f"\nPressure: {P_psi:.0f} psi = {P_atm:.2f} atm")
print(f"\nRunning equilibrium at {len(T_sweep)} temperature points...\n")
 
print(f"{'T(K)':>6}  {'cp(J/kgK)':>10}  {'h(kJ/kg)':>10}  "
      f"{'MW(g/mol)':>10}  Dominant species")
print("-" * 85)
 
cp_vals = []
h_vals  = []
 
for T in T_sweep:
    # Set state with starting composition, then equilibrate at fixed T and P
    gas.TPX = T, P_Pa, X_start
    gas.equilibrate('TP', solver='auto')
 
    cp = gas.cp_mass            # J/kg/K
    h  = gas.enthalpy_mass      # J/kg
    MW = gas.mean_molecular_weight
 
    cp_vals.append(cp)
    h_vals.append(h)
 
    # Print top 4 species by mole fraction
    X = gas.X
    top_idx = np.argsort(X)[-4:][::-1]
    top_str = '  '.join([
        f"{gas.species_names[i]}:{X[i]:.3f}"
        for i in top_idx if X[i] > 0.005
    ])
 
    print(f"{T:>6}  {cp:>10.1f}  {h/1000:>10.1f}  {MW:>10.3f}  {top_str}")
 
#Output
 
print()
print("=" * 65)
print("=" * 65)
print()
print("# Pyrolysis gas cp(T) — Cantera GRI30 equilibrium")
print(f"# Canvas phenolic Grade CE, P = {P_psi:.0f} psi ({P_atm:.2f} atm)")
print(f"# Elemental C:H:O = 1:{H_mol/C_mol:.3f}:{O_mol/C_mol:.3f} (molar)")
print(f"# Mass fractions: C={C_mass:.3f}  H={H_mass:.3f}  O={O_mass:.3f}")
print(f"# Source: Cantera {ct.__version__}, GRI30 thermodynamic database (JANAF)")
print()
print("cp_g_T   = np.array([300, 400, 500, 600, 700, 800, 900,")
print("                     1000, 1100, 1200, 1400, 1600, 1800,")
print("                     2000, 2200, 2400])")
print()
vals_str = ', '.join(f'{v:.0f}' for v in cp_vals)
print(f"cp_g_val = np.array([{vals_str}])")
print()
print("def cp_g(T):")
print("    \"\"\"")
print("    Specific heat of canvas phenolic pyrolysis gas [J/kg/K]")
print("    at chamber pressure (400 psi), from Cantera GRI30 equilibrium.")
print("    \"\"\"")
print("    return np.interp(T, cp_g_T, cp_g_val)")
 
 # plotting

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Canvas Phenolic Pyrolysis Gas — Cantera GRI30 Equilibrium\n"
    f"P = {P_psi:.0f} psi ({P_atm:.1f} atm)  |  "
    f"C:H:O = 1:{H_mol/C_mol:.2f}:{O_mol/C_mol:.2f} (molar)",
    fontsize=12, fontweight='bold'
)
 
# cp(T)
ax1 = axes[0]
ax1.plot(T_sweep, cp_vals, 'o-', color='teal', lw=2, ms=7, label='Cantera GRI30')
ax1.axhline(3200, color='gray', lw=1.5, ls='--', label='Old constant (3200)')
ax1.fill_between(T_sweep, cp_vals, 3200, alpha=0.12, color='red',
                 label='Error from old constant')
ax1.set_xlabel('Temperature (K)', fontsize=11)
ax1.set_ylabel('cp_g  (J/kg/K)', fontsize=11)
ax1.set_title('Pyrolysis Gas Specific Heat')
ax1.legend(fontsize=9)
ax1.grid(True, ls=':', alpha=0.6)
 
# Annotate zones
ax1.axvspan(300, 700,  alpha=0.05, color='blue')
ax1.axvspan(700, 1200, alpha=0.05, color='green')
ax1.axvspan(1200, 2400, alpha=0.05, color='red')
ax1.text(500,  1200, 'Rxn A\nzone', ha='center', fontsize=8, color='blue')
ax1.text(950,  1200, 'Rxn B\nzone', ha='center', fontsize=8, color='darkgreen')
ax1.text(1800, 1200, 'Char\nzone',  ha='center', fontsize=8, color='darkred')
 
# h(T)
ax2 = axes[1]
ax2.plot(T_sweep, np.array(h_vals)/1000, 's-', color='crimson', lw=2, ms=7)
ax2.set_xlabel('Temperature (K)', fontsize=11)
ax2.set_ylabel('h_g  (kJ/kg)', fontsize=11)
ax2.set_title('Pyrolysis Gas Enthalpy')
ax2.grid(True, ls=':', alpha=0.6)
ax2.axhline(0, color='black', lw=0.8, ls=':')
 
plt.tight_layout()
plt.show()