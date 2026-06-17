"""
semi_infinite_benchmark.py
============================
Validates the conduction solver in ablation.py against the analytic
semi-infinite solid solution. Pyrolysis, blowing, recession, and
radiation are all disabled, effectively isolating the pure conduction scheme.

Analytic solution (constant flux q applied at x=0, semi-infinite solid,
initial temperature T0):

    T(x,t) = T0 + (2q/k)*sqrt(alpha*t/pi)*exp(-x^2/(4*alpha*t))
                - (q*x/k)*erfc(x/sqrt(4*alpha*t))

Author: Luke Tocco w/ plots by Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#properties
k_const   = 0.33      # W/m-K
rho_const = 1380.0    # kg/m^3
cp_const  = 1550.0    # J/kg-K
alpha     = k_const / (rho_const * cp_const)

q_flux = 449881.3     # W/m^2  (representative constant flux, no blowing)
T0     = 300.0        # K, initial uniform temperature
L      = 0.0050       # m, virtual test domain sized so x=L/4 has a meaningful temperature rise.
N      = 300
dx     = L / (N - 1)

Fo_target = 0.25
dt = Fo_target * dx**2 / alpha
print(f"alpha = {alpha:.4e} m^2/s")
print(f"dx    = {dx*1000:.4f} mm")
print(f"dt    = {dt:.3e} s   (Fo = {alpha*dt/dx**2:.3f})")

test_times = [1.0, 2.0]   # seconds 
x_axis = np.linspace(0, L, N)

# Analytic solution
def T_analytic(x, t, q, k_, alpha_, T0_):
    term1 = (2 * q / k_) * np.sqrt(alpha_ * t / np.pi) * np.exp(-x**2 / (4*alpha_*t))
    term2 = (q * x / k_) * erfc(x / np.sqrt(4*alpha_*t))
    return T0_ + term1 - term2

# Numerical solver
T_max_time = max(test_times)
n_steps = int(T_max_time / dt)

T = np.ones(N) * T0
T_history = {}  

step_targets = {int(t/dt): t for t in test_times}

for step in range(n_steps + 1):
    if step in step_targets:
        T_history[step_targets[step]] = T.copy()

    T_next = np.zeros(N)

    # Surface node (x=0)
    cond_out = k_const * (T[0] - T[1]) / dx
    dT0 = (2.0 / (rho_const * cp_const * dx)) * (q_flux - cond_out) * dt
    T_next[0] = T[0] + dT0

    # Interior nodes
    T_next[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])

    # Back wall (x=L)
    T_next[-1] = T[-1] + alpha * dt / dx**2 * 2 * (T[-2] - T[-1])

    T = T_next

# Comparison
print()
print("SEMI-INFINITE SOLID BENCHMARK — CONDUCTION SOLVER VALIDATION")

results = []
for t_test in test_times:
    T_num = T_history[t_test]

    x_check = [0.0, L/4]
    print(f"\n--- t = {t_test:.1f} s ---")
    print(f"{'x (mm)':>8}  {'T_numeric (K)':>14}  {'T_analytic (K)':>15}  {'Error (%)':>10}")
    for xc in x_check:
        idx = int(round(xc / dx))
        idx = min(idx, N-1)
        T_num_pt = T_num[idx]
        T_ana_pt = T_analytic(x_axis[idx], t_test, q_flux, k_const, alpha, T0)
        err_pct = abs(T_num_pt - T_ana_pt) / (T_ana_pt - T0) * 100
        print(f"{xc*1000:>8.3f}  {T_num_pt:>14.3f}  {T_ana_pt:>15.3f}  {err_pct:>10.4f}")
        results.append((t_test, xc, T_num_pt, T_ana_pt, err_pct))

#Plots

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Semi-Infinite Solid Benchmark — Numerical vs Analytic",
             fontsize=13, fontweight='bold')

colors = ['crimson', 'steelblue']
for t_test, color in zip(test_times, colors):
    T_num = T_history[t_test]
    T_ana = T_analytic(x_axis, t_test, q_flux, k_const, alpha, T0)

    axs[0].plot(x_axis*1000, T_num, color=color, lw=2.5,
                label=f'Numeric  t={t_test:.1f}s')
    axs[0].plot(x_axis*1000, T_ana, color=color, lw=1.5, ls='--',
                label=f'Analytic t={t_test:.1f}s')

    axs[1].plot(x_axis*1000, np.abs(T_num - T_ana), color=color, lw=2,
                label=f't={t_test:.1f}s')

axs[0].set_xlabel('Distance from hot face (mm)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature Profiles')
axs[0].set_xlim(0, 4)   # zoom into the thermally active region
axs[0].legend(fontsize=8)
axs[0].grid(True, ls=':', alpha=0.6)

axs[1].set_xlabel('Distance from hot face (mm)')
axs[1].set_ylabel('|T_numeric - T_analytic|  (K)')
axs[1].set_title('Absolute Error')
axs[1].set_xlim(0, 4)
axs[1].legend(fontsize=9)
axs[1].grid(True, ls=':', alpha=0.6)

plt.tight_layout()
plt.show()