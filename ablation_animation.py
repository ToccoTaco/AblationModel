"""
ablation_animation.py  –  Engine Genesis: Real-Time Ablation Visualiser
========================================================================
Standalone animation of the 1-D transient ablation process.
Runs the full simulation internally (no dependency on ablation.py) then
plays back a high-fidelity animated cross-section with:
  • Eroding left boundary (Landau grid compression)
  • True-colour heat map (300 K → 2000+ K)
  • Density / char-fraction profile
  • Pyrolysis gas velocity (outgassing flux) overlay
  • Live telemetry panel
"""

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")   # interactive on most desktops
    import tkinter  # noqa: probe whether Tk is available
except Exception:
    matplotlib.use("Agg")     # headless fallback — saves to file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 1.  SIMULATION  (identical physics to patched ablation.py)
# ═══════════════════════════════════════════════════════════
print("Running pre-simulation …", flush=True)

# Material constants
rho_virgin  = 1.38e3;  rho_char = 800.0
k_virgin    = 0.30;    k_char   = 1.05
cp_virgin   = 1500.0;  cp_char  = 1400.0
delta_h_pyro     = 2.3e6
cp_pyrolysis_gas = 3200.0
B_prime_c   = 0.08

B = 3.0e9;  E = 1.4e5;  m_ord = 2;  R = 8.314
L_initial   = 0.0127
N = 300;  dt = 0.0005;  burn_time = 8.0
time_steps  = int(burn_time / dt)

hg_unblown = 1346.9;  q_rad = 449.8813e3;  T_gas = 2494.0
epsilon = 0.85;  sigma = 5.67e-8

def drho_dt(rho, T, dt_):
    if rho <= rho_char: return 0.0
    rt = ((rho - rho_char) / rho_virgin) ** m_ord
    rdot = -B * np.exp(-E / (R * T)) * rho_virgin * rt
    return max(rdot, -(rho - rho_char) / dt_)

def nodal_props(rho_n):
    g = np.clip((rho_virgin - rho_n) / (rho_virgin - rho_char), 0., 1.)
    k  = k_virgin  * (1-g) + g * k_char
    cp = cp_virgin * (1-g) + g * cp_char
    return k, cp

# ── Simulation storage (sub-sampled every 50 steps ≈ 160 frames @ 8 s) ──────
STRIDE = 50
n_frames = time_steps // STRIDE

T_hist   = np.zeros((n_frames, N))
rho_hist = np.zeros((n_frames, N))
mdot_g_hist  = np.zeros((n_frames, N))
L_hist   = np.zeros(n_frames)
recession_hist = np.zeros(n_frames)
mdot_blowing_hist = np.zeros(n_frames)
Tw_hist  = np.zeros(n_frames)

T_cur   = np.ones(N) * 300.0
rho_cur = np.ones(N) * rho_virgin
T_nxt   = np.zeros(N)
rho_nxt = np.zeros(N)
rdot    = np.zeros(N)
recession_depth = 0.0
frame_idx = 0

for t in range(time_steps):
    L_cur  = L_initial - recession_depth
    dx     = L_cur / (N - 1)

    for i in range(N):
        rdot[i] = drho_dt(rho_cur[i], T_cur[i], dt)

    mdot_g = np.zeros(N)
    mdot_g[-1] = abs(rdot[-1]) * dx
    for i in range(N-2, -1, -1):
        mdot_g[i] = mdot_g[i+1] + abs(rdot[i]) * dx
    mdot_total = mdot_g[0]

    k_s, cp_s = nodal_props(rho_cur[0])
    mdot_c    = (hg_unblown / cp_pyrolysis_gas) * B_prime_c
    rec_rate  = mdot_c / rho_char
    recession_depth += rec_rate * dt
    mdot_total += mdot_c

    Bc = (mdot_total * cp_pyrolysis_gas) / hg_unblown
    hg_bl = hg_unblown * (np.log(1+Bc)/Bc) if Bc > 0.001 else hg_unblown
    q_rerad = epsilon * sigma * T_cur[0]**4

    pyro_vol_s = delta_h_pyro * abs(rdot[0])
    cond_s     = k_s * (T_cur[0] - T_cur[1]) / dx
    dT0 = (2.0/(rho_cur[0]*cp_s*dx)) * (
        q_rad + hg_bl*(T_gas - T_cur[0]) - q_rerad - cond_s
        - pyro_vol_s * dx / 2.0) * dt
    T_nxt[0] = max(300., T_cur[0] + dT0)
    rho_nxt[0] = max(rho_char, rho_cur[0] + rdot[0]*dt)

    for i in range(1, N-1):
        k_n, cp_n = nodal_props(rho_cur[i])
        k_im1, _  = nodal_props(rho_cur[i-1])
        k_ip1, _  = nodal_props(rho_cur[i+1])
        kl = 0.5*(k_n + k_im1);  kr = 0.5*(k_n + k_ip1)
        cond = (kr*(T_cur[i+1]-T_cur[i]) - kl*(T_cur[i]-T_cur[i-1])) / dx**2
        pyro = (delta_h_pyro * abs(rdot[i])) / (rho_cur[i] * cp_n)
        adv  = (mdot_g[i]*cp_pyrolysis_gas*(T_cur[i+1]-T_cur[i])/dx) / (rho_cur[i]*cp_n)
        eta  = i / (N-1)
        dTdx = (T_cur[i+1]-T_cur[i-1])/(2*dx)
        gv   = rec_rate * (1-eta) * dTdx
        T_nxt[i] = max(300., T_cur[i] + (cond/(rho_cur[i]*cp_n) - pyro + adv + gv)*dt)
        rho_nxt[i] = max(rho_char, rho_cur[i] + rdot[i]*dt)

    k_o, cp_o = nodal_props(rho_cur[-1])
    pyro_vol_o = delta_h_pyro * abs(rdot[-1])
    dT_o = (2.0/(rho_cur[-1]*cp_o*dx)) * (
        k_o*(T_cur[-2]-T_cur[-1])/dx - pyro_vol_o*dx/2.0)*dt
    T_nxt[-1] = max(300., T_cur[-1] + dT_o)
    rho_nxt[-1] = max(rho_char, rho_cur[-1] + rdot[-1]*dt)

    T_cur[:]   = T_nxt
    rho_cur[:] = rho_nxt

    if t % STRIDE == 0 and frame_idx < n_frames:
        T_hist[frame_idx]       = T_cur.copy()
        rho_hist[frame_idx]     = rho_cur.copy()
        mdot_g_hist[frame_idx]  = mdot_g.copy()
        L_hist[frame_idx]       = L_initial - recession_depth
        recession_hist[frame_idx] = recession_depth
        mdot_blowing_hist[frame_idx] = mdot_total
        Tw_hist[frame_idx]      = T_cur[0]
        frame_idx += 1

print(f"Pre-simulation complete. {frame_idx} animation frames buffered.")

# ═══════════════════════════════════════════════════════════
# 2.  COLOUR MAP  (300 K → ~2500 K: black-body inspired)
# ═══════════════════════════════════════════════════════════
cmap_data = {
    "red":   [(0.00, 0.02, 0.02), (0.30, 0.12, 0.12), (0.55, 0.80, 0.80),
              (0.78, 1.00, 1.00), (1.00, 1.00, 1.00)],
    "green": [(0.00, 0.02, 0.02), (0.30, 0.02, 0.02), (0.55, 0.20, 0.20),
              (0.78, 0.55, 0.55), (1.00, 0.97, 0.97)],
    "blue":  [(0.00, 0.12, 0.12), (0.30, 0.35, 0.35), (0.55, 0.04, 0.04),
              (0.78, 0.02, 0.02), (1.00, 0.90, 0.90)],
}
heat_cmap = mcolors.LinearSegmentedColormap("incandescent", cmap_data, N=512)

T_MIN, T_MAX = 300.0, 2200.0
norm_T = mcolors.Normalize(vmin=T_MIN, vmax=T_MAX)

# ═══════════════════════════════════════════════════════════
# 3.  FIGURE LAYOUT
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9), facecolor="#0a0a0f")
fig.suptitle(
    "ENGINE GENESIS  ·  1-D Transient Ablation  ·  Real-Time Playback",
    color="#e8d8b0", fontsize=13, fontweight="bold", y=0.97,
    fontfamily="monospace",
)

# Grid: 2 rows × 3 cols
gs = fig.add_gridspec(
    2, 3,
    left=0.06, right=0.98,
    top=0.91,  bottom=0.07,
    wspace=0.35, hspace=0.40,
)

ax_main   = fig.add_subplot(gs[:, 0])   # tall: cross-section heat map
ax_rho    = fig.add_subplot(gs[0, 1])   # density / char fraction
ax_flux   = fig.add_subplot(gs[1, 1])   # outgassing flux profile
ax_tw     = fig.add_subplot(gs[0, 2])   # inner wall temperature history
ax_rec    = fig.add_subplot(gs[1, 2])   # recession depth history

for ax in [ax_main, ax_rho, ax_flux, ax_tw, ax_rec]:
    ax.set_facecolor("#0d0d18")
    for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")
    ax.tick_params(colors="#9090c0", labelsize=8)
    ax.xaxis.label.set_color("#9090c0")
    ax.yaxis.label.set_color("#9090c0")
    ax.title.set_color("#c8c0e0")

# ── MAIN: cross-section heat map ──────────────────────────────────────────────
N_STRIP = 80     # visual strips in the height direction
WALL_HEIGHT_VIS = 1.0   # normalised height

ax_main.set_xlim(-0.5, 14)          # x in mm; left is hot wall
ax_main.set_ylim(0, WALL_HEIGHT_VIS)
ax_main.set_xlabel("Depth into wall from hot face (mm)", fontsize=9)
ax_main.set_ylabel("Nozzle axial slice (normalised)", fontsize=9)
ax_main.set_title("Live Wall Cross-Section\n(Heat Map · Char · Flux)", fontsize=9)
ax_main.set_yticks([])

# Temperature image: each column is a node, rows fill the wall height
# We'll render as a stack of horizontal filled bars via LineCollection
# Pre-build a set of horizontal lines that we will update each frame
y_centers = np.linspace(0.05, 0.95, N_STRIP)
strip_height = (0.95 - 0.05) / N_STRIP

# For efficiency use an imshow with extent updated each frame
T_img_data = np.ones((N_STRIP, N)) * 300.0
im_heat = ax_main.imshow(
    T_img_data,
    aspect="auto", origin="lower",
    cmap=heat_cmap, norm=norm_T,
    extent=[0, L_initial*1000, 0, WALL_HEIGHT_VIS],
    interpolation="bilinear",
    zorder=2,
)

# Colorbar
cbar = fig.colorbar(im_heat, ax=ax_main, orientation="vertical",
                    fraction=0.04, pad=0.02)
cbar.set_label("Temperature (K)", color="#9090c0", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="#9090c0", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9090c0")

# Eroding wall boundary line
wall_line, = ax_main.plot([], [], color="#ff4040", lw=2.5, zorder=5, label="Hot face")
orig_wall_line = ax_main.axvline(x=L_initial*1000, color="#4488ff",
                                  lw=1.2, ls="--", zorder=4, label="Original OD")
ax_main.legend(fontsize=7, facecolor="#151528", edgecolor="#3a3a5c",
               labelcolor="#c0c0e0", loc="upper right")

# Gas flux arrows (quiver) — outward direction (leftward in frame)
arrow_y  = np.linspace(0.15, 0.85, 8)
arrow_x  = np.zeros(8)
arrow_u  = np.zeros(8) - 0.8
arrow_v  = np.zeros(8)
quiv = ax_main.quiver(
    arrow_x, arrow_y, arrow_u, arrow_v,
    color="#00ffcc", alpha=0.0,
    scale=15, width=0.006, zorder=6,
    headwidth=4, headlength=5,
)

# Time stamp
time_text = ax_main.text(
    0.97, 0.04, "t = 0.00 s",
    transform=ax_main.transAxes,
    color="#ffd080", fontsize=11, ha="right", va="bottom",
    fontfamily="monospace", fontweight="bold",
)

# ── Density / char fraction ────────────────────────────────────────────────────
ax_rho.set_xlabel("Depth (mm)", fontsize=8)
ax_rho.set_ylabel("Density (kg/m³)", fontsize=8, color="#5599ff")
ax_rho.set_xlim(0, L_initial*1000)
ax_rho.set_ylim(700, 1500)
ax_rho.tick_params(axis="y", labelcolor="#5599ff")
ax_rho.set_title("Density & Char Front", fontsize=8)
ax_rho.grid(True, color="#1e1e3a", lw=0.5)

ax_rho2 = ax_rho.twinx()
ax_rho2.set_ylim(0, 1.05)
ax_rho2.set_ylabel("Char Fraction", fontsize=8, color="#ff9933")
ax_rho2.tick_params(axis="y", labelcolor="#ff9933", labelsize=7)

rho_line,  = ax_rho.plot([], [], color="#5599ff", lw=1.5)
char_line, = ax_rho2.plot([], [], color="#ff9933", lw=1.5, ls="--")
ax_rho.set_facecolor("#0d0d18")

# ── Outgassing flux ────────────────────────────────────────────────────────────
ax_flux.set_xlabel("Depth (mm)", fontsize=8)
ax_flux.set_ylabel("Gas Flux (kg/m²-s)", fontsize=8, color="#00ffcc")
ax_flux.set_xlim(0, L_initial*1000)
ax_flux.set_ylim(0, 0.15)
ax_flux.tick_params(axis="y", labelcolor="#00ffcc")
ax_flux.set_title("Pyrolysis Gas Outflux Profile", fontsize=8)
ax_flux.grid(True, color="#1e1e3a", lw=0.5)

flux_fill = ax_flux.fill_between([], [], color="#00ffcc", alpha=0.25)
flux_line, = ax_flux.plot([], [], color="#00ffcc", lw=1.5)

# ── Inner wall temperature history ─────────────────────────────────────────────
time_axis_full = np.linspace(0, burn_time, n_frames)
ax_tw.set_xlabel("Time (s)", fontsize=8)
ax_tw.set_ylabel("T_wall (K)", fontsize=8, color="#ff6666")
ax_tw.set_xlim(0, burn_time)
ax_tw.set_ylim(200, 2400)
ax_tw.set_title("Inner Wall Temperature", fontsize=8)
ax_tw.tick_params(axis="y", labelcolor="#ff6666")
ax_tw.grid(True, color="#1e1e3a", lw=0.5)

tw_line, = ax_tw.plot([], [], color="#ff6666", lw=1.5)
tw_dot,  = ax_tw.plot([], [], "o", color="#ff6666", ms=5)

# ── Recession depth history ────────────────────────────────────────────────────
ax_rec.set_xlabel("Time (s)", fontsize=8)
ax_rec.set_ylabel("Recession (mm)", fontsize=8, color="#c060ff")
ax_rec.set_xlim(0, burn_time)
ax_rec.set_ylim(0, max(recession_hist)*1000 * 1.15 + 0.01)
ax_rec.set_title("Surface Recession Depth", fontsize=8)
ax_rec.tick_params(axis="y", labelcolor="#c060ff")
ax_rec.grid(True, color="#1e1e3a", lw=0.5)

rec_line, = ax_rec.plot([], [], color="#c060ff", lw=1.5)
rec_dot,  = ax_rec.plot([], [], "o", color="#c060ff", ms=5)

# ── Telemetry box ──────────────────────────────────────────────────────────────
tel_text = ax_main.text(
    0.03, 0.97,
    "",
    transform=ax_main.transAxes,
    color="#a0ffa0", fontsize=7.5, va="top", fontfamily="monospace",
    bbox=dict(facecolor="#0a0a14", edgecolor="#224422", alpha=0.85, pad=3),
    zorder=10,
)

# ═══════════════════════════════════════════════════════════
# 4.  ANIMATION FUNCTION
# ═══════════════════════════════════════════════════════════
def update(frame):
    fi = min(frame, n_frames - 1)
    T_f   = T_hist[fi]
    rho_f = rho_hist[fi]
    mg_f  = mdot_g_hist[fi]
    L_f   = L_hist[fi]
    rec_f = recession_hist[fi]
    t_now = fi * STRIDE * dt
    Tw    = Tw_hist[fi]
    mdot_bw = mdot_blowing_hist[fi]

    # x-axis in mm for this frame (hot face is always x=0)
    x_mm = np.linspace(0, L_f * 1000, N)

    # ── Heat map update ──────────────────────────────────────────────────
    # Broadcast T_f into the N_STRIP-row image
    T_2d = np.tile(T_f, (N_STRIP, 1))
    im_heat.set_data(T_2d)
    im_heat.set_extent([0, L_f*1000, 0, WALL_HEIGHT_VIS])

    # Eroding hot face line
    wall_line.set_data([0, 0], [0, WALL_HEIGHT_VIS])

    # ── Quiver: scale arrows by surface gas flux ──────────────────────────
    flux_scale = np.clip(mdot_bw / 0.10, 0, 1)
    quiv.set_alpha(min(flux_scale * 0.9, 0.85))
    new_x = np.zeros(8) + 0.05  # just inside hot face
    quiv.set_offsets(np.column_stack([new_x, arrow_y]))
    quiv.set_UVC(np.full(8, -flux_scale * 1.0), np.zeros(8))

    # Time stamp
    time_text.set_text(f"t = {t_now:.2f} s")

    # ── Density / char ───────────────────────────────────────────────────
    gamma_f = np.clip((rho_virgin - rho_f) / (rho_virgin - rho_char), 0, 1)
    rho_line.set_data(x_mm, rho_f)
    char_line.set_data(x_mm, gamma_f)
    ax_rho.set_xlim(0, max(x_mm[-1], 0.01))

    # ── Flux profile ─────────────────────────────────────────────────────
    flux_line.set_data(x_mm, mg_f)
    for coll in ax_flux.collections[:]:
        coll.remove()
    ax_flux.fill_between(x_mm, mg_f, color="#00ffcc", alpha=0.20)
    ax_flux.set_xlim(0, max(x_mm[-1], 0.01))
    ax_flux.set_ylim(0, max(mg_f.max() * 1.15, 0.005))

    # ── T_wall history ───────────────────────────────────────────────────
    tw_line.set_data(time_axis_full[:fi+1], Tw_hist[:fi+1])
    tw_dot.set_data([time_axis_full[fi]], [Tw])

    # ── Recession history ─────────────────────────────────────────────────
    rec_vals = recession_hist[:fi+1] * 1000
    rec_line.set_data(time_axis_full[:fi+1], rec_vals)
    rec_dot.set_data([time_axis_full[fi]], [rec_f * 1000])

    # ── Telemetry ─────────────────────────────────────────────────────────
    Bc_now = (mdot_bw * cp_pyrolysis_gas) / hg_unblown
    tel_text.set_text(
        f"  T_wall  : {Tw:6.0f} K\n"
        f"  Recess. : {rec_f*1000:.3f} mm\n"
        f"  ṁ_blow  : {mdot_bw:.4f} kg/m²s\n"
        f"  B_c     : {Bc_now:.4f}\n"
        f"  Wall Δ  : {L_f*1000:.3f} mm left"
    )

    return (im_heat, wall_line, quiv, time_text, rho_line, char_line,
            flux_line, tw_line, tw_dot, rec_line, rec_dot, tel_text)

# ═══════════════════════════════════════════════════════════
# 5.  RUN
# ═══════════════════════════════════════════════════════════
anim = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=12.67,        # ms between frames → ~60 fps playback
    blit=False,
    repeat=False,
)

print("Rendering animation …")
try:
    plt.show()
except Exception:
    # Headless fallback: save to MP4
    print("No display found – saving ablation_animation.mp4 …")
    Writer = animation.FFMpegWriter(fps=25, bitrate=2000)
    anim.save("/mnt/user-data/outputs/ablation_animation.mp4", writer=Writer, dpi=120)
    print("Saved: /mnt/user-data/outputs/ablation_animation.mp4")