"""
ablation_animation.py  -  Engine Genesis: Wall Ablation Visualiser
======================================================================
Usage:
  python ablation_animation.py             -> interactive window
  python ablation_animation.py --save      -> 60fps MP4 to OUTPUT_PATH
"""

# THIS FILE WAS WRITTEN BY CLAUDE CODE, LPTOC's PHYSICS WAS PASSED INTO THE MODEL AND TOLD TO CREATE A VISUALISATION OF THE SIMULATION RESULTS. 
# THE SIMULATION CODE ITSELF IS IN ABLATION.PY

import sys, numpy as np, warnings
warnings.filterwarnings("ignore")

import matplotlib
try:
    matplotlib.use("TkAgg"); import tkinter
except Exception:
    matplotlib.use("Agg")
    if "--save" not in sys.argv: sys.argv.append("--save")

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker

# ── User-configurable paths (edit these for your machine) ────────────────────
OUTPUT_PATH = "C:\\Users\\lptoc\\Downloads\\engine_genesis_ablation_v4.mp4"
matplotlib.rcParams['animation.ffmpeg_path'] = (
    r'C:\Users\lptoc\Downloads\ffmpeg-master-latest-win64-gpl'
    r'\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
)
SAVE_VIDEO = "--save" in sys.argv

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
print("[ Engine Genesis ] Running simulation...", flush=True)

rho_virgin=1.38e3; rho_char=800.0
delta_h_pyro=2.3e6; cp_pyrolysis_gas=3200.0; B_prime_c=0.08
B=3.0e9; E=1.4e5; m_ord=2; R=8.314
L_initial=0.0127; N=300; dt=0.0005; burn_time=8.0
time_steps=int(burn_time/dt)
hg_unblown=1346.9; q_rad=449.8813e3; T_gas=2494.0
epsilon=0.85; sigma=5.67e-8

# T-dependent properties (Phase 1 upgrade)
k_v_T  =np.array([300,  400,  500,  600,  700,  800 ])
k_v_val=np.array([.290, .310, .330, .350, .370, .390])
k_c_T  =np.array([700,  900, 1100, 1400, 1700, 2000, 2500])
k_c_val=np.array([.680, .820, .980,1.200,1.500,1.850,2.600])
cp_v_T  =np.array([300, 373, 473, 573, 673, 800])
cp_v_val=np.array([1300,1420,1560,1720,1870,1950])
cp_c_T  =np.array([700, 900,1100,1400,1700,2000,2500])
cp_c_val=np.array([900, 980,1060,1190,1340,1490,1700])

def drho_dt(rho,T,dt_):
    if rho<=rho_char: return 0.
    rt=((rho-rho_char)/rho_virgin)**m_ord
    return max(-B*np.exp(-E/(R*T))*rho_virgin*rt,-(rho-rho_char)/dt_)

def nodal_props(rho_n,T_n):
    g=np.clip((rho_virgin-rho_n)/(rho_virgin-rho_char),0.,1.)
    kv=np.interp(T_n,k_v_T,k_v_val); kc=np.interp(T_n,k_c_T,k_c_val)
    cv=np.interp(T_n,cp_v_T,cp_v_val); cc=np.interp(T_n,cp_c_T,cp_c_val)
    return kv*(1-g)+kc*g, cv*(1-g)+cc*g

STRIDE=50; n_frames=time_steps//STRIDE
T_hist=np.zeros((n_frames,N)); rho_hist=np.zeros((n_frames,N))
mdot_g_hist=np.zeros((n_frames,N)); L_hist=np.zeros(n_frames)
rec_hist=np.zeros(n_frames); mdot_hist=np.zeros(n_frames)
Tw_hist=np.zeros(n_frames); hgb_hist=np.zeros(n_frames); Bc_hist=np.zeros(n_frames)

T_cur=np.ones(N)*300.; rho_cur=np.ones(N)*rho_virgin
T_nxt=np.zeros(N); rho_nxt=np.zeros(N); rdot=np.zeros(N)
rec_depth=0.; fi=0

for t in range(time_steps):
    L_cur=L_initial-rec_depth; dx=L_cur/(N-1)
    for i in range(N): rdot[i]=drho_dt(rho_cur[i],T_cur[i],dt)
    mg=np.zeros(N); mg[-1]=abs(rdot[-1])*dx
    for i in range(N-2,-1,-1): mg[i]=mg[i+1]+abs(rdot[i])*dx
    mt=mg[0]
    ks,cps=nodal_props(rho_cur[0],T_cur[0])
    mc=(hg_unblown/cp_pyrolysis_gas)*B_prime_c
    rr=mc/rho_char; rec_depth+=rr*dt; mt+=mc
    Bc=(mt*cp_pyrolysis_gas)/hg_unblown
    hgb=hg_unblown*(np.log(1+Bc)/Bc) if Bc>0.001 else hg_unblown
    qr=epsilon*sigma*T_cur[0]**4
    pvs=delta_h_pyro*abs(rdot[0]); cs=ks*(T_cur[0]-T_cur[1])/dx
    dT0=(2./(rho_cur[0]*cps*dx))*(q_rad+hgb*(T_gas-T_cur[0])-qr-cs-pvs*dx/2.)*dt
    T_nxt[0]=max(300.,T_cur[0]+dT0); rho_nxt[0]=max(rho_char,rho_cur[0]+rdot[0]*dt)
    for i in range(1,N-1):
        kn,cpn=nodal_props(rho_cur[i],T_cur[i])
        kim,_=nodal_props(rho_cur[i-1],T_cur[i-1]); kip,_=nodal_props(rho_cur[i+1],T_cur[i+1])
        kl=.5*(kn+kim); kr=.5*(kn+kip)
        cond=(kr*(T_cur[i+1]-T_cur[i])-kl*(T_cur[i]-T_cur[i-1]))/dx**2
        pyro=(delta_h_pyro*abs(rdot[i]))/(rho_cur[i]*cpn)
        adv=(mg[i]*cp_pyrolysis_gas*(T_cur[i+1]-T_cur[i])/dx)/(rho_cur[i]*cpn)
        eta=i/(N-1); dTdx=(T_cur[i+1]-T_cur[i-1])/(2*dx); gv=rr*(1-eta)*dTdx
        T_nxt[i]=max(300.,T_cur[i]+(cond/(rho_cur[i]*cpn)-pyro+adv+gv)*dt)
        rho_nxt[i]=max(rho_char,rho_cur[i]+rdot[i]*dt)
    ko,cpo=nodal_props(rho_cur[-1],T_cur[-1]); pvo=delta_h_pyro*abs(rdot[-1])
    dTo=(2./(rho_cur[-1]*cpo*dx))*(ko*(T_cur[-2]-T_cur[-1])/dx-pvo*dx/2.)*dt
    T_nxt[-1]=max(300.,T_cur[-1]+dTo); rho_nxt[-1]=max(rho_char,rho_cur[-1]+rdot[-1]*dt)
    T_cur[:]=T_nxt; rho_cur[:]=rho_nxt
    if t%STRIDE==0 and fi<n_frames:
        T_hist[fi]=T_cur.copy(); rho_hist[fi]=rho_cur.copy()
        mdot_g_hist[fi]=mg.copy(); L_hist[fi]=L_initial-rec_depth
        rec_hist[fi]=rec_depth; mdot_hist[fi]=mt
        Tw_hist[fi]=T_cur[0]; hgb_hist[fi]=hgb; Bc_hist[fi]=Bc; fi+=1

print(f"[ Engine Genesis ] Done. {fi} frames.", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  COLOUR MAPS
# ═══════════════════════════════════════════════════════════════════════════════
bb_ramp={"red":  [(0.,.02,.02),(.20,.08,.08),(.45,.75,.75),(.72,1.,1.),(1.,1.,1.)],
         "green":[(0.,.02,.02),(.20,.02,.02),(.45,.15,.15),(.72,.50,.50),(1.,.96,.96)],
         "blue": [(0.,.18,.18),(.20,.42,.42),(.45,.04,.04),(.72,.02,.02),(1.,.88,.88)]}
cmap_bb  = mcolors.LinearSegmentedColormap("bb",  bb_ramp,  N=512)
char_ramp={"red":  [(0.,.97,.97),(.35,.88,.88),(.65,.50,.50),(1.,.12,.12)],
           "green":[(0.,.95,.95),(.35,.62,.62),(.65,.28,.28),(1.,.06,.06)],
           "blue": [(0.,.90,.90),(.35,.18,.18),(.65,.08,.08),(1.,.04,.04)]}
cmap_char= mcolors.LinearSegmentedColormap("char",char_ramp,N=256)
norm_T   = mcolors.Normalize(vmin=300.,vmax=2400.)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FIGURE  — dark cinematic canvas
# ═══════════════════════════════════════════════════════════════════════════════
BG="#06060e"; PANEL="#0b0b18"; EDGE="#252540"
fig=plt.figure(figsize=(20,11),facecolor=BG)
fig.patch.set_facecolor(BG)

# Grid: main wall panel left, three right panels stacked
ax_wall = fig.add_axes([0.01, 0.08, 0.62, 0.88])   # big wall cross-section
ax_eng  = fig.add_axes([0.66, 0.70, 0.32, 0.26])   # mini engine schematic
ax_tw   = fig.add_axes([0.66, 0.40, 0.32, 0.26])   # T_wall history
ax_rec  = fig.add_axes([0.66, 0.08, 0.32, 0.26])   # recession + blowing

for ax in [ax_wall,ax_eng,ax_tw,ax_rec]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
    ax.tick_params(colors="#7070a0",labelsize=8)
    ax.xaxis.label.set_color("#7070a0"); ax.yaxis.label.set_color("#7070a0")
    ax.title.set_color("#b0a8d0")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN WALL PANEL
#
#  We draw the wall as a TALL vertical rectangle in (depth_mm, height) space.
#  depth_mm: 0 = hot face (left), L_initial*1000 = cold back face (right)
#  height:   0 to 1  (normalised, no physical meaning — just for visual weight)
#
#  The recession is EXAGGERATED 10x so you can actually see it move.
#  A scale bar and callout note make the exaggeration explicit.
# ═══════════════════════════════════════════════════════════════════════════════
EXAG = 10.0          # visual exaggeration factor for recession
WALL_MM = L_initial * 1000.   # 12.7 mm
H = 1.0              # wall height in normalised units

ax_wall.set_xlim(-2.5, WALL_MM + 1.5)
ax_wall.set_ylim(-0.08, H + 0.12)
ax_wall.set_xlabel("Depth into wall from hot face  (mm)", fontsize=10, labelpad=6)
ax_wall.set_facecolor(BG)
ax_wall.set_yticks([])
for sp in ax_wall.spines.values(): sp.set_visible(False)
ax_wall.tick_params(axis='x',colors="#7070a0",labelsize=9)

# Title
ax_wall.text(WALL_MM/2, H+0.09,
             "ENGINE GENESIS  ·  COMBUSTION CHAMBER LINER  ·  Canvas Phenolic Grade CE",
             ha="center", va="bottom", color="#d0c8f0", fontsize=13,
             fontweight="bold", fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3,foreground=BG)])

# ── Static wall background (full thickness, cold grey) ───────────────────────
ax_wall.fill_betweenx([0, H], 0, WALL_MM, color="#14142a", zorder=1)
ax_wall.fill_betweenx([0, H], WALL_MM, WALL_MM+0.8, color="#1a1a30", zorder=1)  # structure behind

# Back-face structural wall label
ax_wall.text(WALL_MM+0.9, H/2, "structural\ncasing",
             color="#2a3a50", fontsize=7.5, fontfamily="monospace",
             va="center", ha="left", rotation=90)

# ── N_COL vertical columns coloured by T and char fraction ───────────────────
N_COL = 200   # number of vertical colour columns across wall thickness
col_edges = np.linspace(0, WALL_MM, N_COL+1)
col_patches_T    = []   # temperature heat-map rectangles
col_patches_char = []   # char-fraction overlay rectangles

from matplotlib.patches import Rectangle
for c in range(N_COL):
    x0, x1 = col_edges[c], col_edges[c+1]
    # Heat-map layer (full height)
    rT = Rectangle((x0,0), x1-x0, H, color=cmap_bb(0.), zorder=2, lw=0)
    ax_wall.add_patch(rT)
    col_patches_T.append(rT)
    # Char overlay (inner 40% of each column — nearer the hot face the char sits)
    rC = Rectangle((x0,0), x1-x0, H, color="#111", zorder=3, lw=0, alpha=0.)
    ax_wall.add_patch(rC)
    col_patches_char.append(rC)

# ── Hot face glow line (dynamic, left edge of wall) ──────────────────────────
# Physical recession is tiny; we show it exaggerated by EXAG factor
hotface_line = ax_wall.axvline(0., color="#ff3300", lw=3, zorder=8,
    path_effects=[pe.withStroke(linewidth=6,foreground="#ff110022")])

# Original hot-face reference
ax_wall.axvline(0., color="#336688", lw=0.8, ls=":", zorder=4, alpha=0.5)

# Recession marker (exaggerated, moves rightward as material is consumed)
rec_marker_line = ax_wall.axvline(0., color="#ff8800", lw=2., ls="--", zorder=9, alpha=0.9)

# Hot-gas region left of hot face
hg_fill = ax_wall.fill_betweenx([0,H], -2.5, 0., color="#1a0800", zorder=1)
hg_edge, = ax_wall.plot([0.,0.], [0.,H], color="#ff3300", lw=3, zorder=8,
    path_effects=[pe.withStroke(linewidth=6,foreground="#ff110022")])

# Gas-core label
ax_wall.text(-1.25, H/2, "hot gas\ncore", color="#553322", fontsize=8,
             fontfamily="monospace", ha="center", va="center")

# ── Pyrolysis gas arrows — emerge from hot face leftward into gas core ────────
n_arr = 9
arr_y = np.linspace(0.08, 0.92, n_arr)
gas_arrows = []
for ay in arr_y:
    a = ax_wall.annotate("",
        xy=(-1.5, ay), xytext=(1.2, ay),
        arrowprops=dict(arrowstyle="->", color="#00ffaa", lw=1.8, mutation_scale=12),
        zorder=10, annotation_clip=False)
    a.set_alpha(0.)
    gas_arrows.append(a)

# ── Depth annotation lines at key depths ─────────────────────────────────────
# Char front depth indicator (dynamic dashed vertical)
char_front_line = ax_wall.axvline(0., color="#ff9933", lw=1.2, ls="-.", zorder=7, alpha=0.)
char_front_label = ax_wall.text(0., H+0.03, "", color="#ff9933",
    fontsize=8, fontfamily="monospace", ha="center", va="bottom")

# Fixed depth callout at bondline (right edge)
ax_wall.axvline(WALL_MM, color="#336688", lw=1.0, ls=":", zorder=4, alpha=0.6)
ax_wall.text(WALL_MM, -0.04, f"{WALL_MM:.1f} mm\n(bondline)", color="#336688",
             fontsize=7.5, fontfamily="monospace", ha="center", va="top")

# ── Scale bar showing actual vs exaggerated recession ────────────────────────
# Drawn once, updated each frame
sb_y = -0.055
scalebar_actual = ax_wall.annotate("",
    xy=(0.,sb_y), xytext=(0.,sb_y),
    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2),
    annotation_clip=False)
scalebar_label = ax_wall.text(0., sb_y-0.012, "", color="#888888",
    fontsize=7.5, fontfamily="monospace", ha="center", va="top")

exag_note = ax_wall.text(WALL_MM/2, -0.055,
    f"Recession shown at {EXAG:.0f}× exaggeration for visibility",
    color="#556688", fontsize=8, fontfamily="monospace", ha="center", va="center")

# ── Colorbar ─────────────────────────────────────────────────────────────────
sm=plt.cm.ScalarMappable(cmap=cmap_bb,norm=norm_T); sm.set_array([])
cbar_ax=fig.add_axes([0.02,0.03,0.38,0.020])
cbar=fig.colorbar(sm,cax=cbar_ax,orientation="horizontal")
cbar.set_label("Wall temperature  (K)", color="#7070a0", fontsize=8.5)
cbar.ax.xaxis.set_tick_params(color="#7070a0",labelsize=7.5)
plt.setp(cbar.ax.xaxis.get_ticklabels(),color="#7070a0")
cbar_ax.set_facecolor(BG)

# ── Legend ────────────────────────────────────────────────────────────────────
leg_patches=[
    mpatches.Patch(color="#ff3300",label="Hot face (ablating)"),
    mpatches.Patch(color="#ff8800",label=f"Receded surface ({EXAG:.0f}× exag.)"),
    mpatches.Patch(color="#ff9933",label="Char front"),
    mpatches.Patch(color="#00ffaa",label="Pyrolysis gas flux"),
]
ax_wall.legend(handles=leg_patches, loc="upper right", fontsize=8,
    facecolor="#0d0d22", edgecolor=EDGE, labelcolor="#c0b8e0", framealpha=0.92)

# ── Timestamp & telemetry ─────────────────────────────────────────────────────
time_lbl=ax_wall.text(0.01,0.97,"t = 0.00 s",transform=ax_wall.transAxes,
    color="#ffd080",fontsize=14,fontweight="bold",fontfamily="monospace",
    va="top",ha="left",path_effects=[pe.withStroke(linewidth=3,foreground=BG)])

tel=ax_wall.text(0.01,0.88,"",transform=ax_wall.transAxes,
    color="#88ffbb",fontsize=9,va="top",ha="left",fontfamily="monospace",
    bbox=dict(facecolor="#080818",edgecolor="#1a3a2a",alpha=0.92,pad=5,
              boxstyle="round,pad=0.4"),zorder=12)

# ── Thermal wave depth marker ─────────────────────────────────────────────────
wave_line = ax_wall.axvline(0., color="#8866ff", lw=1.2, ls=":", zorder=6, alpha=0.7)
wave_label= ax_wall.text(0., H*0.55, "", color="#8866ff",
    fontsize=7.5, fontfamily="monospace", ha="left", va="center",
    path_effects=[pe.withStroke(linewidth=2,foreground=BG)])

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MINI ENGINE SCHEMATIC (top-right, context only)
# ═══════════════════════════════════════════════════════════════════════════════
R_ch=5.; T_ch=1.6; Xc=14.; R_th=2.1; X_conv=5.; R_ex=3.8; X_div=12.
Xt=Xc+X_conv; Xe=Xt+X_div
N_PTS=300; x_e=np.linspace(0,Xe,N_PTS)

def r_inner_fn(x):
    r=np.empty_like(x,dtype=float)
    mc=x<=Xc; r[mc]=R_ch
    mcv=(x>Xc)&(x<=Xt); t=(x[mcv]-Xc)/X_conv
    r[mcv]=R_ch+(R_th-R_ch)*(1-np.cos(np.pi*t))/2
    mdv=x>Xt; t2=(x[mdv]-Xt)/X_div
    r[mdv]=R_th+(R_ex-R_th)*(1-np.cos(np.pi*t2))/2
    return r

ri_e=r_inner_fn(x_e)
ro_ch=R_ch+T_ch; ro_noz=ri_e+0.9
ro_noz[x_e<=Xc]=ro_ch

ax_eng.set_xlim(-1,Xe+1); ax_eng.set_ylim(-(R_ch+T_ch+1),(R_ch+T_ch+1))
ax_eng.set_aspect("equal"); ax_eng.axis("off")
ax_eng.set_title("Engine cross-section", fontsize=8, pad=3)

# Gas core
for idx,xi in enumerate(np.linspace(0,1,40)):
    x0=xi*Xe; x1=x0+Xe/40
    ri0=r_inner_fn(np.array([x0]))[0]; ri1=r_inner_fn(np.array([x1]))[0]
    alpha=0.10+0.08*np.sin(np.pi*xi)
    ax_eng.fill_between([x0,x1],[-ri0,-ri1],[ri0,ri1],color=(1.,.4,.05,alpha),zorder=2)

# Nozzle wall
x_noz_s=x_e[x_e>Xc]; ri_ns=r_inner_fn(x_noz_s); ro_ns=ri_ns+0.9
ax_eng.fill_between(x_noz_s, ri_ns, ro_ns,  color="#1c2530",zorder=3)
ax_eng.fill_between(x_noz_s,-ro_ns,-ri_ns,  color="#1c2530",zorder=3)
ax_eng.plot(x_noz_s, ro_ns,  color="#3a5060",lw=0.8,zorder=4)
ax_eng.plot(x_noz_s,-ro_ns,  color="#3a5060",lw=0.8,zorder=4)

# Chamber ablative liner (coloured rectangle, updated per frame)
ch_patch_u=ax_eng.fill_between([0,Xc],[R_ch,R_ch],[ro_ch,ro_ch],
    color=cmap_bb(0.),zorder=3)
ch_patch_l=ax_eng.fill_between([0,Xc],[-ro_ch,-ro_ch],[-R_ch,-R_ch],
    color=cmap_bb(0.),zorder=3)
ax_eng.plot([0,Xc],[ro_ch,ro_ch], color="#3a5060",lw=0.8,zorder=5)
ax_eng.plot([0,Xc],[-ro_ch,-ro_ch],color="#3a5060",lw=0.8,zorder=5)
ax_eng.plot([0,Xc],[R_ch,R_ch],   color="#6688aa",lw=0.8,zorder=5)
ax_eng.plot([0,Xc],[-R_ch,-R_ch], color="#6688aa",lw=0.8,zorder=5)
ax_eng.axhline(0,color="#1a1a38",lw=0.5,ls="--",zorder=1)

# Chamber label
ax_eng.text(Xc/2,ro_ch+0.3,"ablative liner",color="#4466aa",
    fontsize=5.5,fontfamily="monospace",ha="center")
ax_eng.text(Xt+X_div/2,R_ex+1.2,"nozzle (steel)",color="#445566",
    fontsize=5.5,fontfamily="monospace",ha="center")

# Zoom-arrow pointing from engine chamber to main wall panel
ax_eng.annotate("", xy=(0.0,0.5), xytext=(-0.5,0.5),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>",color="#6688aa",lw=1.5,
                    connectionstyle="arc3,rad=0.0"),annotation_clip=False)
ax_eng.text(-0.55,0.5,"  zoom →",transform=ax_eng.transAxes,
    color="#4466aa",fontsize=7,fontfamily="monospace",va="center",ha="right")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  RIGHT CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
t_ax=np.linspace(0,burn_time,n_frames)

ax_tw.set_xlim(0,burn_time); ax_tw.set_ylim(200,2600)
ax_tw.set_xlabel("Time (s)",fontsize=8); ax_tw.set_ylabel("T (K)",fontsize=8,color="#ff6644")
ax_tw.tick_params(axis="y",labelcolor="#ff6644")
ax_tw.set_title("Inner wall temperature",fontsize=8)
ax_tw.grid(True,color="#151528",lw=0.5)
ax_tw.axhline(1800,color="#ff6633",lw=0.7,ls="--",alpha=0.55)
ax_tw.text(7.6,1840,"char onset",color="#ff6633",fontsize=7,ha="right")
tw_line,=ax_tw.plot([],[],color="#ff6644",lw=1.8)
tw_dot, =ax_tw.plot([],[],">",color="#ffaa44",ms=6,zorder=5)

ax_rec.set_xlim(0,burn_time)
ax_rec.set_xlabel("Time (s)",fontsize=8)
ax_rec.set_ylabel("Recession (mm)",fontsize=8,color="#cc88ff")
ax_rec.tick_params(axis="y",labelcolor="#cc88ff")
ax_rec.set_ylim(0, max(rec_hist)*1000*1.3+0.01)
ax_rec.set_title("Recession depth  &  blowing flux",fontsize=8)
ax_rec.grid(True,color="#151528",lw=0.5)
ax_rec2=ax_rec.twinx()
ax_rec2.set_ylim(0,max(mdot_hist)*1.4+0.005)
ax_rec2.set_ylabel("ṁ_blow (kg/m²·s)",fontsize=8,color="#44ccff")
ax_rec2.tick_params(axis="y",labelcolor="#44ccff",labelsize=7)
ax_rec2.set_facecolor(PANEL)
for sp in ax_rec2.spines.values(): sp.set_edgecolor(EDGE)
rec_line, =ax_rec.plot([],[],color="#cc88ff",lw=1.8)
rec_dot,  =ax_rec.plot([],[],"o",color="#ee88ff",ms=6,zorder=5)
blow_line,=ax_rec2.plot([],[],color="#44ccff",lw=1.4,ls="--")

fig.text(0.50,0.003,
    "Engine Genesis  —  1-D Charring Ablation  |  Canvas Phenolic Grade CE  |  "
    f"Recession exaggerated {EXAG:.0f}× for visibility",
    ha="center",color="#2a2a50",fontsize=7,fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  UPDATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
# Pre-compute node positions in mm for the full initial wall
node_x_mm = np.linspace(0, WALL_MM, N)   # fixed spatial reference

def update(frame):
    fi_=min(frame,n_frames-1)
    T_f   =T_hist[fi_]; rho_f =rho_hist[fi_]; mg_f=mdot_g_hist[fi_]
    L_f   =L_hist[fi_]; rec_f =rec_hist[fi_];  Tw =Tw_hist[fi_]
    mb    =mdot_hist[fi_]; Bc_f=Bc_hist[fi_];  hgb=hgb_hist[fi_]
    t_now =fi_*STRIDE*dt

    # ── Wall column colours ──────────────────────────────────────────────────
    gamma=np.clip((rho_virgin-rho_f)/(rho_virgin-rho_char),0.,1.)

    for c in range(N_COL):
        xi_norm=(c+0.5)/N_COL          # 0=hot face, 1=cold back
        # Corresponding 1D node index
        node_idx=int(xi_norm*(N-1))
        T_c   = T_f[node_idx]
        char_c= gamma[node_idx]

        col_patches_T[c].set_facecolor(cmap_bb(norm_T(T_c)))
        # Char overlay alpha proportional to char fraction
        col_patches_char[c].set_facecolor(cmap_char(char_c))
        col_patches_char[c].set_alpha(np.clip(0.1+0.85*char_c, 0, 0.9))

    # ── Exaggerated recession — hot face line shifts right ───────────────────
    rec_vis_mm = rec_f * 1000. * EXAG   # mm, exaggerated
    # Clamp so it never exceeds 40% of wall (prevents visual confusion)
    rec_vis_mm = min(rec_vis_mm, WALL_MM * 0.40)

    # Update hot-face position
    hotface_line.set_xdata([rec_vis_mm, rec_vis_mm])
    hg_edge.set_xdata([rec_vis_mm, rec_vis_mm])
    rec_marker_line.set_xdata([rec_vis_mm, rec_vis_mm])

    # Hot-gas fill now extends to rec_vis_mm
    for coll in ax_wall.collections[:]:
        pass  # don't clear all — just update specific ones below

    # ── Char front indicator ──────────────────────────────────────────────────
    # Find deepest node where gamma > 0.05
    char_nodes = np.where(gamma > 0.05)[0]
    if len(char_nodes) > 0:
        char_depth_mm = node_x_mm[char_nodes[-1]]
        char_front_line.set_xdata([char_depth_mm, char_depth_mm])
        char_front_line.set_alpha(0.75)
        char_front_label.set_x(char_depth_mm)
        char_front_label.set_text(f"char\n{char_depth_mm:.1f} mm")
    else:
        char_front_line.set_alpha(0.)
        char_front_label.set_text("")

    # ── Thermal wave penetration depth (T > 350 K) ───────────────────────────
    hot_nodes = np.where(T_f > 350.)[0]
    if len(hot_nodes) > 0:
        wave_depth_mm = node_x_mm[hot_nodes[-1]]
        wave_line.set_xdata([wave_depth_mm, wave_depth_mm])
        wave_label.set_x(wave_depth_mm + 0.15)
        wave_label.set_y(H * 0.30)
        wave_label.set_text(f"thermal\nwave\n{wave_depth_mm:.1f} mm")

    # ── Pyrolysis arrows: alpha scales with blowing flux ─────────────────────
    flux_scale = np.clip(mb / 0.12, 0., 1.)
    for a in gas_arrows:
        a.set_alpha(np.clip(flux_scale * 0.92, 0., 0.95))
        # Move arrow tail to current exaggerated hot face
        a.xyann = (rec_vis_mm + 0.8, a.xyann[1])
        a.xy    = (-1.5, a.xy[1])

    # ── Scale bar annotation ──────────────────────────────────────────────────
    actual_mm = rec_f * 1000.
    scalebar_label.set_text(
        f"actual recession: {actual_mm:.3f} mm  "
        f"(shown as {rec_vis_mm:.2f} mm  ×{EXAG:.0f})"
    )
    scalebar_label.set_x(rec_vis_mm / 2.)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    hg_redn=(1.-hgb/hg_unblown)*100.
    remaining_mm=(L_f*1000.)
    tel.set_text(
        f"  T_wall    {Tw:6.0f} K\n"
        f"  Recession {actual_mm:.4f} mm\n"
        f"  ṁ_blow   {mb:.4f} kg/m²s\n"
        f"  Bc        {Bc_f:.4f}\n"
        f"  hg redn  {hg_redn:.1f} %\n"
        f"  Liner rem {remaining_mm:.2f} mm\n"
        f"  Margin    {remaining_mm/L_initial/10.:.0f}× (vs full thickness)"
    )
    time_lbl.set_text(f"t = {t_now:.2f} s")

    # ── Mini engine: colour chamber liner by surface temperature ─────────────
    col_eng = cmap_bb(norm_T(Tw))
    ch_patch_u.set_facecolor(col_eng)
    ch_patch_l.set_facecolor(col_eng)

    # ── Right panels ──────────────────────────────────────────────────────────
    tw_line.set_data(t_ax[:fi_+1],Tw_hist[:fi_+1])
    tw_dot.set_data([t_ax[fi_]],[Tw])
    rec_line.set_data(t_ax[:fi_+1],rec_hist[:fi_+1]*1000.)
    rec_dot.set_data([t_ax[fi_]],[rec_f*1000.])
    blow_line.set_data(t_ax[:fi_+1],mdot_hist[:fi_+1])

    return []

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  RENDER
# ═══════════════════════════════════════════════════════════════════════════════
FPS=60 if SAVE_VIDEO else 30
anim=animation.FuncAnimation(fig,update,frames=n_frames,
    interval=int(1000/FPS),blit=False,repeat=False)

if SAVE_VIDEO:
    print(f"[ Engine Genesis ] Saving {n_frames} frames @ {FPS}fps...",flush=True)
    writer=animation.FFMpegWriter(fps=FPS,codec="libx264",bitrate=12000,
        extra_args=["-crf","12","-preset","slow",
                    "-pix_fmt","yuv420p","-movflags","+faststart"])
    anim.save(OUTPUT_PATH,writer=writer,dpi=150,
        savefig_kwargs={"facecolor":fig.get_facecolor()})
    print(f"[ Engine Genesis ] Saved: {OUTPUT_PATH}",flush=True)
else:
    print("[ Engine Genesis ] Opening interactive window...",flush=True)
    plt.show()