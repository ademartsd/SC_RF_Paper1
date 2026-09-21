import numpy as np

# Number of models (best ones) to plot:
# I recommend a small number to make sure the code runs
number_to_plot = 10 # change it to 50, 100, 1000...
step = 1

# Activate radials[:1] and just the first model for KDE
FAST_VIEW_MODE = True # change it to False once you want the real plots (slow)

# ======== CHOOSE YOUR PRESET HERE ========
PRESET = "NW deep"  
# ========================================

presets = {
    "NW deep":  {"VS_MIN": 2.3,  "VS_MAX": 4.7, "DEPTH_MIN": 0.0, "DEPTH_MAX": 55.0}
    }

# Stream properties
sps = 40 #samples per second
end_time = 30 #second when the last multiple arrives
begin_time = -2 #when we start the trace
number_pts = int((sps * (end_time - begin_time)) + 1)


if PRESET not in presets:
    raise ValueError(f"Unknown PRESET '{PRESET}'. Choose one of: {', '.join(presets.keys())}")

cfg = presets[PRESET]
VS_MIN, VS_MAX = cfg["VS_MIN"], cfg["VS_MAX"]
DEPTH_MIN, DEPTH_MAX = cfg["DEPTH_MIN"], cfg["DEPTH_MAX"]


if FAST_VIEW_MODE:
    print("--------------------------------------------------------------")
    print("[INFO] FAST_VIEW_MODE=True")# → Using only 2 traces for debugging")
    print("--------------------------------------------------------------")
else:
    print("--------------------------------------------------------------")
    print("[INFO] FAST_VIEW_MODE=False → Using full RF dataset")
    print("--------------------------------------------------------------")


# --- Standard library ---
import glob
import re
import sys
import ast
import csv
import os

# --- Scientific stack ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# --- Seismology / RF packages ---
import obspy
from obspy import read, Trace, UTCDateTime
from rf import read_rf, RFStream, rfstats
import pyraysum as prs
from pyraysum import Geometry, Control

################################################################
# --- Detect station name from current folder ---
station = os.path.basename(os.getcwd()).strip()
#print(f"[station] Detected station folder: {station}")

# Vp/Vs from Crotwell & Owens (2005):
vpvs = 1.72

#############################################################

# --- USER DEFINED SETTINGS ---
csv_file = f"./output/MATRIX_res_{station}.csv"
threshold_vs = 3600 # It does nothing, but in case you want to modify it...

# --- Read full chain ---
df_all = pd.read_csv(csv_file)

# --- Total iterations and 10% burn-in ---
last_iter = int(df_all["Iteration"].iloc[-1])
burn_in = round(last_iter * 0.10)
total_iterations_run = last_iter

# --- Remove burn-in ---
df_postburn = df_all[df_all["Iteration"] > burn_in].reset_index(drop=True)
first_iter = int(df_postburn["Iteration"].iloc[0])
last_iter_postburn = int(df_postburn["Iteration"].iloc[-1])
iterations_after_burnin = last_iter_postburn - burn_in

# --- Acceptance rate from the entire post-burn-in chain ---
num_accepted = df_postburn[df_postburn["Acceptance"] == 1].shape[0]
acceptance_rate = 100 * num_accepted / iterations_after_burnin

# ==========================================================================================
# Full posterior statistics
# ==========================================================================================

moho_depths = []
crustal_vs_values = []
misfit_values = []

for i in range(len(df_postburn)):

    vs = np.array(ast.literal_eval(df_postburn.iloc[i]["VS"]), dtype=float)
    thickness = np.array(ast.literal_eval(df_postburn.iloc[i]["Thickness"]), dtype=float)

    # Require mantle Vs >= threshold
    if vs[-1] < threshold_vs:
        continue

    # Skip LVZ models
    if not np.all(np.diff(vs) >= 0):
        continue

    # Moho depth
    moho = np.sum(thickness)

    # Effective average crustal Vs
    crust_thickness = thickness[:-1]
    crust_vs = vs[:-1]
    total_crust_thickness = np.sum(crust_thickness)
    total_s_travel_time = np.sum(crust_thickness / crust_vs)
    vs_eff = total_crust_thickness / total_s_travel_time

    moho_depths.append(moho)
    crustal_vs_values.append(vs_eff)
    misfit_values.append(float(df_postburn.iloc[i]["Misfit"]))

moho_depths = np.array(moho_depths)
crustal_vs_values = np.array(crustal_vs_values)
misfit_values = np.array(misfit_values)

# --- Posterior statistics ---

number_models_used = len(moho_depths)

moho_mean_km = np.mean(moho_depths) / 1000
moho_std_km = np.std(moho_depths, ddof=1) / 1000

crustal_vs_mean_km = np.mean(crustal_vs_values) / 1000
crustal_vs_std_km = np.std(crustal_vs_values, ddof=1) / 1000

misfit_min = np.min(misfit_values)
misfit_max = np.max(misfit_values)

# ==========================================================================================
# Summary
# ==========================================================================================

summary_text = (
    "--------------------------------------------------\n"
    f"Station:                           {station}\n"
    f"Total iterations (full chain):    {total_iterations_run}\n"
    f"Iterations after burn-in:         {iterations_after_burnin}\n"
    f"Acceptance rate:                  {acceptance_rate:.2f}%\n"
    f"Number of models used:            {number_models_used}\n"
    f"Misfit range of models used:      {misfit_min:.2f} - {misfit_max:.2f}\n"
    f"Estimated Avg. Crustal Vs:         {crustal_vs_mean_km:.2f} ± {crustal_vs_std_km:.2f} km/s\n"
    f"Estimated Moho Depth:              {moho_mean_km:.2f} ± {moho_std_km:.2f} km\n"
    "--------------------------------------------------\n"
)

print(summary_text)

with open(f"./output/{station}_summary.txt", "w") as f:
    f.write(summary_text)

# sys.exit()


# ==========================================================================================
# Select best models for plotting
# ==========================================================================================

df = df_postburn.sort_values(by="Likelihood", ascending=False).reset_index(drop=True)

filtered_rows = []
idx = 0

while len(filtered_rows) < number_to_plot and idx < len(df):
    vs_vector = ast.literal_eval(df.iloc[idx]["VS"])

    if vs_vector[-1] >= threshold_vs and np.all(np.diff(vs_vector) >= 0):
        filtered_rows.append(df.iloc[idx])

    idx += 1

df = pd.DataFrame(filtered_rows).reset_index(drop=True)

print(f"Selected {len(df)} best models for plotting.")


# ==========================================================================================
# Plot models with preset ranges + Target (mT) and Initial (mI)
# ==========================================================================================

plt.figure(figsize=(5, 6))
lvz_violations = []

def normalize_misfit(misfit, min_misfit, max_misfit):
    if max_misfit == min_misfit:
        return 0.5
    return (misfit - min_misfit) / (max_misfit - min_misfit)

# Plot violet models
n_models = len(df)
top10 = int(np.ceil(0.10 * n_models))

for i in range(n_models):

    vs = ast.literal_eval(df.iloc[i]['VS'])
    thickness = ast.literal_eval(df.iloc[i]['Thickness'])

    # Skip models with LVZs
    if not np.all(np.diff(vs) >= 0):
        lvz_violations.append(df.iloc[i]['Iteration'])
        continue

    num_layers = min(len(vs), len(thickness))
    vs = vs[:num_layers]
    thickness = thickness[:num_layers]

    # Depth in km
    depth = np.cumsum(thickness) / 1000.0

    # ----------------------------------------------------------
    # Alpha based on model rank
    # Top 10%:       1.00 → 0.90
    # Remaining 90%: 0.20 → 0.02
    # ----------------------------------------------------------
    if i < top10:
        if top10 == 1:
            alpha = 1.0
        else:
            alpha = 1.0 - 0.10 * (i / (top10 - 1))
    else:
        remaining = n_models - top10
        if remaining == 1:
            alpha = 0.02
        else:
            frac = (i - top10) / (remaining - 1)
            alpha = 0.20 - 0.18 * frac

    linewidth = 2.0

    plt.step(vs, depth,
             color='violet',
             where='post',
             alpha=alpha,
             linewidth=linewidth)

    plt.vlines(x=vs[0], ymin=0, ymax=depth[0],
               color='violet',
               alpha=alpha,
               linewidth=linewidth)

    plt.vlines(x=vs[-1], ymin=depth[-1], ymax=cfg["DEPTH_MAX"],
               color='violet',
               alpha=alpha,
               linewidth=linewidth)


# Plot settings from preset
plt.ylim(cfg["DEPTH_MAX"], cfg["DEPTH_MIN"])
plt.xlim(cfg["VS_MIN"]*1000, cfg["VS_MAX"]*1000)  # convert km/s back to m/s
plt.xlabel('Vs (m s$^{-1}$)')
plt.ylabel('Depth (km)')
plt.title(f"{station}: best {number_to_plot} models")

# Exporting first figure
os.makedirs("./figs", exist_ok=True)
plt.savefig(f'./figs/models_{station}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'./figs/models_{station}.png', format='png', dpi=300, bbox_inches='tight')

# Report LVZ models
if lvz_violations:
    print(f"LVZ detected in iterations: {lvz_violations}")
    print(f"Total LVZ-violating models ignored: {len(lvz_violations)}")
else:
    print("No LVZ violations detected.")
print("  ")


# ==========================================================================================
# MODEL GRID EXPORTATION
# ==========================================================================================

TOTAL_DEPTH = int(cfg["DEPTH_MAX"] * 1000)  # preset depth in meters
num_models = len(df) # best models selected for plotting

with open(f"./output/MODEL_GRID_{station}.csv", "w", newline="") as f:
    writer = csv.writer(f)

    for idx in range(num_models):
        vs = ast.literal_eval(df.iloc[idx]['VS'])
        thick = ast.literal_eval(df.iloc[idx]['Thickness'])

        # Fix final layer thickness to reach TOTAL_DEPTH
        thick[-1] = TOTAL_DEPTH - sum(thick[:-1])

        depths = np.cumsum(thick)
        depths = np.insert(depths, 0, 0)

        points = []

        # Vertical segments
        for i in range(len(vs)):
            for d in range(int(depths[i]), int(depths[i + 1])):
                points.append(f"{vs[i]}_{d}")

        # Horizontal connectors
        for i in range(len(vs) - 1):
            start_vs = min(vs[i], vs[i + 1])
            end_vs = max(vs[i], vs[i + 1])
            fixed_depth = int(depths[i + 1])
            for v in range(start_vs, end_vs + 1):
                points.append(f"{v}_{fixed_depth}")

        writer.writerow(points)

print(f"MODEL_GRID_{station}.csv has been created with {num_models} models (one per row).")
print(" ")


# ==========================================================================================
# RF PLOT
# ==========================================================================================

#And now, the RF collection one:
radials = read_rf(f"./input/stream_{station}.h5", format="H5")

# Optionally reduce to 2 traces for debugging
if FAST_VIEW_MODE:
    radials = radials[:1] # just to see if the code run!

# ----------------- helper: bootstrap mean stack -----------------
def bootstrap(array_RFs, nboot=100):
    nrf = array_RFs.count()
    ns  = array_RFs[0].stats.npts
    trace   = np.column_stack([tr.data for tr in array_RFs])
    trace_b = np.full([ns, nboot], np.nan)
    ispl = np.random.randint(0, nrf, (nboot, nrf))
    for ib in range(nboot):
        trace_b[:, ib] = np.mean(trace[:, ispl[ib]], axis=1)
    return np.mean(trace_b, axis=1)

# ----------------- stack for the DF models as before -----------------
stack_matrix = []

for i in range(len(df)):
    vs     = ast.literal_eval(df.loc[i, 'VS'])
    thickn = ast.literal_eval(df.loc[i, 'Thickness'])

    if PRESET in ["NW deep"]:
        vp = [v * vpvs for v in vs]
    else:
        vpvs0 = float(df.loc[i, 'VpVs0'])
        vp = [vs[0] * vpvs0] + [v * vpvs for v in vs[1:]]

    rho   = [1000 * 0.31 * (V ** 0.25) for V in vp]
    model = prs.Model(thickn, rho, vp, vs)
    SYNTH = RFStream()

    for tr in radials:
        baz  = tr.stats.back_azimuth
        slow = tr.stats.slowness / 111.32
        geom = prs.Geometry(baz=[baz], slow=[slow])
        ctrl = prs.Control(verbose=False, rot=1, mults=2, dt=1/sps, npts=number_pts, align=0)
        seis, _ = prs.run(model, geom, ctrl)[0]
        for j, comp in enumerate(['R', 'T', 'Z']):
            new_tr = Trace(data=seis[j].data, header=tr.stats)
            new_tr.stats.channel = f"HH{comp}"
            SYNTH.append(new_tr)

    rfstats(SYNTH)
    SYNTH.filter('highpass', freq=0.05, corners=2, zerophase=True)
    syn_radials = SYNTH.copy().rf(deconvolve='iterative', gauss=0.5, itmax=1000, minderr=0.001, mute_shift=False, normalize=0)
    syn_radials = syn_radials.select(component='R')
    # for tr in syn_radials:
    #     tr.data -= np.mean(tr.data)
    syn_radials.trim2(begin_time, end_time, 'onset')
    syn_radials = RFStream([tr for tr in syn_radials if len(tr.data) == number_pts])

    syn_mean = bootstrap(syn_radials)
    syn_mean /= np.max(np.abs(syn_mean))
    stack_matrix.append(syn_mean)

    percent_complete = (i + 1) / len(df) * 100
    print(f"\rRF plot progress: {percent_complete:.1f}% completed", end='')

print(" ")
print(f"Total models stacked: {len(stack_matrix)}")
print(" ")

# ----------------- plotting -----------------
dt = 1/sps
time_axis = np.linspace(begin_time, end_time, number_pts)


# And we read the RF stack and std
obs_mean = np.loadtxt(f"./input/{station}_trace_mean.txt")
obs_std = np.loadtxt(f"./input/{station}_trace_std.txt")
# and normalize:
norm_factor = np.max(np.abs(obs_mean))# and normalize:
obs_mean /= norm_factor
obs_std /= norm_factor


plt.figure(figsize=(5.4, 6))

# Just to see them fading:
n_models = len(stack_matrix)
top10 = int(np.ceil(0.10 * n_models))

for i, stack in enumerate(stack_matrix):

    # ----------------------------------------------------------
    # Alpha based on model rank
    # Top 10%:       1.00 → 0.90
    # Remaining 90%: 0.20 → 0.02
    # ----------------------------------------------------------
    if i < top10:
        if top10 == 1:
            alpha = 1.0
        else:
            alpha = 1.0 - 0.10 * (i / (top10 - 1))
    else:
        remaining = n_models - top10
        if remaining == 1:
            alpha = 0.02
        else:
            frac = (i - top10) / (remaining - 1)
            alpha = 0.20 - 0.18 * frac

    plt.plot(
        time_axis,
        stack,
        linewidth=1.5,
        alpha=alpha,
        color='violet'
    )



# observational (already in your code below this point)
plt.plot(time_axis, obs_mean, color='black', linewidth=3, linestyle='--')
plt.fill_between(time_axis,
                 obs_mean - obs_std,
                 obs_mean + obs_std,
                 color='silver', alpha=1)

plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Normalized Amplitude", fontsize=14)
plt.tick_params(
    axis='both',
    which='both',
    labelsize=14,
    top=True,
    right=True
)

plt.text(
    0.98, 0.98,
    station,
    transform=plt.gca().transAxes,
    color="black",
    fontsize=22,
    fontweight="bold",
    ha="right",
    va="top"
)

plt.text(
    0.98, 0.92,
    f"Best {number_to_plot} models",
    transform=plt.gca().transAxes,
    color="black",
    fontsize=14,
    ha="right",
    va="top"
)


plt.xlim(begin_time+1 , end_time-5) #let's just see 25 sec
plt.tight_layout()
plt.savefig(f'./figs/rfs_{station}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'./figs/rfs_{station}.png', format='png', dpi=300, bbox_inches='tight')


# sys.exit()



# ---------------------------------------------------------------------------------
# KDE PLOT for Vs–Depth model density
# Presets: "NW deep", "NW", "nofilter", "SE", "SE deep"
# ---------------------------------------------------------------------------------
from scipy.ndimage import gaussian_filter1d

print("Initializing KDE plot... be patient...")

# --- Load MODEL_GRID.csv ---
with open(f"./output/MODEL_GRID_{station}.csv", "r") as f:
    lines = f.readlines()  # each model is one line

# --- FLAG to read only the first row for quick testing ---
if FAST_VIEW_MODE:
    lines = lines[:1]

# --- Build Vs–Depth point list from all models ---
vs_all, depth_all = [], []
for line in lines:
    entries = line.strip().split(",")
    for entry in entries:
        try:
            vs_str, depth_str = entry.split("_")
            vs_all.append(float(vs_str) / 1000.0)        # m/s → km/s
            depth_all.append(float(depth_str) / 1000.0)  # m → km
        except ValueError:
            continue

# --- Stack into KDE input ---
X = np.vstack([vs_all, depth_all]).T

# --- KDE ---
kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
kde.fit(X)

# ---- Grid resolution ----

# --- FLAG to read only the first row for quick testing ---
if FAST_VIEW_MODE:
    VS_STEP = 0.05
    DEPTH_STEP = 0.10
else:
    VS_STEP = 0.02
    DEPTH_STEP = 0.05

VS_RANGE = VS_MAX - VS_MIN
DEPTH_RANGE = DEPTH_MAX - DEPTH_MIN

N_VS = int(np.ceil(VS_RANGE / VS_STEP))
N_DEPTH = int(np.ceil(DEPTH_RANGE / DEPTH_STEP))

vs_grid = np.linspace(VS_MIN, VS_MAX, N_VS)
depth_grid = np.linspace(DEPTH_MIN, DEPTH_MAX, N_DEPTH)

vs_mesh, depth_mesh = np.meshgrid(vs_grid, depth_grid)
grid_coords = np.vstack([vs_mesh.ravel(), depth_mesh.ravel()]).T

# --- Evaluate KDE ---
log_density = kde.score_samples(grid_coords)
density = np.exp(log_density).reshape(vs_mesh.shape)
density /= np.max(density) if np.max(density) > 0 else 1

print("KDE density computed.")

# ==================================================================
#           REPRESENTATIVE MODEL CALCULATION
# ==================================================================

vs_rep = []
depth_rep = []

for i, depth_val in enumerate(depth_grid):
    max_idx = np.argmax(density[i])
    vs_rep.append(vs_grid[max_idx])
    depth_rep.append(depth_val)

vs_rep = np.array(vs_rep)
depth_rep = np.array(depth_rep)

# Optional smoothing
vs_rep_smooth = gaussian_filter1d(vs_rep, sigma=1)

# Export representative model CSV
df_out = pd.DataFrame({
    "Depth_km": depth_rep,
    "Vs_km_s": vs_rep_smooth
})

rep_file = f"./output/repmodel_{station}.csv"
df_out.to_csv(rep_file, index=False)

# # Also copy to ../../REPR_ALL/
# output_dir2 = os.path.abspath(os.path.join(os.getcwd(), "../../REPR_ALL"))
# os.makedirs(output_dir2, exist_ok=True)
# df_out.to_csv(os.path.join(output_dir2, rep_file), index=False)

print(f"Representative model saved to: {rep_file}")
# print(f"Also copied to REPR_ALL folder.\n")

# ==================================================================
#                           KDE PLOT
# ==================================================================

fig, ax = plt.subplots(figsize=(5.4, 6))

# --- Percentile 99.0 clipping + rescale to 0–1 ---
p = 99.0

vmax = np.percentile(density, p)
vmax = max(vmax, 1e-12)  # safety against zero

density_plot = np.clip(density, 0.0, vmax) / vmax

pm = ax.pcolormesh(
    vs_grid, depth_grid, density_plot,
    shading="auto",
    cmap="turbo",
    vmin=0.0,
    vmax=1.0
)

ax.invert_yaxis()
ax.set_xlim(VS_MIN, VS_MAX)
ax.set_ylim(DEPTH_MAX, DEPTH_MIN)

ax.set_xlabel("Vs (km s$^{-1}$)", fontsize=14)
ax.set_ylabel("Depth (km)", fontsize=14)
#ax.set_title(f"{station}: best {number_to_plot} models")

# --- Colorbar ---
cbar = plt.colorbar(pm)
cbar.set_label("Probability Density (normalized)", fontsize=13)
cbar.set_ticks([0, 1])
cbar.ax.tick_params(labelsize=16)

# --- Fix Y-axis ticks at 5 km ---
y_ticks = np.arange(DEPTH_MIN, DEPTH_MAX + 0.0001, 5.0)
ax.set_yticks(y_ticks)

# --- Mirror ticks, labels only lower-left ---
ax.tick_params(
    axis='both', which='both',
    top=True, bottom=True, left=True, right=True,
    labeltop=False, labelright=False,
    labelsize=14#,      # <-- increase axis numbers
    #width=2,           # thicker ticks (optional)
    #length=6           # longer ticks (optional)
)

# --- Add station name + summary text ---
ax.text(
    0.02, 0.125,
    station,
    transform=ax.transAxes,
    color="white",
    fontsize=26,
    fontweight="bold",
    ha="left",
    va="bottom"
)

summary_text = (
    #f"Misfit: {df['Misfit'].min():.2f} – {df['Misfit'].max():.2f}\n"
    #f"Best {number_to_plot} models\n"
    f"Moho:\n"
    f"{moho_mean_km:.2f} ± {moho_std_km:.2f} km"
)

ax.text(
    0.02, 0.02,
    summary_text,
    transform=ax.transAxes,
    color="white",
    fontsize=16,
    ha="left",
    va="bottom",
    linespacing=1.3
)

plt.tight_layout()
plt.savefig(f"./figs/kde_{station}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"./figs/kde_{station}.pdf", dpi=300, bbox_inches='tight')

print(f"[KDE grid] N_VS={N_VS}, N_DEPTH={N_DEPTH} "
      f"(VS_STEP={VS_STEP} km/s, DEPTH_STEP={DEPTH_STEP} km)")
print(f"\nKDE plot completed for station {station} (preset='{PRESET}')\n")