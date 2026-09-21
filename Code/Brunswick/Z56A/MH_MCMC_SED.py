#!/usr/bin/env/ python3

################################################################
#                                                              #
#              MH-MCMC SEDIMENTARY ENVIRONMENT                 #
#                                                              #
#        Bayesian inversion of receiver functions (RFs)        # 
#        Output: MATRIX_res_{station}.csv                      #
#        Uncomment "radials = radials[:1]" for a full run      #
#        By Ademar Fernández, September 2026                   # 
#        Contact: cfernandez@seoe.sc.edu                       #
#                                                              #
################################################################


import pickle
import numpy as np
import obspy
from pyraysum import prs, Geometry, Control
from obspy import read, Trace, Stream, UTCDateTime
import h5py
from rf import read_rf, rfstats, RFStream
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
import time
import os, re


################################################################
###################### SECTION 1: Setup ########################
################################################################

# Detect station name from current folder
station = os.path.basename(os.getcwd()).strip()

# Define the output file name for storing the MATRIX data
output_file = f"./output/MATRIX_res_{station}.csv"

# Specify the stream file
hdf5_filename = f"./input/stream_{station}.h5"

# Read the radial receiver functions using the proper format
radials = read_rf(hdf5_filename, format="H5")

# # In case you want a FAST RUN, please uncomment the following line:
radials = radials[:1]   # <---- comment/uncomment
print(f"\n{len(radials)} traces were loaded.\n")

# mI: The starting model
# Sediment thickness from Boyd et at. (2024)
thickn0 = 322
vs = [828, 3267, 3574, 3722, 3825, 3895, 4500]
thickn = [thickn0, 2813, 4452, 5622, 6536, 14037, 0]

# Initial value for the Vp/Vs0 (sedimentary layer)
vpvs0 = 3.44

# Vp/Vs from Crotwell & Owens (2005):
vpvs = 1.77 

# Step sizes:
proposal_std_vs = 500
proposal_std_thickn = 3000
proposal_std_vpvs0 = 0.2

# Moho variation allowed (based on what we know about the station):
moho_min, moho_max = 23000, 35000

# Total iterations to run
num_iterations = 100000
np.random.seed(44)

# Stream properties
sps = 40 # samples per second
end_time = 30 # stream end
begin_time = -2 
number_pts = int((sps * (end_time - begin_time)) + 1) #+1 to include both endpoints (1281 samples)

# We define the columns of the output:
MATRIX = pd.DataFrame(columns=['Iteration', 'VS', 'Thickness', 'Likelihood', 'Misfit', 'Acceptance'])
likelihood_iterations = []
accepted_models = 0

# We prepare the arrays to start
current_vs = np.array(vs, dtype=float)
current_thickn = np.array(thickn, dtype=float)
current_vpvs0 = float(vpvs0)

# And we read the RF stack and std
trace_mean = np.loadtxt(f"./input/{station}_trace_mean.txt")
trace_std = np.loadtxt(f"./input/{station}_trace_std.txt")
# and normalize:
norm_factor = np.max(np.abs(trace_mean))
trace_mean /= norm_factor
trace_std /= norm_factor

######################################################################
#################### SECTION 2: Functions ############################
######################################################################

# Function to calculate the misfit (log-likelihood) for a given set of model parameters
def calculate_likelihood(vs, thickn, vpvs0, trace_mean, trace_std, radials_clean):
    vp = []
    #vp = [v * vpvs for v in vs]
    # First layer: Vp/Vs = 2.8
    vp.append(vs[0] * vpvs0)

    # Remaining layers: Vp/Vs = 1.60
    for v in vs[1:]:
        vp.append(v * vpvs)

    rho = [1000 * 0.31 * np.power(V, 0.25) for V in vp]
    n_layers = len(vs)
    model = prs.Model(thickn[:n_layers], rho[:n_layers], vp[:n_layers], vs[:n_layers])
    SYNTH = RFStream()

    for trace in radials_clean:
        baz = trace.stats.back_azimuth
        slow = trace.stats.slowness / 111.32
        event_time = UTCDateTime(trace.stats['event_time'])
        geom = prs.Geometry(baz=[baz], slow=[slow])  # Ensure they are iterable lists
        ctrl = prs.Control(verbose=False, rot=1, mults=2, dt=1 / sps, npts=number_pts, align=0)
        result = prs.run(model, geom, ctrl)
        seis, rfs = result[0]

        # Directly append synthetic traces to SYNTH without writing/reading SAC files
        for i, component in enumerate(['R', 'T', 'Z']):
            new_trace = Trace(data=seis[i].data, header=trace.stats)
            new_trace.stats.channel = f"HH{component}"
            if not isinstance(new_trace.stats.processing, list):
                new_trace.stats.processing = []
            SYNTH.append(new_trace)

    rfstats(SYNTH)
    SYNTH.filter('highpass', freq=0.05, corners=2, zerophase=True)
    rf_iter = SYNTH.copy().rf(deconvolve='iterative', gauss=0.5, itmax=1000, minderr=0.001, mute_shift=False, normalize=0)
    syn_radials = rf_iter.select(component='R')

    syn_radials.trim2(begin_time, end_time, 'onset')


    # Faster RF stack computation (No bootstrapping)
    def compute_rf_stack(array_RFs):
        return np.mean([tr.data for tr in array_RFs], axis=0)

    syn_trace_mean = compute_rf_stack(syn_radials)  # Using the optimized function

    # Normalize synthetic RF
    syn_trace_mean /= np.max(np.abs(syn_trace_mean))

    #Okay, so since I am comparing the entire thing now:
    misfit = (1/len(trace_mean)) * np.sum(((trace_mean - syn_trace_mean) ** 2)/ (trace_std ** 2))
    # # and here I convert misfit to log likelihood
    return np.exp(-0.5 * misfit), misfit


def propose_noise_element(current_vs, current_thickn, proposal_std_vs, proposal_std_thickn):
    vs = current_vs.copy()
    thickn = current_thickn.copy()

    # randomly choose whether to perturb vs or thickn
    if np.random.rand() < 0.5:
        # perturb a single velocity (can be any layer, including vs[0])
        idx = np.random.randint(0, len(vs))
        vs[idx] += np.random.randn() * proposal_std_vs
    else:
        # perturb a single thickness, but skip sediment [0] and half-space [-1]
        idx = np.random.randint(1, len(thickn) - 1)
        thickn[idx] += np.random.randn() * proposal_std_thickn

    # enforce bounds (hard box)
    vs[0] = np.clip(vs[0], 450, 1250)        # sediment velocity range
    vs[1:] = np.clip(vs[1:], 2400, 4500)     # crustal velocities
    thickn[0] = thickn0                      # lock sediment thickness
    thickn = np.clip(thickn, 0, None)
    thickn[-1] = 0                           # lock half-space thickness

    return vs, thickn


def propose_noise_model(current_vs, current_thickn, proposal_std_vs, proposal_std_thickn):
    # shrink jump size for model-wide moves to 60%
    std_vs  = proposal_std_vs * 0.6
    std_thk = proposal_std_thickn * 0.6

    # global perturbation of ALL layers
    vs     = current_vs     + np.random.randn(len(current_vs))     * std_vs
    thickn = current_thickn + np.random.randn(len(current_thickn)) * std_thk

    # SE hard-box constraints
    vs[0]  = np.clip(vs[0], 450, 1250)     # sediment
    vs[1:] = np.clip(vs[1:], 2400, 4500)   # crust
    thickn = np.clip(thickn, 0, None)
    thickn[0]  = thickn0
    thickn[-1] = 0

    return vs, thickn

#########################################################################
############### SECTION 3: Pickling part one ############################
#########################################################################

def save_checkpoint(checkpoint_path,
                    iteration,
                    current_vs,
                    current_thickn,
                    current_vpvs0,
                    current_likelihood,
                    current_misfit,
                    best_vs_so_far,
                    best_thickn_so_far,
                    lowest_misfit,
                    accepted_models,
                    likelihood_iterations,
                    config):

    state = {
        "iteration": iteration,
        "current_vs": current_vs,
        "current_thickn": current_thickn,
        "current_vpvs0": current_vpvs0,
        "current_likelihood": current_likelihood,
        "current_misfit": current_misfit,
        "best_vs_so_far": best_vs_so_far,
        "best_thickn_so_far": best_thickn_so_far,
        "lowest_misfit": lowest_misfit,
        "accepted_models": accepted_models,
        "likelihood_iterations": likelihood_iterations,
        "rng_state_numpy": np.random.get_state(),
        "config": config
    }

    # --- 1. Ensure archive folder exists ---
    archive_folder = "pickle.files"
    os.makedirs(archive_folder, exist_ok=True)

    # --- 2. Save main working checkpoint ---
    with open(checkpoint_path, "wb") as f:
        pickle.dump(state, f)

    # --- 3. Save versioned archive copy ---
    archive_path = os.path.join(archive_folder, f"checkpoint_{iteration}.pkl")
    with open(archive_path, "wb") as f:
        pickle.dump(state, f)

    print(f"[checkpoint] Saved: {checkpoint_path} (working) and {archive_path} (archive)")

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    return state


#########################################################################
############### SECTION 4: Pickling part two ############################
#########################################################################

# -------------------- Resume / Fresh start setup (auto) --------------------
checkpoint_path = "checkpoint.pkl"
resume_from_checkpoint = os.path.exists(checkpoint_path)

if resume_from_checkpoint:
    try:
        state = load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"[warn] Failed to load checkpoint '{checkpoint_path}': {e}. Starting fresh.")
        resume_from_checkpoint = False

    # Resume from *exactly* after the last saved iteration
    last_saved_iter = state["iteration"]
    start_iteration = last_saved_iter + 1

    # Keep the chain counter continuous with the saved checkpoint
    chain_iteration = last_saved_iter     # resume from last saved chain count
    first_kept_iter = 0                          # ensures we're already post-burn-in

    current_vs = np.array(state["current_vs"])
    current_thickn = np.array(state["current_thickn"])
    current_vpvs0 = float(state["current_vpvs0"])
    current_likelihood = state["current_likelihood"]
    current_misfit = state["current_misfit"]
    best_vs_so_far = np.array(state["best_vs_so_far"])
    best_thickn_so_far = np.array(state["best_thickn_so_far"])
    lowest_misfit = state["lowest_misfit"]
    accepted_models = state["accepted_models"]
    likelihood_iterations = state["likelihood_iterations"]

    # --- Restore iteration counters properly ---
    stop_iter = start_iteration + num_iterations  # ensure stop condition valid

    # Restore RNG so the chain continues deterministically
    np.random.set_state(state["rng_state_numpy"])

    # Reload CSV so you keep the history you already wrote
    if os.path.exists(output_file):
        MATRIX = pd.read_csv(output_file)

        # --- Trim any rows beyond the saved iteration so we don't duplicate ---
        if "Iteration" in MATRIX.columns:
            MATRIX["Iteration"] = pd.to_numeric(MATRIX["Iteration"], errors="coerce")
            before_rows = len(MATRIX)
            MATRIX = MATRIX[MATRIX["Iteration"] <= last_saved_iter].copy()
            MATRIX.reset_index(drop=True, inplace=True)
            MATRIX.to_csv(output_file, index=False)
        else:
            print("[warn] 'Iteration' column not found in MATRIX; skipping trim.")

    print("-------------------------------------------------------------------------")
    print(f"Resuming from checkpoint: {checkpoint_path} (iteration {last_saved_iter})")
    print("-------------------------------------------------------------------------")
    print(" ")

else:
    start_iteration = 0
    current_likelihood, current_misfit = calculate_likelihood(
        current_vs, current_thickn, current_vpvs0, trace_mean, trace_std, radials
    )
    best_vs_so_far = current_vs.copy()
    best_thickn_so_far = current_thickn.copy()
    lowest_misfit = current_misfit
    print("Starting fresh run.")


# In order to start once we accept our first model, we will do:
if not resume_from_checkpoint:
    first_kept_iter = None
    chain_iteration = 0
    stop_iter = None
target_extra_after_first = num_iterations

config = {
    "station": station,
    "output_file": output_file,
    "hdf5_filename": hdf5_filename,
    "num_iterations": num_iterations,
    "vpvs": vpvs,
    "proposal_std_vs": proposal_std_vs,
    "proposal_std_thickn": proposal_std_thickn,
    "proposal_std_vpvs0": proposal_std_vpvs0
}


######################################################################
########### SECTION 5: METROPOLIS-HASTINGS SAMPLING LOOP  ############
######################################################################

I = start_iteration
while True:

    p_model = 0.45      # global jiggle
    p_element = 0.45    # single-element jiggle
    p_vpvs0 = 0.10       # vpvs0 jiggle (even though we don't use this object)

    u = np.random.rand()

    # fixed step size for Vs[0]
    proposal_std_vs0 = 60.0

    if u < p_model:
        # Global (model-wide) perturbation
        proposed_vs, proposed_thickn = propose_noise_model(current_vs, current_thickn, proposal_std_vs, proposal_std_thickn)
        proposed_vpvs0 = current_vpvs0

    elif u < p_model + p_element:
        # Single-element perturbation (vs or thickn element, excluding thickn[0] & thickn[-1])
        proposed_vs, proposed_thickn = propose_noise_element(current_vs, current_thickn, proposal_std_vs, proposal_std_thickn)
        proposed_vpvs0 = current_vpvs0

    else:
        # --- Vp/Vs0 move (sometimes coupled with Vs[0]) ---
        proposed_vs = current_vs.copy()
        proposed_thickn = current_thickn.copy()

        # propose new Vp/Vs0 value
        vpvs0 = float(current_vpvs0) + np.random.randn() * proposal_std_vpvs0
        lo, hi = 2.5, 3.5
        if vpvs0 < lo:
            vpvs0 = lo + (lo - vpvs0)
        if vpvs0 > hi:
            vpvs0 = hi - (vpvs0 - hi)
        proposed_vpvs0 = round(min(max(vpvs0, lo), hi), 2)

        # --- 50% chance to also perturb Vs[0] ---
        if np.random.rand() < 0.5:
            v0 = proposed_vs[0] + np.random.randn() * proposal_std_vs0
            vlo, vhi = 450.0, 1250.0
            if v0 < vlo:
                v0 = vlo + (vlo - v0)
            if v0 > vhi:
                v0 = vhi - (v0 - vhi)
            proposed_vs[0] = int(round(min(max(v0, vlo), vhi)))

    # --- compute total thickness ---
    total_thickness = np.sum(proposed_thickn)

    # count all iterations only after burn-in starts
    if first_kept_iter is not None:
        chain_iteration += 1

    # --- geological rejection criteria (SE pure MH) ---
    if (
        proposed_vs[0] < 450 or proposed_vs[0] > 1250
        or np.any(proposed_vs[1:] < 2400) or np.any(proposed_vs[1:] > 4500) or (proposed_vs[-1] < 3800) \
        or proposed_thickn[0] != thickn0
        or proposed_thickn[-1] != 0
        or np.any(proposed_thickn[1:-1] < 1000)
        or total_thickness < moho_min or total_thickness > moho_max
        or np.any(np.diff(proposed_vs) < 25)
        or np.any(np.diff(proposed_thickn[:-1]) < 0)
        or np.any(proposed_thickn[:-1] > 30000)
        or np.sum(proposed_thickn[:-1] > 20000) > 2
    ):
        if first_kept_iter is not None:
            # use RAW count since burn-in, not the chain counter
            raw_since_burnin = I - first_kept_iter + 1
            if raw_since_burnin % 50 == 0:
                acceptance_rate = 100 * accepted_models / raw_since_burnin
                print()
                print(f"--- ACCEPTANCE RATIO: {acceptance_rate:.2f}% ---")
                print()
        #print(f"I am gonna pass {I}")
        I += 1
        continue



    # Attempt to compute log-likelihood and misfit for the proposed STRUCTURE
    try:
        proposed_likelihood, proposed_misfit = calculate_likelihood(
            proposed_vs, proposed_thickn, proposed_vpvs0,  # <-- use proposed_vpvs0
            trace_mean, trace_std, radials
        )
    except ValueError:
        continue  # Reject iterations that fail likelihood evaluation


    # --- METROPOLIS-HASTINGS ACCEPTANCE STEP ---
    if proposed_likelihood > current_likelihood:
        current_vs = proposed_vs.copy()
        current_thickn = proposed_thickn.copy()
        current_likelihood = proposed_likelihood
        current_misfit = proposed_misfit
        current_vpvs0 = proposed_vpvs0
        accepted_models += 1
        acceptance = 1

        if first_kept_iter is None:
            first_kept_iter = I
            stop_iter = first_kept_iter + target_extra_after_first
            iterations_wasted = first_kept_iter
            chain_iter_counter = 0
            chain_iteration = 1
            print(f"\nStarting from iter {I+1}... iterating {num_iterations} from now on! \n")

        chain_iter_counter = chain_iteration
        print(f"Iter {chain_iter_counter:6d}: {'ACCEPTED':15s}, Misfit: {current_misfit:.2f}")

    else:
        likelihood_diff = proposed_likelihood / current_likelihood
        accept_ratio = likelihood_diff
        rand_val = np.random.rand()

        if rand_val < accept_ratio:
            current_vs = proposed_vs.copy()
            current_thickn = proposed_thickn.copy()
            current_likelihood = proposed_likelihood
            current_misfit = proposed_misfit
            current_vpvs0 = proposed_vpvs0
            accepted_models += 1
            acceptance = 1

            if first_kept_iter is None:
                first_kept_iter = I
                stop_iter = first_kept_iter + target_extra_after_first
                iterations_wasted = first_kept_iter
                chain_iter_counter = 0
                chain_iteration = 1
                print(f"\nStarting from iter {I+1}... \n")

            chain_iter_counter = chain_iteration
            print(f"Iter {chain_iter_counter:6d}: {'Prob. accepted':15s}, Misfit: {current_misfit:.2f}")
        else:
            acceptance = 0
            print(f"Iter {chain_iteration:6d}: Rejected: Misfit = {current_misfit:.2f}, Best Misfit = {lowest_misfit:.2f} ({np.sum(best_thickn_so_far)/1000:.1f} km)")
            
    # STORE RESULTS IN DATAFRAME
    data_row = {
        'Iteration': chain_iteration if first_kept_iter is not None else 0,
        'VS': json.dumps([round(v) for v in current_vs]),
        'Thickness': json.dumps([round(t) for t in current_thickn]),
        'VpVs0': round(current_vpvs0, 3),
        'Likelihood': current_likelihood,
        'Misfit': round(current_misfit, 2),
        'Acceptance': acceptance
    }

    # Append to DataFrame and save results incrementally
    MATRIX = pd.concat([MATRIX, pd.DataFrame([data_row])], ignore_index=True)
    MATRIX.to_csv(output_file, index=False)

    SAVE_EVERY = 50

    # Save checkpoint right after writing to MATRIX
    if first_kept_iter is not None and chain_iteration > 0 and chain_iteration % SAVE_EVERY == 0:
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            iteration=chain_iteration,   # <-- use the true chain iteration
            current_vs=current_vs,
            current_thickn=current_thickn,
            current_vpvs0=current_vpvs0,
            current_likelihood=current_likelihood,
            current_misfit=current_misfit,
            best_vs_so_far=best_vs_so_far,
            best_thickn_so_far=best_thickn_so_far,
            lowest_misfit=lowest_misfit,
            accepted_models=accepted_models,
            likelihood_iterations=likelihood_iterations,
            config=config
        )
        print()
        print(f"[checkpoint] Saved checkpoint at chain iteration {chain_iteration} -> {checkpoint_path}")
        print()

    # UPDATE BEST MODEL FOUND SO FAR
    if current_misfit < lowest_misfit:
        best_vs_so_far = current_vs.copy()
        best_thickn_so_far = current_thickn.copy()
        lowest_misfit = current_misfit

    # we check the acceptance ratio each 50 iter
    if first_kept_iter is not None and chain_iteration % 50 == 0:
        acceptance_rate = 100 * accepted_models / chain_iteration
        print()
        print(f"--- ACCEPTANCE RATIO: {acceptance_rate:.2f}% ---")
        print()



    # --- safeguard: ensure stop_iter is valid ---
    if stop_iter is None:
        stop_iter = I + target_extra_after_first


    # --- dynamic stop condition ---
    if (first_kept_iter is not None) and (I >= stop_iter):
        print(f"\nReached target of {target_extra_after_first:,} iterations after first acceptance "
              f"(I = {I}). Total wasted burn-in: {first_kept_iter:,} iterations.\n")
        break

    I += 1

#  **FINAL SUMMARY OUTPUT**
print(f"\nAccepted Models: {accepted_models} / {num_iterations}")
print(f"Best Misfit: {lowest_misfit}, Layers: {len(best_vs_so_far)}")
#print(f"Total runtime: {time.time() - start_time:.2f} seconds")


save_checkpoint(
    checkpoint_path=checkpoint_path,
    iteration=num_iterations,
    current_vs=current_vs,
    current_thickn=current_thickn,
    current_vpvs0=current_vpvs0,
    current_likelihood=current_likelihood,
    current_misfit=current_misfit,
    best_vs_so_far=best_vs_so_far,
    best_thickn_so_far=best_thickn_so_far,
    lowest_misfit=lowest_misfit,
    accepted_models=accepted_models,
    likelihood_iterations=likelihood_iterations,
    config=config
)

