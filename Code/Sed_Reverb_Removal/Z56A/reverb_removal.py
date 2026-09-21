#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rf
from rf import read_rf, RFStream
from obspy.core.trace import Trace
from scipy.signal import correlate
from matplotlib.ticker import MultipleLocator


# ============================================================
# Station
# ============================================================

station = os.path.basename(os.getcwd()).strip()
input_file = f"./input/stream_{station}_RAW_STILL_REVERB.h5"

# We create the folder to save the figures.
os.makedirs("./figs", exist_ok=True)

# And the output
os.makedirs("./output", exist_ok=True)


# ============================================================
# Functions
# ============================================================

def max_p_onset(tr):
    onset = tr.stats.onset

    tr_copy = tr.copy()
    tr_copy.trim(onset - 5, onset + 35)
    tr_copy.taper(max_percentage=0.05)

    st = tr_copy.stats
    rel_time = st.onset

    winP_st = rel_time - st.starttime - 0.1
    winP_end = rel_time - st.starttime + 1.8

    t = np.arange(st.npts) / st.sampling_rate
    window = tr_copy.data[(t >= winP_st) & (t <= winP_end)]

    if len(window) == 0:
        tr.stats.max_abs_P = np.nan
        tr.stats.max_P = np.nan
        return np.nan, np.nan

    max_abs = np.max(np.abs(window))
    max_p = np.max(window)

    tr.stats.max_abs_P = max_abs
    tr.stats.max_P = max_p

    return max_abs, max_p


def quality_check(rfs):
    stream_rf = RFStream()

    for tr in rfs:
        tr = tr.copy()

        onset = tr.stats.onset
        tr.trim(onset - 5, onset + 45)

        if len(tr.data) == 0:
            continue

        tr.normalize()

        max_rf_r = np.max(tr.data)
        _, max_P = max_p_onset(tr)

        if np.isnan(max_P):
            continue

        if max_P == max_rf_r:
            stream_rf.append(tr)

    return stream_rf


def apply_reverberation_filter(cha_data):
    result_stream = []
    auto_c = RFStream()

    for tr in cha_data:
        lead_time = tr.stats.onset - tr.stats.starttime
        relative_time = tr.times() - lead_time

        mask = np.array((relative_time < 0) | (relative_time > 5.0))
        masked = np.ma.masked_array(tr.data, mask=mask)

        if masked.count() == 0:
            continue

        loc = np.argmax(masked)
        dt = relative_time[loc]

        data = correlate(tr.data, tr.data, mode="full")

        max_auto = np.max(np.abs(data))
        if max_auto == 0:
            continue

        data = data / max_auto
        data = data[len(data) // 2:]

        r0 = -np.min(data)
        Dt = np.argmin(data) / tr.stats.sampling_rate

        tr_copy = tr.copy()

        resonance_filter = np.zeros(len(tr.data))
        resonance_filter[0] = 1.0

        resonance_index = int(Dt * tr.stats.sampling_rate)

        if resonance_index >= len(resonance_filter):
            continue

        resonance_filter[resonance_index] = r0

        filtered = np.convolve(tr_copy.data, resonance_filter, mode="full")
        tr_copy.data = filtered[:len(tr_copy.data)]

        if tr.data.shape != tr_copy.data.shape:
            raise ValueError("Input/output length mismatch detected in reverberation removal routine.")

        tr_copy.stats.update({
            "t1_offset": dt,
            "t2_offset": Dt - dt,
            "t3_offset": Dt,
            "r0": r0,
        })

        result_stream.append(tr_copy)

        auto_c_temp = Trace(data=data.copy())
        auto_c_temp.stats = tr_copy.stats.copy()
        auto_c_temp.stats.starttime = tr_copy.stats.onset

        t0 = auto_c_temp.stats.starttime
        auto_c_temp.trim(t0, t0 + 30)

        auto_c.append(auto_c_temp)

    return rf.RFStream(result_stream), auto_c


def plot_auto(auto_c, station):
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_axes([0.1, 0.07, 0.85, 0.85])

    ax1.set_ylim(26, 95)
    ax1.set_xlim(0, 10)
    ax1.set_yticks(np.linspace(30, 90, 3))

    for tr in auto_c:
        dist = tr.stats.distance
        time = np.arange(tr.stats.npts) * tr.stats.delta

        max_amp = np.max(np.abs(tr.data))
        if max_amp == 0:
            continue

        auto_data = tr.data / max_amp

        r0 = tr.stats.r0
        Dt = tr.stats.t3_offset

        ax1.plot(time, 5 * auto_data + dist, lw=0.15, color="black", alpha=0.5)
        ax1.plot(Dt, 5 * -r0 + dist, marker="o", color="slateblue", markersize=1, alpha=0.8)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    plt.setp(ax1.get_xticklabels(), fontsize=8)

    ax1.grid(which="major", axis="x", color="LightGrey", linestyle="--", linewidth=0.45, alpha=0.75)

    ax1.set_ylabel("Distance")
    ax1.set_xlabel("Time")
    ax1.set_title(f"AutoC for {station}", fontsize=11)

    fig.savefig(f"./figs/{station}_autoC.pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def curate_rvr(rfs, station):
    stream_rever_rmv_cur = RFStream()

    for tr in rfs:
        dt_rf = tr.stats.t1_offset
        Dt = tr.stats.t3_offset

        if 1.9 * dt_rf < Dt < 3.1 * dt_rf:
            stream_rever_rmv_cur.append(tr)

    print(f"Length after removing large DT = {len(stream_rever_rmv_cur)}")

    if len(stream_rever_rmv_cur) > 0:
        kw = {
            "fig_width": 6,
            "fillcolors": ("steelblue", "gainsboro"),
            "trace_height": 0.1,
            "show_vlines": "True",
            "scale": 2.5,
        }

        stream_rever_rmv_cur.sort(["distance"]).plot_rf(**kw)
        fig = plt.gcf()

        fig.savefig(f"./figs/{station}_q_rmvDt.pdf", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

        stream_rever_rmv_cur.write(f"./output/stream_{station}.h5", "H5")

    return stream_rever_rmv_cur


# ============================================================
# Read RFs
# ============================================================

rfs_raw = read_rf(input_file, format="H5")
n_before = len(rfs_raw)

print(f"Station: {station}")
print(f"Loaded {n_before} RFs.")


# ============================================================
# Fix mixed network/location metadata
# ============================================================

changed = False

networks = [tr.stats.network for tr in rfs_raw]
majority_network = max(set(networks), key=networks.count)

if any(net != majority_network for net in networks):
    print(f"Majority network = {majority_network}")

    for tr in rfs_raw:
        tr.stats.network = majority_network

    changed = True

locations = [tr.stats.location for tr in rfs_raw]
majority_location = max(set(locations), key=locations.count)

if any(loc != majority_location for loc in locations):
    print(f"Majority location = '{majority_location}'")

    for tr in rfs_raw:
        tr.stats.location = majority_location

    changed = True

if changed:
    print("Updating metadata...")
    rfs_raw.write(input_file, format="H5")


# ============================================================
# Quality control
# ============================================================

rfs_quality = quality_check(rfs_raw)
n_quality = len(rfs_quality)

print(f"No. of RF before/after = {n_before} / {n_quality}")

kw = {
    "fig_width": 6,
    "fillcolors": ("seagreen", "white"),
    "trace_height": 0.1,
    "show_vlines": "True",
    "scale": 2.5,
}

rfs_quality.select(component="R").sort(["back_azimuth"]).plot_rf(**kw)

fig = plt.gcf()
fig.savefig(f"./figs/{station}_q.pdf", bbox_inches="tight", pad_inches=0.2)
plt.close(fig)


# ============================================================
# Reverberation removal
# ============================================================

print("Applying reverberation filter...")

rfs_quality_rvr, auto_c = apply_reverberation_filter(rfs_quality)

print(f"RFs after reverberation filter: {len(rfs_quality_rvr)}")

if len(auto_c) > 0:
    plot_auto(auto_c, station)

kw = {
    "fig_width": 6,
    "fillcolors": ("maroon", "white"),
    "trace_height": 0.1,
    "show_vlines": "True",
    "scale": 2.5,
}

rfs_quality_rvr.select(component="R").sort(["back_azimuth"]).plot_rf(**kw)
fig = plt.gcf()
fig.savefig(f"./figs/{station}_q_rvr.pdf", bbox_inches="tight", pad_inches=0.2)
plt.close(fig)


# ============================================================
# Final DT curation
# ============================================================

final_rfs = curate_rvr(rfs_quality_rvr, station)
n_final = len(final_rfs)


# ============================================================
# Summary
# ============================================================

with open("./output/reverb_removal_summary.txt", "w") as summary:
    summary.write(f"--------- Station {station} -------------\n")
    summary.write(f"No. of RF before/after = {n_before} / {n_quality}\n")
    summary.write(f"Length after removing large DT = {n_final}\n")

print("Saved reverb_removal_summary.txt")
print(f"Finished station: {station}")