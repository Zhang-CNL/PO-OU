import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from typing import Optional

import hippocampalseq.utils as hseu
from .core import save_wrapper

@save_wrapper
def plot_theta_lfp_segment(
        lfp_data: nap.TsdFrame,
        trough_times: nap.Ts, 
        n_seconds: float,
        segment_start: Optional[float] = None,
        plot_trough_markers: bool = True
    ):
    times = lfp_data.index.values
    if segment_start is None:
        t0 = times[0]
    t1 = min(t0 + n_seconds, times[-1])

    seg_mask = hseu.restrict_indices(times, t0, t1)
    tseg = times[seg_mask]
    phi_seg = lfp_data['Phase Deg'].values[seg_mask]
    lfp_seg = lfp_data['Filtered LFP'].values[seg_mask]

    trough_times = times[trough_times]
    troughs = hseu.restrict_indices(trough_times, t0, t1)

    fig,ax = plt.subplots(2,1, figsize=(12,5), dpi=300, sharex=True)
    ax[0].plot(tseg, lfp_seg, linewidth=0.5)
    ax[0].set_ylabel(r"Theta-filtered LFP ($\mu$V)")
    ax[0].set_title("Theta segment" + "with trough markers" if plot_trough_markers else "")

    ax[1].plot(tseg, phi_seg, linewidth=0.5)
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Time (s)")

    if plot_trough_markers:
        ax[0].vlines(
            trough_times[troughs], 
            ymin=lfp_seg.min(), 
            ymax=lfp_seg.max(), 
            linestyles='dashed',
            alpha=0.6
        )
        ax[1].vlines(
            trough_times[troughs], 
            ymin=0,
            ymax=360,
            linestyles='dashed',
            alpha=0.6
        )

    return fig
    
def plot_theta_cycle_dist(
        lfp_data: nap.TsdFrame,
        bins: int = 40,
        theta_length_s: Optional[tuple] = (0.08, 0.16)
    ):
    cycle_duration = lfp_data['Cycle Duration']
    valid_duration = cycle_duration[cycle_duration > 0]
    fig,ax = plt.subplots(figsize=(6,4), dpi=300)

    ax.hist(valid_duration * 1000, bins=bins)
    ax.set_xlabel("Cycle duration (ms)")
    ax.set_ylabel("Count")

    if theta_length_s:
        ax.axvline(theta_length_s[0] * 1000, color='red', linestyle='--',
            alpha=0.5, label=f"{theta_length_s[0]*1000:.0f}ms"
        )
        ax.axvline(theta_length_s[1] * 1000, color='red', linestyle='--',
            alpha=0.5, label=f"{theta_length_s[1]*1000:.0f}ms"
        )
        ax.legend()
    return fig
