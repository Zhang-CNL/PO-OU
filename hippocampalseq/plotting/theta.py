import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from collections.abc import Iterable
from typing import Optional

from torch._dynamo import exc

import hippocampalseq.utils as hseu
from .core import save_wrapper

@save_wrapper
def plot_theta_lfp_segment(
        lfp_data: nap.TsdFrame,
        trough_times: nap.Ts, 
        trough_indices: np.ndarray,
        n_seconds: float,
        segment_start: Optional[float] = None,
        plot_trough_markers: bool = True
    ):
    times = lfp_data.index.values
    if segment_start is None:
        t0 = times[0]
    else:
        t0 = segment_start
    t1 = min(t0 + n_seconds, times[-1])

    seg_mask = hseu.restrict_indices(times, t0, t1)
    tseg = times[seg_mask]
    phi_seg = lfp_data['Phase Deg'].values[seg_mask]
    lfp_seg = lfp_data['Filtered LFP'].values[seg_mask]

    trough_times = times[trough_indices]
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
            ymin=np.min(lfp_seg), 
            ymax=np.max(lfp_seg), 
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
    
@save_wrapper
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

@save_wrapper
def plot_theta_phase_assignment(
        spike_info_with_phase: dict[int,nap.TsdFrame],
        lfp_with_cycles: nap.TsdFrame,
        lfp_trough_indices: Optional[np.ndarray] = None,
        excitatory_neurons: Optional[np.ndarray|Iterable[int]] = None,
        n_cells: int = 6,
        time_window: int = 10,
        skip_top_n: int = 3,
        seed: int = 42
    ):
    rng = np.random.default_rng(seed)
    if excitatory_neurons is None:
        candidates = list(spike_info_with_phase.keys())
    else:
        eset = set(excitatory_neurons)
        candidates = [c for c in spike_info_with_phase.keys() if c in eset]
    
    sorted_by_count = sorted(
        candidates,
        key=lambda c: len(spike_info_with_phase[c]),
        reverse=True
    )
    top_third_size = max(1, len(sorted_by_count) // 3)
    plottable_cells = sorted_by_count[skip_top_n:top_third_size]

    lfp_times = lfp_with_cycles['Filtered LFP'].index.values
    start = lfp_times[len(lfp_times) // 2]
    end   = start + time_window
    window = nap.IntervalSet(start, end)

    cells_in_window = []
    for cell_id in plottable_cells:
        spike_times = spike_info_with_phase[cell_id].index.values
        if np.any((spike_times >= start) & (spike_times <= end)):
            cells_in_window.append(cell_id)

    if len(cells_in_window) > n_cells:
        plot_cells = np.sort(rng.choice(cells_in_window, size=n_cells, replace=False))
    else:
        plot_cells = np.sort(cells_in_window)

    # Figure starts
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    lfp_window = lfp_with_cycles['Filtered LFP'].restrict(window)
    ax.plot(lfp_window.index, lfp_window.values, 'k-', label='Filtered LFP')

    # Trough markers
    if lfp_trough_indices is not None:
        trough_times         = lfp_times[lfp_trough_indices]
        trough_times_window  = trough_times[(trough_times >= start) &
                                            (trough_times <= end)]
        for t in trough_times_window:
            ax.axvline(t, color='red', linestyle='--', alpha=0.5)

    # Spike rasters below the LFP trace
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(plot_cells), 1)))
    for i, cell_id in enumerate(plot_cells):
        info        = spike_info_with_phase[cell_id]
        spike_times = info.index.values
        spike_mask  = (spike_times >= start) & (spike_times <= end)
        n_spikes_total = len(spike_times)
        ax.scatter(spike_times[spike_mask],
                np.zeros(np.sum(spike_mask)) - 50 - i * 20,
                color=colors[i], s=50, marker='|',
                edgecolors='none',
                label=f'Cell {cell_id} (n={n_spikes_total})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (µV)')
    ax.set_title(
        f'Theta LFP with Detected Cycles '
        f'({len(plot_cells)} cells from top third of excitatory, ',
        fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

@save_wrapper
def plot_cell_phase_polar(
        spike_info_with_phase: dict[int, nap.TsdFrame],
        n_cells: int = 1,
        bins: int = 24
    ):
    cell_ids = list(spike_info_with_phase.keys())[:n_cells]
    n_plots = len(cell_ids)
    
    fig, axes = plt.subplots(1, n_plots, subplot_kw={'projection': 'polar'}, figsize=(4*n_plots, 4))
    
    # make axes iterable even if n_cells=1
    if n_plots == 1:
        axes = [axes]

    for ax, cid in zip(axes, cell_ids):
        phases = np.deg2rad(spike_info_with_phase[cid]['Phase Deg'])
        counts, edges = np.histogram(phases, bins=bins, range=(0, 2*np.pi))
        centers = (edges[:-1] + edges[1:]) / 2

        ax.bar(centers, counts, width=edges[1]-edges[0], alpha=0.7)
        ax.set_title(f"Cell {cid}", pad=12)
        ax.set_rticks([])  # cleaner look

    plt.tight_layout()
    plt.show()
    return fig