from pathlib import Path
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from scipy.stats import circmean, circstd


def plot_theta_phase_assignment(spike_info_with_phase, lfp_with_cycles,
                                excitatory_neurons=None,
                                n_cells=6, time_window=10,
                                skip_top_n=3,
                                random_seed=83):


    print(f"  Cells with phase data: {len(spike_info_with_phase)}")
    rng = np.random.default_rng(random_seed)

    # Filter to excitatory if provided
    if excitatory_neurons is not None:
        excitatory_set = set(excitatory_neurons)
        candidate_cells = [c for c in spike_info_with_phase.keys()
                            if c in excitatory_set]
        print(f"  Excitatory cells: {len(candidate_cells)}")
    else:
        candidate_cells = list(spike_info_with_phase.keys())

    # Sort by spike count, take top third, skip outliers
    sorted_by_count = sorted(
        candidate_cells,
        key=lambda c: len(spike_info_with_phase[c]),
        reverse=True,
    )
    top_third_size = max(1, len(sorted_by_count) // 3)
    top_third      = sorted_by_count[skip_top_n : top_third_size]
    print(f"  Top third (skipping top {skip_top_n}): "
            f"{len(top_third)} cells, "
            f"spike counts {len(spike_info_with_phase[top_third[0]])}-"
            f"{len(spike_info_with_phase[top_third[-1]])}")

    # Time window
    lfp_times  = lfp_with_cycles['filtered_lfp'].index.values
    start_time = lfp_times[len(lfp_times) // 2]
    end_time   = start_time + time_window
    window     = nap.IntervalSet(start_time, end_time)

    # Of the top third, keep those that actually fire in the window
    cells_in_window = []
    for cell_id in top_third:
        spike_times = spike_info_with_phase[cell_id].index.values
        if np.any((spike_times >= start_time) & (spike_times <= end_time)):
            cells_in_window.append(cell_id)

    if len(cells_in_window) > n_cells:
        plot_cells = sorted(rng.choice(cells_in_window, size=n_cells, replace=False).tolist())
    else:
        plot_cells = sorted(cells_in_window)

    # Figure starts
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    lfp_window = lfp_with_cycles['filtered_lfp'].restrict(window)
    ax.plot(lfp_window.index, lfp_window.values, 'k-', label='Filtered LFP')

    # Trough markers
    trough_indices       = lfp_with_cycles['trough_indices']
    trough_times         = lfp_times[trough_indices]
    trough_times_window  = trough_times[(trough_times >= start_time) &
                                        (trough_times <= end_time)]
    for t in trough_times_window:
        ax.axvline(t, color='red', linestyle='--', alpha=0.5)

    # Spike rasters below the LFP trace
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(plot_cells), 1)))
    for i, cell_id in enumerate(plot_cells):
        info        = spike_info_with_phase[cell_id]
        spike_times = info.index.values
        spike_mask  = (spike_times >= start_time) & (spike_times <= end_time)
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


def plot_cell_phase_polar(spike_info_with_phase, n_cells=1, bins=24):
    cell_ids = list(spike_info_with_phase.keys())[:n_cells]
    n_plots = len(cell_ids)
    
    fig, axes = plt.subplots(1, n_plots, subplot_kw={'projection': 'polar'}, figsize=(4*n_plots, 4))
    
    # make axes iterable even if n_cells=1
    if n_plots == 1:
        axes = [axes]

    for ax, cid in zip(axes, cell_ids):
        phases = np.deg2rad(spike_info_with_phase[cid]['theta_phase'])
        counts, edges = np.histogram(phases, bins=bins, range=(0, 2*np.pi))
        centers = (edges[:-1] + edges[1:]) / 2

        ax.bar(centers, counts, width=edges[1]-edges[0], alpha=0.7)
        ax.set_title(f"Cell {cid}", pad=12)
        ax.set_rticks([])  # cleaner look

    plt.tight_layout()
    plt.show()

# def LFP_position_alignment(lfp_data, position_data):