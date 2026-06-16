from scipy.signal import filtfilt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from math import ceil, sqrt

def plot_place_fields(
    initial_conditions,
    pf_results,
    spike_info,
    position_data,
    excitatory_neurons=None,
    n_cells_to_plot=20,
    raster_cell_idx=0,
    max_cells_2d=100,
):

    Field_Data        = pf_results["Field_Data"]
    Field_Data_Linear = pf_results["Field_Data_Linear"]
    cell_ids          = np.array(pf_results["cell_ids"])
    x_edges           = pf_results["x_bin_edges"]
    y_edges           = pf_results["y_bin_edges"]
    bin_size          = pf_results["bin_size"]
    velocity_cutoff   = pf_results["velocity_cutoff"]

    n_cells = Field_Data.shape[2]
    if excitatory_neurons is None:
        excitatory_neurons = set()
    else:
        excitatory_neurons = set(excitatory_neurons)

    fr_cutoff = initial_conditions.get(
        "place_field_firing_rate_cutoff",
        pf_results.get("firing_rate_cutoff", 1.0),
    )

    # Containers — initialised so later blocks can run even if linear is empty
    linear_figs         = {}
    sorted_cell_indices = np.array([], dtype=int)
    pos_centers         = None
    lin                 = None
    two_d_fig           = None

    # Linear Fields
    if Field_Data_Linear is not None:
        lin        = Field_Data_Linear
        n_lin_bins = lin.shape[0]

        lin_peak           = np.nanmax(lin, axis=0)
        has_field          = lin_peak > fr_cutoff
        cells_with_fields  = np.where(has_field)[0]

        peak_locs           = np.argmax(lin[:, cells_with_fields], axis=0)
        sort_idx            = np.argsort(peak_locs)
        sorted_cell_indices = cells_with_fields[sort_idx]

        pos_bins    = np.arange(n_lin_bins) * bin_size
        pos_centers = pos_bins + bin_size / 2.0

        #  Heatmap of all linear fields, sorted by peak
        fig_lin_heat, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(
            lin[:, sorted_cell_indices].T,
            aspect="auto", cmap="hot", origin="lower", interpolation="nearest",
        )
        ax.set_xlabel("Linear position bin", fontsize=12)
        ax.set_ylabel("Cell (sorted by peak)", fontsize=12)
        ax.set_title("Bidirectional Linear Place Fields",
                     fontsize=14, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Firing rate (Hz)")
        plt.tight_layout()

        # ---- Example cells: evenly spaced across sorted order ----
        if len(sorted_cell_indices) > n_cells_to_plot:
            sample_idx = np.linspace(
                0, len(sorted_cell_indices) - 1, n_cells_to_plot
            ).astype(int)
            example_indices = sorted_cell_indices[sample_idx]
        else:
            example_indices = sorted_cell_indices

        n_to_plot = len(example_indices)
        fig_lin_examples, axes = plt.subplots(
            n_to_plot, 1, figsize=(8, 1.8 * n_to_plot), sharex=True,
        )
        if n_to_plot == 1:
            axes = np.array([axes])

        for i in range(n_to_plot):
            k       = example_indices[i]
            cell_id = cell_ids[k]
            is_exc  = cell_id in excitatory_neurons

            ax = axes[i]
            ax.plot(pos_centers, lin[:, k], "k-", linewidth=2)
            ax.fill_between(
                pos_centers, lin[:, k],
                alpha=0.3, color="blue" if is_exc else "red",
            )
            ax.set_ylabel(
                f"Cell {cell_id}\n{'Exc' if is_exc else 'Inh'}", fontsize=9,
            )
            if i == 0:
                ax.set_title(
                    "Example linear fields (sampled across track)", fontsize=12,
                )
            if i == n_to_plot - 1:
                ax.set_xlabel("Position (track units)", fontsize=11)

        plt.tight_layout()
        linear_figs["linear_heatmap"]  = fig_lin_heat
        linear_figs["linear_examples"] = fig_lin_examples

    # Raster plot: one cell's spikes across trials (track passes)
    if len(sorted_cell_indices) > 0:
        raster_cell_idx = min(raster_cell_idx, len(sorted_cell_indices) - 1)
        cell_idx        = sorted_cell_indices[raster_cell_idx]
        cell_id         = cell_ids[cell_idx]

        spk_info    = spike_info[cell_id]
        spk_df      = spk_info.as_dataframe()
        spike_times = spk_info.index.values
        spike_x     = spk_df['x'].values
        spike_y     = spk_df['y'].values
        spike_vel   = spk_df['velocity'].values

        pos_df = position_data.as_dataframe()
        t      = position_data.index.values
        x_pos  = pos_df['x'].values
        y_pos  = pos_df['y'].values
        vel    = pos_df['velocity'].values

        # Track axis (matches find_PFs logic)
        vel_mask = vel >= velocity_cutoff
        x_run    = x_pos[vel_mask]
        y_run    = y_pos[vel_mask]
        x_span   = x_run.max() - x_run.min()
        y_span   = y_run.max() - y_run.min()

        if x_span >= y_span:
            pos_values_full = x_pos
            spike_positions = spike_x
        else:
            pos_values_full = y_pos
            spike_positions = spike_y

        spike_vel_mask           = spike_vel >= velocity_cutoff
        spike_times_filtered     = spike_times[spike_vel_mask]
        spike_positions_filtered = spike_positions[spike_vel_mask]

        pos_values = pos_values_full[vel_mask]
        t_filtered = t[vel_mask]

        pos_min   = pos_values.min()
        pos_max   = pos_values.max()
        pos_range = pos_max - pos_min

        pos_values_norm      = pos_values - pos_min
        spike_positions_norm = spike_positions_filtered - pos_min

        # Trial detection by smoothed direction changes
        from scipy.ndimage import uniform_filter1d
        pos_diff      = np.diff(pos_values_norm)
        smoothed_diff = uniform_filter1d(pos_diff, size=10, mode='nearest')
        direction     = np.sign(smoothed_diff)
        direction     = np.concatenate([[direction[0]], direction])
        dir_changes   = np.where(np.diff(direction) != 0)[0]
        trial_boundaries = np.concatenate(
            [[0], dir_changes + 1, [len(pos_values_norm)]]
        )
        n_trials = len(trial_boundaries) - 1

        fig_raster, axes_raster = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        ax_raster, ax_rate = axes_raster

        trial_count = 0
        for trial_idx in range(n_trials):
            start_idx = trial_boundaries[trial_idx]
            end_idx   = trial_boundaries[trial_idx + 1]

            if end_idx - start_idx < 10:
                continue

            trial_t   = t_filtered[start_idx:end_idx]
            trial_pos = pos_values_norm[start_idx:end_idx]

            # Skip trials that span a session-boundary gap
            if len(trial_t) > 1 and np.max(np.diff(trial_t)) > 5:
                continue

            # Skip trials that don't traverse enough of the track
            if trial_pos.max() - trial_pos.min() < pos_range * 0.3:
                continue

            trial_spike_mask = (
                (spike_times_filtered >= trial_t[0])
                & (spike_times_filtered <= trial_t[-1])
            )
            trial_spike_pos = spike_positions_norm[trial_spike_mask]

            trial_dir = 'out' if np.mean(np.diff(trial_pos)) > 0 else 'in'
            color     = 'green' if trial_dir == 'out' else 'blue'

            ax_raster.plot(
                trial_spike_pos,
                np.ones(len(trial_spike_pos)) * trial_count,
                '.', markersize=3, color=color,
            )
            trial_count += 1

        ax_raster.set_ylabel('Trial', fontsize=12)
        ax_raster.set_title(f'Neural Response for Cell {cell_id}',
                            fontsize=14, fontweight='bold')
        ax_raster.set_xlim([0, pos_range])
        ax_raster.invert_yaxis()
        ax_raster.grid(True, alpha=0.3)

        ax_rate.plot(pos_centers, lin[:, cell_idx], 'k-', linewidth=2)
        ax_rate.fill_between(pos_centers, lin[:, cell_idx],
                             alpha=0.3, color='blue')
        ax_rate.set_xlabel('Position (cm)', fontsize=12)
        ax_rate.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax_rate.set_title('Average Place Field',
                          fontsize=12, fontweight='bold')
        ax_rate.set_xlim([0, pos_range])
        ax_rate.grid(True, alpha=0.3)

        plt.tight_layout()
        linear_figs["raster_plot"] = fig_raster


    # 2D fields
    
    two_d_peak   = np.nanmax(Field_Data, axis=(0, 1))
    has_field_2d = two_d_peak > fr_cutoff

    if has_field_2d.any():
        if Field_Data_Linear is not None and len(sorted_cell_indices) > 0:
            example_cells_2d = np.array(
                [c for c in sorted_cell_indices if has_field_2d[c]]
            )
        else:
            example_cells_2d = np.where(has_field_2d)[0]

        # Cap at max_cells_2d, sampled evenly across the sort order
        if len(example_cells_2d) > max_cells_2d:
            sample_idx       = np.linspace(
                0, len(example_cells_2d) - 1, max_cells_2d
            ).astype(int)
            example_cells_2d = example_cells_2d[sample_idx]

        n_example = len(example_cells_2d)

        # Subplot dimensions matched to track aspect (clamped)
        x_extent      = x_edges[-1] - x_edges[0]
        y_extent      = y_edges[-1] - y_edges[0]
        ratio         = y_extent / x_extent
        ratio_clamped = max(0.3, min(ratio, 3.0))

        subplot_w = 2.5
        subplot_h = subplot_w * ratio_clamped

        n_cols = max(1, min(int(ceil(sqrt(n_example))),
                            int(round(15 / subplot_w))))
        n_rows = int(ceil(n_example / n_cols))

        two_d_fig, axes_2d = plt.subplots(
            n_rows, n_cols,
            figsize=(subplot_w * n_cols, subplot_h * n_rows + 0.5),
            squeeze=False,
        )

        im = None
        for idx, k in enumerate(example_cells_2d):
            r, c = idx // n_cols, idx % n_cols
            ax   = axes_2d[r, c]

            fr_map = Field_Data[:, :, k]   # (ny, nx), no transpose/flip
            im = ax.imshow(
                fr_map,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                aspect="auto",
                origin="lower",
                cmap="hot",
                interpolation="nearest",
            )
            cell_id_plot = cell_ids[k]
            is_exc       = cell_id_plot in excitatory_neurons
            ax.set_title(
                f"Cell {cell_id_plot} ({'Exc' if is_exc else 'Inh'})",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused axes in the grid
        for idx in range(n_example, n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes_2d[r, c].axis("off")

        plt.tight_layout()
        if im is not None:
            cbar = two_d_fig.colorbar(
                im, ax=axes_2d.ravel().tolist(), shrink=0.5,
            )
            cbar.set_label("Firing rate (Hz)")

    return {
        "linear_figs": linear_figs,
        "fig_2d":      two_d_fig,
    }
        
        
def decoding_error_plots(results,time_bin_size):
    
    position_error = results['position_error']          # shape (n, 2): [time, error]
    cumulative_error = results['cumulative_error']      # shape (n, 2): [sorted_error, cum_prob]
    true_occ = results['cumulative_position_occupancy'] # (ny, nx)
    cum_posterior = results['cumulative_posterior']     # (ny, nx)

    x_cdf = cumulative_error[:, 0]   # sorted errors
    y_cdf = cumulative_error[:, 1]   # cumulative probability

    mean_error = position_error[:, 1].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(cum_posterior, origin='lower',cmap='hot', aspect='auto')
    plt.colorbar(im0, ax=axes[0], label='Posterior')
    axes[0].set_title('Cumulative decoded posterior')

    axes[0].set_xlabel('x bins (cm)')
    axes[0].set_ylabel('y bins (cm)')

    im1 = axes[1].imshow(true_occ, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im1, ax=axes[1], label='Time (s)')
    axes[1].set_title('True position occupancy')
    axes[1].set_xlabel('x bins (cm)')
    axes[1].set_ylabel('y bins (cm)')
    
    axes[2].plot(cumulative_error[:, 0],cumulative_error[:, 1],'k',linewidth=2)
    axes[2].set_xlim(0, np.ceil(position_error[:, 1].max()))
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel('Decoding error (cm)')
    axes[2].set_ylabel('Cumulative probability')
    axes[2].set_title(f'Cumulative decoding error\n(mean = {mean_error:.1f} cm)')

    plt.tight_layout()
    plt.show()
    
    return fig