import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_figure_1b_linear_track(pooled_dec_dist, forward_window, reverse_window,
                                phase_bin=10, position_bin=2, save_dir=None):

    n_pos, n_phase = pooled_dec_dist.shape
    n_phase_bin_360 = int(round(360 / phase_bin))

    # Double horizontally so wrap-around forward windows display contiguously
    doubled = np.concatenate([pooled_dec_dist, pooled_dec_dist], axis=1)
    n_phase_doubled = doubled.shape[1]

    fwd_start_col = int(round(forward_window[0] / phase_bin))
    # Forward typically wraps (e.g. [200, 60]); end column extends by 360°
    if forward_window[1] < forward_window[0]:
        fwd_end_col = int(round(forward_window[1] / phase_bin)) + n_phase_bin_360
    else:
        fwd_end_col = int(round(forward_window[1] / phase_bin))

    rev_start_col = int(round(reverse_window[0] / phase_bin))
    if reverse_window[1] < reverse_window[0]:
        rev_end_col = int(round(reverse_window[1] / phase_bin)) + n_phase_bin_360
    else:
        rev_end_col = int(round(reverse_window[1] / phase_bin))

    fwd_data    = doubled[:, fwd_start_col:fwd_end_col]
    rev_data    = doubled[:, rev_start_col:rev_end_col]
    fwd_argmax  = fwd_data.argmax(axis=0)      # row of peak per column
    rev_argmax  = rev_data.argmax(axis=0)

    fwd_x_cols  = np.arange(fwd_start_col, fwd_end_col)
    rev_x_cols  = np.arange(rev_start_col, rev_end_col)
    fwd_slope_bins, fwd_intercept_bins = np.polyfit(fwd_x_cols, fwd_argmax, 1)
    rev_slope_bins, rev_intercept_bins = np.polyfit(rev_x_cols, rev_argmax, 1)
    fwd_fit_line = fwd_slope_bins * fwd_x_cols + fwd_intercept_bins
    rev_fit_line = rev_slope_bins * rev_x_cols + rev_intercept_bins

    # Convert slopes to cm per degree of phase
    fwd_slope_cm_per_deg = fwd_slope_bins * position_bin / phase_bin
    rev_slope_cm_per_deg = rev_slope_bins * position_bin / phase_bin

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                            gridspec_kw={'height_ratios': [3, 1]},
                            sharex=True)

    # Top: heatmap 
    mid_row = n_pos // 2
    pos_extent = [-mid_row * position_bin, (n_pos - 1 - mid_row) * position_bin]
    phase_extent = [0, n_phase_doubled * phase_bin]

    im = axes[0].imshow(
        doubled,
        aspect = 'auto',
        origin = 'lower',
        cmap   = 'hot',
        extent = [phase_extent[0], phase_extent[1],
                    pos_extent[0],   pos_extent[1]],
        vmin   = 0,
        vmax   = doubled.max(),
        interpolation="nearest"
    )

    # Best-fit lines in degrees x cm
    fwd_x_deg     = fwd_x_cols * phase_bin + phase_bin / 2
    rev_x_deg     = rev_x_cols * phase_bin + phase_bin / 2
    fwd_fit_cm    = (fwd_fit_line - mid_row) * position_bin
    rev_fit_cm    = (rev_fit_line - mid_row) * position_bin
    axes[0].plot(fwd_x_deg, fwd_fit_cm, 'c--', linewidth=2.5,
                label=f'Forward sweep: {fwd_slope_cm_per_deg:+.3f} cm/°')
    axes[0].plot(rev_x_deg, rev_fit_cm, 'c--', linewidth=2.5,
                label=f'Reverse sweep: {rev_slope_cm_per_deg:+.3f} cm/°')

    # Window boundaries (start lines, doubled)
    for deg in [forward_window[0], reverse_window[0],
                forward_window[0] + 360, reverse_window[0] + 360]:
        axes[0].axvline(deg, color='white', linestyle='--', linewidth=1, alpha=0.7)

    axes[0].set_ylabel('Decoded position\nrelative to rat (cm)', fontsize=11)
    axes[0].set_xlim(0, n_phase_doubled * phase_bin)
    axes[0].set_ylim(pos_extent[0], pos_extent[1])
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.85)
    axes[0].set_title('Linear Track: pooled peak posterior distribution\n'
                    'Cyan = best-fit sweep direction; White = window boundaries',
                    fontsize=11)
    plt.colorbar(im, ax=axes[0], label='Normalized probability', pad=0.02)

    # Bottom: smoothed max-probability trace
    max_prob_smooth = gaussian_filter1d(doubled.max(axis=0), sigma=1.0)
    phase_axis_deg  = np.arange(n_phase_doubled) * phase_bin + phase_bin / 2
    axes[1].plot(phase_axis_deg, max_prob_smooth, 'ko-',
                linewidth=2, markersize=4, markerfacecolor='black')
    for deg in [forward_window[0], reverse_window[0],
                forward_window[0] + 360, reverse_window[0] + 360]:
        axes[1].axvline(deg, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Theta phase (°)', fontsize=11)
    axes[1].set_ylabel('Max\nposterior', fontsize=11)
    axes[1].set_xlim(0, n_phase_doubled * phase_bin)
    axes[1].set_xticks(np.arange(0, 721, 90))

    plt.tight_layout()

    if save_dir is not None:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / 'figure_1b_linear_track.png'
        plt.savefig(path, dpi=600, bbox_inches='tight')
        print(f"Saved to {path}")

    plt.show()

    return {
        'doubled_distribution':  doubled,
        'forward_fit_argmax':    fwd_argmax,
        'reverse_fit_argmax':    rev_argmax,
        'forward_slope_cm_per_deg': fwd_slope_cm_per_deg,
        'reverse_slope_cm_per_deg': rev_slope_cm_per_deg,
        'forward_window':        forward_window,
        'reverse_window':        reverse_window,
    }