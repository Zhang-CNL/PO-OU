import pynapple as nap
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

import os
from datetime import datetime

import torch
from torch.distributions import MultivariateNormal
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import hippocampalseq.utils as hseu

def gaussians_to_grids(
        means: list[torch.Tensor], 
        covs: list[torch.Tensor], 
        environment_size: list[tuple[int, ...]], 
        bin_size_cm: int, 
        marginal_position_slice: slice
    ):
    """Turn per-window Gaussian posteriors into (T, Nx, Ny) grids like emission_probabilities."""
    sz = tuple(int((es[1] - es[0])/bin_size_cm) for es in environment_size)
    if len(sz) == 1:
        sz += (1,)
    Z = hseu.make_ndgrid(environment_size, bin_size_cm, indexing='ij')
    Z = hseu.ensure_numpy(Z)
    normalized_posteriors = []
    for m, c in zip(means, covs):
        m = hseu.ensure_numpy(m[:, marginal_position_slice])
        c = hseu.ensure_numpy(c[:, marginal_position_slice, marginal_position_slice])
        c = np.atleast_2d(c)
        segment_grid = np.zeros((len(m),)+sz, dtype=float)
        for t in range(len(m)):
            mvn = multivariate_normal(mean=m[t].ravel(), cov=c[t])
            lp = mvn.logpdf(Z)
            segment_grid[t] = np.exp(lp - logsumexp(lp, 0)).reshape(sz)
        normalized_posteriors.append(segment_grid)
    return normalized_posteriors

def bayesian_map_to_grids(
        posteriors: list[torch.Tensor],
        *,
        environment_size: list[tuple[int, ...]]|None=None,
        bin_size_cm: int|None = None
    ) -> list[np.ndarray]:
    """Convert posteriors from Bayesian MAP to normalized grids. 
    Checks for correct shape, non-negativity, and finiteness. 
    Args:
        posteriors (list[torch.Tensor]): List of posterior tensors (T, Nx, Ny).
        environment_size (list[tuple[int, ...]]|None): Optional list of environment sizes for shape validation.
        bin_size_cm (int|None): Optional bin size in cm for shape validation.

    Returns:
        list[np.ndarray]: List of normalized posterior grids (T, Nx, Ny).
    """
    normalized_grids = []
    for i,posterior in enumerate(posteriors):
        grid = hseu.ensure_numpy(posterior)

        assert grid.ndim == 3, f"Posterior {i} must be 3D (T, Nx, Ny), got {grid.ndim}D"
        if environment_size is not None and bin_size_cm is not None:
            expected_shape = tuple(int(np.ceil(s / bin_size_cm)) for s in environment_size)
            assert grid.shape[1:3] == expected_shape, (
                f"Posterior {i} shape {grid.shape[1:3]} does not match expected "
                f"shape {expected_shape} for environment size {environment_size} "
                f"and bin size {bin_size_cm}"
            )
        assert np.all(np.isfinite(grid)), f"Posterior {i} contains non-finite values"
        assert np.all(grid >= 0), f"Posterior {i} contains negative values"
        
        rowsum = grid.sum(axis=(1, 2), keepdims=True)
        normalized_posterior = np.divide(grid, rowsum, out=np.zeros_like(grid), where=(rowsum > 0))
        normalized_grids.append(normalized_posterior)

    return normalized_grids

def _phase_in_count_window(
        phase: np.ndarray,
        count_window: tuple[float, float],
    ) -> np.ndarray:
    """Return a mask for phases inside a possibly wrapping phase window."""
    lo, hi = count_window
    if lo < hi:
        return (phase > lo) & (phase <= hi)
    return (phase > lo) | (phase <= hi)

def _passes_cycle_flags(
        indices: np.ndarray,
        velocity: np.ndarray,
        direction_per_sample: np.ndarray,
        timestamps: np.ndarray,
        phase: np.ndarray,
        velocity_cutoff: float,
        velocity_range: tuple[float,float]|None,
        direction: float|None,
        theta_length_s: tuple[float,float],
        theta_start: float,
        step_s: float,
    ) -> bool:
    """Check velocity, direction, duration, and phase progression for one cycle."""
    if len(indices) < 2 or np.any(velocity[indices] < velocity_cutoff):
        return False

    if velocity_range is not None:
        median_velocity = np.median(velocity[indices])
        if not (velocity_range[0] <= median_velocity < velocity_range[1]):
            return False

    if direction is not None and direction_per_sample[indices[0]] != direction:
        return False

    duration = timestamps[indices[-1]] - timestamps[indices[0]] + step_s
    if not (theta_length_s[0] <= duration <= theta_length_s[1]):
        return False

    unwrapped_phase = phase[indices].copy()
    unwrapped_phase[unwrapped_phase <= theta_start] += 360.0
    return not np.any(np.diff(unwrapped_phase) < 0)

def select_oscillations(
        osc: np.ndarray,
        velocity: np.ndarray,
        direction_per_sample: np.ndarray,
        timestamps: np.ndarray,
        phase: np.ndarray,
        spike_count: np.ndarray,
        *,
        velocity_cutoff: float = 10.0,
        velocity_range: tuple[float,float]|None = None,
        direction: float|None = None,
        theta_length_s: tuple[float,float] = (0.08, 0.16),
        theta_start: float = 70.0,
        step_s: float = 5e-3,
        count_window: tuple[float,float] = (80, 190),
        spike_count_percentile: float|None = 66.67,
        verbose: bool = False,
    ) -> tuple[list[int], float]:
    """
    Select cycles that pass the cycle checks and spike-count percentile.

    Returns the selected cycle IDs and spike-count threshold.
    """
    in_window = _phase_in_count_window(phase, count_window)
    cycle_spike_counts = {}

    for cycle_id_value in np.unique(osc):
        cycle_id = int(cycle_id_value)
        indices = np.flatnonzero(osc == cycle_id)

        passes_flags = _passes_cycle_flags(
            indices,
            velocity=velocity,
            direction_per_sample=direction_per_sample,
            timestamps=timestamps,
            phase=phase,
            velocity_cutoff=velocity_cutoff,
            velocity_range=velocity_range,
            direction=direction,
            theta_length_s=theta_length_s,
            theta_start=theta_start,
            step_s=step_s,
        )
        if not passes_flags:
            continue

        window_spike_count = spike_count[indices][in_window[indices]].sum()
        cycle_spike_counts[cycle_id] = window_spike_count

    counts = np.array(list(cycle_spike_counts.values()))
    if spike_count_percentile is not None:
        threshold = (
            np.percentile(counts, spike_count_percentile)
            if counts.size
            else float("inf")
        )
        selected = [
            cycle_id
            for cycle_id, count in cycle_spike_counts.items()
            if count >= threshold
        ]
    else:
        threshold = 0.0
        selected = list(cycle_spike_counts.keys())

    if verbose:
        print(
            f"{len(cycle_spike_counts)} oscillations pass flags; "
            f"{len(selected)} selected "
            f"(>= {threshold:.0f} spikes in {count_window})"
        )

    return selected

def peak_posterior(
    decoded_grids: list[np.ndarray],
    environment_size: list[tuple[int,...]],
    segments: list[nap.TsdFrame],
    spikemats: list[np.ndarray],
    *,
    bin_size_cm: int = 2,
    phase_bin: int = 10,
    half_width: int = 30,
    minimum_max: float = 0.01,
    velocity_cutoff: float = 10.0,
    spike_count_percentile: float = 66.67,
    count_window: tuple[float,float] = (80, 190),
    theta_start: float = 70.0,
    theta_length_s: tuple[float,float] = (0.08, 0.16),
    step_s: float = 5e-3,
    direction: float|None = None,
    velocity_range: tuple[float,float]|None = None,
    verbose: bool = False,
):
    if not (len(decoded_grids) == len(segments) == len(spikemats)):
        raise ValueError(
            "decoded_grids, segments, and spikemats must have the same "
            "number of segments"
        )
    """Build a per-position peak-phase histogram for selected theta cycles."""
    grid = hseu.ensure_numpy(hseu.make_ndgrid(environment_size, bin_size_cm, indexing="ij"))[:, 0]

    # The declared axis can be wrong when the track runs along y. Choose the
    # position column whose span best matches the decoded grid's track length.
    axis_col = min(
        (column for column in ("x", "y") if column in segments[0].columns),
        key=lambda column: abs(np.ptp(
                np.concatenate([segment[column].values for segment in segments])
            )
            - np.ptp(grid)
        ),
    )

    n_pos      = 2 * half_width + 1
    dist_edges = (np.arange(n_pos + 1) - half_width - 0.5) * bin_size_cm
    n_ph       = int(np.ceil(360 / phase_bin))

    rel         = []
    phase_all   = []
    osc_all     = []
    vel_all     = []
    spk_all     = []
    dir_all     = []
    t_all_parts = []
    osc_offset  = 0

    for segment_index, (ep, frame, spk) in enumerate(zip(decoded_grids, segments, spikemats)):
        ep_array = hseu.ensure_numpy(ep).reshape(len(ep), -1)
        if len(ep_array) != len(frame) or len(ep_array) != len(spk):
            raise ValueError(
                f"Segment {segment_index} has mismatched samples: "
                f"decoded={len(ep_array)}, frame={len(frame)}, spikes={len(spk)}"
            )

        pos                = frame[axis_col].values
        run_direction      = np.sign(pos[-1] - pos[0]) or 1.0
        relative_posterior = np.zeros((n_pos, len(ep_array)))
        
        assert ep_array.shape[1] == len(grid), f"Grid and ep shapes not equal {ep_array.shape} {len(grid)}"
        for time_index in range(len(ep_array)):
            distance_bins = np.digitize(
                (grid - pos[time_index]) * run_direction,
                dist_edges,
            ) - 1
            valid_bins = ((distance_bins >= 0) & (distance_bins < n_pos))
            np.add.at(
                relative_posterior[:, time_index],
                distance_bins[valid_bins],
                ep_array[time_index][valid_bins],
            )
            total_mass = ep_array[time_index].sum()
            kept_mass  = ep_array[time_index][valid_bins].sum()
            if total_mass > 0 and verbose:
                dropped_fraction = 1 - kept_mass / total_mass
                if dropped_fraction > .01:
                    print(
                        f"time={time_index}: dropped {dropped_fraction:.1%} "
                        f"of posterior mass"
                    )

        rel.append(relative_posterior)

        phase_values = np.mod(frame["Phase Deg"].values, 360.0)
        phase_values[phase_values == 0] = 360.0
        phase_all.append(phase_values)

        oscillation_values = frame["Oscillation Number"].values.astype(int) + osc_offset
        osc_all.append(oscillation_values)
        osc_offset = int(oscillation_values.max()) + 1

        vel_all.append(frame["Velocity"].values)
        spk_all.append(np.asarray(spk).sum(axis=1))
        dir_all.append(np.full(len(ep_array), run_direction))
        t_all_parts.append(frame.index.values)

    R     = np.concatenate(rel, axis=1) # Rat-relative position
    phase = np.concatenate(phase_all)
    osc   = np.concatenate(osc_all)
    vel   = np.concatenate(vel_all)
    spk   = np.concatenate(spk_all)
    dirs  = np.concatenate(dir_all)
    t_all = np.concatenate(t_all_parts)

    selected = select_oscillations(
        osc,
        vel,
        dirs,
        t_all,
        phase,
        spk,
        velocity_cutoff=velocity_cutoff,
        velocity_range=velocity_range,
        direction=direction,
        theta_length_s=theta_length_s,
        theta_start=theta_start,
        step_s=step_s,
        count_window=count_window,
        spike_count_percentile=spike_count_percentile,
        verbose=verbose,
    )

    dist_of_peak = np.zeros((n_pos, n_ph))
    for oscillation_id in selected:
        mask = (osc == oscillation_id) & (spk > 0)
        if mask.sum() < 2:
            continue

        current_posterior = R[:, mask]
        current_phase     = phase[mask]
        row_maxima        = current_posterior.max(axis=1)

        for position_index in range(n_pos):
            if (row_maxima[position_index] >= minimum_max 
                and not np.all(current_posterior[position_index] == row_maxima[position_index])
            ):
                peak_phases = current_phase[
                    current_posterior[position_index] == row_maxima[position_index]
                ]
                for peak_phase in peak_phases:
                    phase_index = int(np.ceil(peak_phase / phase_bin)) - 1
                    phase_index = int(np.clip(phase_index, 0, n_ph - 1))
                    dist_of_peak[position_index, phase_index] += 1

    column_totals = dist_of_peak.sum(axis=0, keepdims=True)
    column_totals[column_totals == 0] = 1

    return {
        "dist_of_peak_post"     : dist_of_peak,
        "norm_dist"             : dist_of_peak / column_totals,
        "n_oscillations_used"   : len(selected),
    }

def peak_posterior_2d(
        decoded_grids: list[np.ndarray],
        environment_size: list[tuple[int,...]],
        segments: list[nap.TsdFrame],
        spikemats: list[np.ndarray],
        *,
        bin_size_cm: int = 2,
        phase_bin: int = 10,
        half_width: int = 30,
        minimum_max: float = 0.01,
        velocity_cutoff: float = 10.0,
        spike_count_percentile: float = 66.67,
        count_window: tuple[float,float] = (80, 190),
        theta_start: float = 70.0,
        theta_length_s: tuple[float,float] = (0.08, 0.16),
        step_s: float = 5e-3,
        direction: float|None = None,
        velocity_range: tuple[float,float]|None = None,
        verbose: bool = False,
    ):
    if not (len(decoded_grids) == len(segments) == len(spikemats)):
        raise ValueError(
            "decoded_grids, segments, and spikemats must have the same length"
        )
    if not environment_size or len(environment_size[0]) != 2:
        raise ValueError("environment_size must describe a 2D grid")
    if phase_bin <= 0 or bin_size_cm <= 0 or half_width < 0:
        raise ValueError("phase_bin and bin_size_cm must be positive; half_width nonnegative")

    grid = hseu.ensure_numpy(
        hseu.make_ndgrid(environment_size, bin_size_cm, indexing="ij")
    )
    grid_x = grid[:, 0]
    grid_y = grid[:, 1]

    n_side = 2 * half_width + 1
    n_ph = int(np.ceil(360 / phase_bin))
    relative_edges = (np.arange(n_side + 1, dtype=float) - half_width - 0.5) * bin_size_cm

    rel         = []
    phase_all   = []
    osc_all     = []
    vel_all     = []
    spk_all     = []
    dir_all     = []
    t_all_parts = []
    osc_offset  = 0
    for ep, frame, spk in zip(decoded_grids, segments, spikemats):
        ep_array = hseu.ensure_numpy(ep).reshape(len(ep), -1)

        if ep_array.shape[1] != len(grid):
            raise ValueError(
                f"Decoded grid has {ep_array.shape[1]} spatial bins, "
                f"but environment grid has {len(grid)}"
            )
        if len(frame) != len(ep_array):
            raise ValueError("Each decoded grid must have one row per segment sample")
        if not {"x", "y"}.issubset(frame.columns):
            raise ValueError("Each segment must contain 'x' and 'y' position columns")

        pos_x = np.asarray(frame["x"].values, dtype=float)
        pos_y = np.asarray(frame["y"].values, dtype=float)
        if len(pos_x) == 0:
            continue

        # Keep the original direction filter semantics: for 2D, direction is
        # based on the dominant displacement axis of the segment.
        delta_x = pos_x[-1] - pos_x[0]
        delta_y = pos_y[-1] - pos_y[0]
        run_direction = np.sign(delta_x if abs(delta_x) >= abs(delta_y) else delta_y)
        if run_direction == 0:
            run_direction = 1.0

        relative_posterior = np.zeros((n_side, n_side, len(ep_array)))

        for time_index in range(len(ep_array)):
            relative_x = grid_x - pos_x[time_index]
            relative_y = grid_y - pos_y[time_index]

            x_bin = np.digitize(relative_x, relative_edges) - 1
            y_bin = np.digitize(relative_y, relative_edges) - 1
            valid = (
                (x_bin >= 0) & (x_bin < n_side)
                & (y_bin >= 0) & (y_bin < n_side)
            )

            np.add.at(
                relative_posterior[:, :, time_index],
                (y_bin[valid], x_bin[valid]),
                ep_array[time_index, valid],
            )

        rel.append(relative_posterior)

        phase_values = np.mod(
            np.asarray(frame["Phase Deg"].values, dtype=float), 360.0
        )
        phase_values[phase_values == 0] = 360.0
        phase_all.append(phase_values)

        oscillation_values = (
            np.asarray(frame["Oscillation Number"].values).astype(int) + osc_offset
        )
        osc_all.append(oscillation_values)
        osc_offset = oscillation_values.max() + 1

        vel_all.append(np.asarray(frame["Velocity"].values, dtype=float))
        spk_all.append(np.asarray(spk).sum(axis=1))
        dir_all.append(np.full(len(ep_array), run_direction))
        t_all_parts.append(np.asarray(frame.index.values))

    if not rel:
        raise ValueError("No non-empty segments were provided")

    R     = np.concatenate(rel, axis=2)
    phase = np.concatenate(phase_all)
    osc   = np.concatenate(osc_all)
    vel   = np.concatenate(vel_all)
    spk   = np.concatenate(spk_all)
    dirs  = np.concatenate(dir_all)
    t_all = np.concatenate(t_all_parts)

    selected = select_oscillations(
        osc,
        vel,
        dirs,
        t_all,
        phase,
        spk,
        velocity_cutoff=velocity_cutoff,
        velocity_range=velocity_range,
        direction=direction,
        theta_length_s=theta_length_s,
        theta_start=theta_start,
        step_s=step_s,
        count_window=count_window,
        spike_count_percentile=spike_count_percentile,
        verbose=verbose,
    )

    dist_of_peak = np.zeros((n_side, n_side, n_ph), dtype=float)
    for oscillation_id in selected:
        mask = (osc == oscillation_id) & (spk > 0)
        if mask.sum() < 2:
            continue

        current_posterior = R[:, :, mask]
        current_phase = phase[mask]
        maxima = current_posterior.max(axis=2)

        for y_index in range(n_side):
            for x_index in range(n_side):
                bin_values = current_posterior[y_index, x_index, :]
                bin_maximum = maxima[y_index, x_index]

                if (bin_maximum >= minimum_max and not np.all(bin_values == bin_maximum)):
                    peak_phases = current_phase[bin_values == bin_maximum]
                    for peak_phase in peak_phases:
                        phase_index = int(np.ceil(peak_phase / phase_bin)) - 1
                        phase_index = int(np.clip(phase_index, 0, n_ph - 1))
                        dist_of_peak[y_index, x_index, phase_index] += 1

    phase_totals = dist_of_peak.sum(axis=(0, 1), keepdims=True)
    phase_totals[phase_totals == 0] = 1

    return {
        "dist_of_peak_post": dist_of_peak,
        "norm_dist": dist_of_peak / phase_totals,
        "n_oscillations_used": len(selected),
        "relative_x_centers_cm": (
            np.arange(n_side, dtype=float) - half_width
        ) * bin_size_cm,
        "relative_y_centers_cm": (
            np.arange(n_side, dtype=float) - half_width
        ) * bin_size_cm,
        "phase_bin_centers_deg": (
            np.arange(n_ph, dtype=float) + 0.5
        ) * phase_bin,
    }

def save_all_figures(figures_dict, save_dir, filename_tag):
    """Save all figures to organized PDFs."""
    date_folder = datetime.now().strftime('%Y-%m-%d')
    top_folder = os.path.join(save_dir, date_folder, filename_tag)
    os.makedirs(top_folder, exist_ok=True)

    for category, figs in figures_dict.items():
        pdf_path = os.path.join(top_folder, f"{category}.pdf")

        with PdfPages(pdf_path) as pdf:
            if isinstance(figs, dict):
                for name, fig in figs.items():
                    if isinstance(fig, list):
                        for f in fig:
                            pdf.savefig(f, bbox_inches='tight')
                            plt.close(f)
                    else:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            elif isinstance(figs, list):
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            else:
                pdf.savefig(figs, bbox_inches='tight')
                plt.close(figs)
        print(f"Saved {category} to {pdf_path}")

    return top_folder

def max_probability(decoded_grids: list[np.ndarray], axis: int = 0):
    if not isinstance(decoded_grids, list):
        decoded_grids = [decoded_grids]
    max_probs = [
        g.reshape(len(g),-1).max(axis=axis) for g in decoded_grids
    ]
    return max_probs
            

def _two_troughs(trace: np.ndarray, phase_bin: int, smooth_sigma: float|None=1.0):
    n = len(trace)
    x = np.concatenate([trace, trace, trace])
    if smooth_sigma:
        x = gaussian_filter1d(x, smooth_sigma)
    idx, _ = find_peaks(-x)
    idx = np.unique(idx[(idx >= n) & (idx < 2 * n)] - n)
    j0 = idx[np.argmin(x[idx + n])]  # deepest local minimum
    if len(idx) == 1:
        j1 = (j0 + n // 2) % n
    else:
        other = idx[idx != j0]
        sep = np.abs(((other - j0 + n // 2) % n) - n // 2) * phase_bin
        j1 = other[np.argmin(np.abs(sep - 180.0))]   # the one nearest opposite
    return np.sort([j0, j1]) * phase_bin + phase_bin / 2


def identify_theta_sequence_windows(bimodal_firing_rate: np.ndarray, decoding_distribution: np.ndarray, phase_bin: int = 10):
    fr_t  = _two_troughs(np.asarray(bimodal_firing_rate), phase_bin, smooth_sigma=1.0)
    dec_t = _two_troughs(np.asarray(decoding_distribution).max(axis=0), phase_bin, smooth_sigma=1.0)
    w = lambda a, b: [a % 360, b % 360]
    return {
        'major_peak_window': w(fr_t[1] + phase_bin,  fr_t[0] - phase_bin),
        'minor_peak_window': w(fr_t[0],              fr_t[1]),
        'forward_window':    w(dec_t[1] + phase_bin, dec_t[0] - phase_bin),
        'reverse_window':    w(dec_t[0] + phase_bin, dec_t[1] - phase_bin),
        'firing_rate_troughs_deg': fr_t.tolist(),
        'decoding_troughs_deg':    dec_t.tolist(),
    }