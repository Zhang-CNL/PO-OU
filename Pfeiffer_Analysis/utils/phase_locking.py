
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from scipy.signal import filtfilt


def calculate_phase_locked(spike_info_with_phase,
                            total_duration,
                            speed_cutoff=10,
                            phase_bin=10,
                            gaussian_sigma=12,
                            theta_length_min_max=(0.08, 0.16),
                            minimum_spike_count=100,
                            limit_analysis_by_theta_length=True):

    n_phase_bins  = int(360 / phase_bin)
    phase_edges   = np.arange(0, 361, phase_bin)
    phase_centers = phase_edges[:-1] + phase_bin / 2

    # Gaussian filter: 7 bins, sigma = sigma_deg / phase_bin (in bin units)
    kernel_size  = 7
    kernel_sigma = gaussian_sigma / phase_bin
    g = gaussian(kernel_size, kernel_sigma)
    g = g / g.sum()

    firing_rate_per_phase = {}

    for cell_id, rec in spike_info_with_phase.items():
        cols = list(rec.columns)

        # Pull spike-side columns (positional access for safety vs 1-row bug)
        phases    = rec.values[:, cols.index('theta_phase')].astype(float)
        velocity  = rec.values[:, cols.index('velocity')].astype(float)
        cyc_dur   = rec.values[:, cols.index('cycle_duration')].astype(float)
        monotonic = rec.values[:, cols.index('monotonic_increasing')].astype(float)

        if len(phases) == 0:
            continue

        # Phase == 0 to 360 so it lands in the last bin (matches MATLAB)
        phases = phases.copy()
        phases[phases == 0] = 360

        # checks min spike count after velocity filter 
        vel_mask = velocity >= speed_cutoff
        if vel_mask.sum() < minimum_spike_count:
            continue

        #  Theta length filter applied (or not) based on flag 
        if limit_analysis_by_theta_length:
            theta_valid = ((cyc_dur >= theta_length_min_max[0])
                        & (cyc_dur <= theta_length_min_max[1])
                        & (monotonic == 1))
            keep = vel_mask & theta_valid
        else:
            keep = vel_mask

        phases_kept = phases[keep]
        if len(phases_kept) == 0:
            continue

        # Bin into phase bins (matches MATLAB convention: > lower, <= upper)
        # np.histogram uses [lower, upper) by default — slight difference but
        # negligible since phase is continuous
        spike_counts, _ = np.histogram(phases_kept, bins=phase_edges)
        raw_rate = spike_counts / max(total_duration, 1e-9) * n_phase_bins

        # Wrap-smooth-unwrap, exactly like MATLAB
        double_rate = np.concatenate([raw_rate, raw_rate])
        padlen = min(3 * len(g), len(double_rate) - 1)
        sm = filtfilt(g, [1.0], double_rate, padlen=padlen)

        half_bins   = n_phase_bins // 2
        smooth_rate = np.concatenate([
            sm[n_phase_bins : n_phase_bins + half_bins],
            sm[half_bins : n_phase_bins],
        ])

        double_counts = np.concatenate([spike_counts, spike_counts]).astype(float)
        smc           = filtfilt(g, [1.0], double_counts, padlen=padlen)
        smooth_counts = np.concatenate([
            smc[n_phase_bins : n_phase_bins + half_bins],
            smc[half_bins : n_phase_bins],
        ])

        # FRI = (smoothed - min) / max
        if smooth_rate.max() > 0:
            fri = (smooth_rate - smooth_rate.min()) / smooth_rate.max()
        else:
            fri = np.zeros_like(smooth_rate)
        
            

        firing_rate_per_phase[cell_id] = {
            'raw_rate':        raw_rate,
            'smooth_rate':     smooth_rate,
            'fri':             fri,
            'raw_counts':      spike_counts,
            'smooth_counts':   smooth_counts,
            'n_spikes':        int(keep.sum()),
            'n_spikes_vel':    int(vel_mask.sum()),
            'filtered_phases': phases_kept,
        }

    return firing_rate_per_phase, phase_centers

