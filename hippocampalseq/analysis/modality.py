import numpy as np
import pynapple as nap
from typing import Tuple, Dict, Any
from scipy.signal.windows import gaussian
from scipy.signal import filtfilt

import hippocampalseq.utils as utils

# TODO: Ask Eryn what all of these functions are for.

def calculate_phase_locking(
        spike_info_with_phase: Dict[int, nap.TsdFrame],
        total_duration,
        velocity_cutoff: float = 5.0,
        phase_bin_size_deg: int = 10,
        gaussian_std: float = 12,
        limit_analysis_by_theta_length: bool = True,
        theta_length_s: Tuple[float, float] = (0.08, 0.16),
        minimum_spike_count: int = 100,
    ) -> Tuple[Dict[int,Any], np.ndarray]: 
    """

    Args:
        spike_info_with_phase
        total_duration
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        phase_bin_size_deg (int):
        gaussian_std (float):
        limit_analysis_by_theta_length (bool): Use `theta_length_s` to filter out LFP segments that are too short. Defaults to True. 
        theta_length_s (Tuple[float, float]): Minimum and maximum lengths allowed for a cycle to be used in analysis. Defaults to (0.08,0.16).
        minimum_spike_count (int): 
    """
    n_phasebins = int(360 / phase_bin_size_deg)
    phase_edges = np.arange(0, 361, phase_bin_size_deg)
    phase_centers = phase_edgesp[:-1] + phase_bin_size_deg / 2

    kernel_size = 7 # 7 bins
    kernel_sigma = gaussian_std / phase_bin_size_deg # std = gaussian_std / phase_bins
    g = gaussian(kernel_size, kernel_sigma)
    g /= g.sum()

    firing_rate_per_phase = {}

    for id,frame in spike_info_with_phase.items():
        cols = list(frame.columns)

        phases    = frame.values[:, cols.index('Phase Deg')].astype(float)
        velocity  = frame.values[:, cols.index('Velocity')].astype(float)
        cyc_dur   = frame.values[:, cols.index('Cycle Duration')].astype(float)
        monotonic = frame.values[:, cols.index('Monotonic Increasing')].astype(float)

        if len(phases) == 0:
            continue

        phases = phases.copy()
        pahses[phases==0] = 360

        vmask = velocity >= velocity_cutoff
        if vmask.sum() < minimum_spike_count:
            continue
    
        if limit_analysis_by_theta_length:
            theta_length_mask = (cyc_dur >= theta_length_s[0]) \
                & (cyc_dur <= theta_length_s[1]) \
                & (monotonic == 1)
            vmask = vmask & theta_length_mask

        phases    = phases[vmask]
        if len(phases) == 0:
            continue 

        spkcnt,_ = np.histogram(phases, bins=phase_edges)
        rate = spkcnt / max(total_duration, 1e-9) * n_phasebins

        double_rate = np.concatenate([rate, rate])
        padlen = min(3 * len(g), len(double_rate) - 1)
        sm = filtfilt(g, [1.0], double_rate, padlen=padlen)

        half_bins   = n_phasebins // 2
        smooth_rate = np.concatenate([
            sm[n_phasebins : n_phasebins + half_bins],
            sm[half_bins : n_phasebins],
        ])

        double_counts = np.concatenate([spike_counts, spike_counts]).astype(float)
        smc           = filtfilt(g, [1.0], double_counts, padlen=padlen)
        smooth_counts = np.concatenate([
            smc[n_phasebins : n_phasebins + half_bins],
            smc[half_bins : n_phase_bins],
        ])

        # FRI = (smoothed - min) / max
        if smooth_rate.max() > 0:
            fri = (smooth_rate - smooth_rate.min()) / smooth_rate.max()
        else:
            fri = np.zeros_like(smooth_rate)

        firing_rate_per_phase[id] = {
            'raw_rate'      : rate,
            'smooth_rate'   : smooth_rate,
            'fri'           : fri,
            'raw_counts'    : spkcnt,
            'smooth_counts' : smooth_counts,
            'n_spikes'      : phases.sum(),
        }

    return (
        firing_rate_per_phase,
        phase_centers
    )


def classify_theta_modality(
        firing_rate_per_phase, 
        phase_centers, 
        spike_info_with_phase, 
        excitatory_neurons: np.ndarray, 
        phase_bin_size_deg: int = 10, 
        rayleigh_p_cutoff: float = 0.05
    ):
    pass

def calculate_population_firing_rates(
        firing_rate_per_phase,
        modality,
        excitatory_neurons: np.ndarray
    ):
    pass

def classify_place_cell_modality(
        place_fields: np.ndarray,
        spike_info,
        modality,
        excitatory_neurons: np.ndarray,
        velocity_cutoff: float = 5.0,
        min_field_fraction: float = 0.2,
        min_contiguous_bins: int = 20
    ): 
    pass