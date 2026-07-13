import numpy as np
from typing import Tuple
from scipy.signal.windows import gaussian

import hippocampalseq.utils as utils

# TODO: Ask Eryn what all of these functions are for.

def calculate_phase_locking(
        spike_info_with_phase,
        total_duration,
        velocity_cutoff: float = 5.0,
        phase_bin_size_deg: int = 10,
        gaussian_std: float = 12,
        limit_analysis_by_theta_length: bool = True,
        theta_length_s: Tuple[float, float] = (0.08, 0.16),
        minimum_spike_count: int = 100,
    ): 
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