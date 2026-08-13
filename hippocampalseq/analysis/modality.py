import copy
import numpy as np
import pynapple as nap
from typing import Any,Optional
from collections.abc import Iterable
from scipy.signal import find_peaks, filtfilt
from scipy.signal.windows import gaussian
from scipy.ndimage import label

import hippocampalseq.utils as hseu

# TODO: Ask Eryn what all of these functions are for.

def map_theta_modalities_to_str(modalities: dict[int, Any]) -> list[str]:
    """Convert theta modality numerical codes to strings.
    Call this with the output of `calculate_theta_modality`

    Args:
        modalities (dict[int, Any]): Output of `calculate_theta_modality`

    Returns:
        list[str]: List of strings where each element corresponds to the modality of a cycle.
    """
    modality_names ={
        -1: 'Non-model',
        0: 'Too few spikes',
        1: 'Unimodal',
        2: 'Bimodal',
        3: 'Multimodal'
    }
    named = []
    for modality in modalities.values():
        named.append(modality_names[modality['modality']])
    return named

def calculate_phase_locking(
        spike_info_with_phase: dict[int, nap.TsdFrame],
        total_duration: float,
        velocity_cutoff: float = 5.0,
        phase_bin_size_deg: int = 10,
        gaussian_std: float = 12,
        limit_analysis_by_theta_length: bool = True,
        theta_length_s: tuple[float, float] = (0.08, 0.16),
        minimum_spike_count: int = 100,
    ) -> tuple[dict[int,Any], np.ndarray]: 
    """

    Args:
        spike_info_with_phase
        total_duration
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        phase_bin_size_deg (int):
        gaussian_std (float):
        limit_analysis_by_theta_length (bool): Use `theta_length_s` to filter out LFP segments that are too short. Defaults to True. 
        theta_length_s (tuple[float, float]): Minimum and maximum lengths allowed for a cycle to be used in analysis. Defaults to (0.08,0.16).
        minimum_spike_count (int): 
    """
    n_phasebins = int(360 / phase_bin_size_deg)
    phase_edges = np.arange(0, 361, phase_bin_size_deg)
    phase_centers = phase_edges[:-1] + phase_bin_size_deg / 2

    kernel_size = 7 # 7 bins
    kernel_sigma = gaussian_std / phase_bin_size_deg # std = gaussian_std / phase_bins
    g = gaussian(kernel_size, kernel_sigma)
    g /= g.sum()

    firing_rate_per_phase = {}

    for id,frame in spike_info_with_phase.items():
        phases    = frame['Phase Deg'].values
        velocity  = frame['Velocity'].values
        cyc_dur   = frame['Cycle Duration'].values 
        monotonic = frame['Monotonic Increasing'].values 

        if len(phases) == 0:
            continue

        phases = phases.copy()
        phases[phases==0] = 360

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

        double_counts = np.concatenate([spkcnt,spkcnt]).astype(float)
        smc           = filtfilt(g, [1.0], double_counts, padlen=padlen)
        smooth_counts = np.concatenate([
            smc[n_phasebins : n_phasebins + half_bins],
            smc[half_bins : n_phasebins],
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
            'n_spikes'      : np.sum(phases),
        }

    return (
        firing_rate_per_phase,
        phase_centers
    )

def classify_theta_modality(
        firing_rate_per_phase: dict[int, Any], 
        spike_info_with_phase: dict[int, nap.TsdFrame], 
        phase_bin_size_deg: int = 10, 
        rayleigh_p_cutoff: float = 0.05
    ) -> dict[int, Any]:
    modalities = {}
    min_width_bins = 30 / phase_bin_size_deg 
    
    for id,data in firing_rate_per_phase.items():
        rec = spike_info_with_phase[id]
        phases_rad = np.deg2rad(rec['Phase Deg'].values)

        if len(phases_rad) >= 1:
            rayleigh_p = hseu.rayleightest(phases_rad)
        else: 
            rayleigh_p = 1.0
        if len(phases_rad) == 0:
            R = 0
        else:
            R = np.sqrt(
                np.mean(np.cos(phases_rad))**2 + np.mean(np.sin(phases_rad))**2
            )

        fri    = data['fri']
        n_bins = len(fri)
        wfri   = np.concatenate([fri, fri])

        candidate_peaks,_ = find_peaks(
            wfri,
            height     = np.max(wfri) / 10,
            distance   = int(round(50 / phase_bin_size_deg)),
            prominence = 0.05,
        )
        if len(candidate_peaks) == 0:
            peaks = candidate_peaks
        else:
            widths = np.array([
                hseu.find_halfheight_peaks(wfri, p) for p in candidate_peaks
            ])
            peaks = candidate_peaks[widths >= min_width_bins]

        peaks_in_range = peaks % n_bins 
        peak_phases    = (peaks_in_range + 1) * phase_bin_size_deg
        peak_heights   = wfri[peaks]

        unique_phase,unique_idx = np.unique(peak_phases, return_index=True)
        peak_data = np.column_stack([peak_heights[unique_idx], unique_phase])
        peak_data = peak_data[np.argsort(-peak_data[:,0])]
        n_peaks   = len(peak_data)

        if n_peaks > 0 and rayleigh_p <= rayleigh_p_cutoff:
            modality = min(n_peaks, 3)
        elif np.max(fri) == 0:
            modality = 0
        else:
            modality = -1

        modalities[id] = {
            'modality'     : modality,
            'n_peaks'      : n_peaks,
            'peak_heights' : peak_data[:,0] if n_peaks > 0 else [],
            'peak_phases'  : peak_data[:,1] if n_peaks > 0 else [],
            'rayleigh_p'   : rayleigh_p,
            'rayleigh_R'   : R,
            'n_spikes'     : len(phases_rad),
        }

    return modalities

def calculate_population_firing_rates(
        firing_rate_per_phase: dict[int,Any],
        modalities: dict[int, Any],
        excitatory_neurons: Optional[np.ndarray|Iterable[int]] = None
    ) -> dict[str, dict[str, Any]]:
    """

    Args:
        firing_rate_per_phase (dict[int,Any]):
        modalities (dict[int,Any]):
        excitatory_neurons (np.ndarray|Iterable[int]|None):
    
    Returns:
        (dict[str,dict[str,Any]]):
    """
    if excitatory_neurons is None:
        excitatory_neurons = firing_rate_per_phase.keys()
    excitatory_neurons = set(excitatory_neurons)


    groups = {
        'all_excitatory': {'raw': [], 'smooth': [], 'fri': []},
        'unimodal':       {'raw': [], 'smooth': [], 'fri': []},
        'bimodal':        {'raw': [], 'smooth': [], 'fri': []},
        'inhibitory':     {'raw': [], 'smooth': [], 'fri': []},
    }

    exc  = groups['all_excitatory']
    uni  = groups['unimodal']
    bi   = groups['bimodal']
    inh  = groups['inhibitory']

    for cell_id, data in firing_rate_per_phase.items():
        raw, smooth, fri = data['raw_rate'], data['smooth_rate'], data['fri']

        if cell_id in excitatory_neurons:
            exc['raw'].append(raw)
            exc['smooth'].append(smooth)
            exc['fri'].append(fri)

            modality = modalities[cell_id]['modality']
            if modality == 1:
                uni['raw'].append(raw)
                uni['smooth'].append(smooth)
                uni['fri'].append(fri)
            elif modality == 2:
                bi['raw'].append(raw)
                bi['smooth'].append(smooth)
                bi['fri'].append(fri)
        else:
            inh['raw'].append(raw)
            inh['smooth'].append(smooth)
            inh['fri'].append(fri)

    def compute_mean_sem(data_list):
        n = len(data_list)
        if n == 0:
            return None, None
        arr  = np.asarray(data_list)
        mean = np.mean(arr, axis=0)
        if n > 1:
            sem = np.std(arr, axis=0, ddof=1) / np.sqrt(n)
        else:
            sem = np.zeros_like(mean)
        return mean, sem
        
    population_stats = {}
    for group_name, group_data in groups.items():
        raw_mean,    raw_sem    = compute_mean_sem(group_data['raw'])
        smooth_mean, smooth_sem = compute_mean_sem(group_data['smooth'])
        fri_mean,    fri_sem    = compute_mean_sem(group_data['fri'])

        population_stats[group_name] = {
            'raw_rate':    (raw_mean,    raw_sem),
            'smooth_rate': (smooth_mean, smooth_sem),
            'rate_index':  (fri_mean,    fri_sem),
            'n_cells':     len(group_data['raw']),
        }

    return population_stats


def classify_place_cell_modality(
        place_fields: np.ndarray,
        place_cell_ids: np.ndarray,
        position_histogram: np.ndarray,
        spike_info: dict[int, nap.TsdFrame],
        modalities: dict[int, Any],
        excitatory_neurons: np.ndarray,
        velocity_cutoff: float = 5.0,
        min_field_fraction: float = 0.2,
        min_contiguous_bins: int = 20
    ): 

    if place_fields.ndim < 3:
        place_fields = place_fields[...,np.newaxis]
    if position_histogram.ndim < 2:
        position_histogram = position_histogram[:,np.newaxis]
    # 8-connectivity structure like in MATLAB's grayconnected
    eight_conn = np.ones((3,3), dtype=int)
    cell2idx = {cid: k for k,cid in enumerate(place_cell_ids)}

    total_duration = np.sum(position_histogram)
    normalized_time_in_position = position_histogram 
    if total_duration > 0:
        normalized_time_in_position /= total_duration

    place_cell_properties = {}

    unimodal_cell_properties = {
        'mean_field_size'         : [],
        'n_fields'                : [],
        'peak_firing_rate'        : [],
        'mean_firing_rate'        : [],
        'information_per_spike'   : [],
        'mean_infield_firing_rate': [],
        'field_sizes'             : []
    }
    bimodal_cell_properties = copy.deepcopy(unimodal_cell_properties)

    for id in excitatory_neurons:
        if id not in cell2idx or id not in modalities:
            continue
        modality = modalities[id]['modality']
        if modality not in [1,2]:
            continue

        cell_idx = cell2idx[id]
        place_field = place_fields[cell_idx]
        peak_fr = np.max(place_field)
        if peak_fr <= 0:
            continue

        cell_spikes = spike_info[id]
        spike_vel = cell_spikes['Velocity'].values
        n_spikes_running = int(np.sum(spike_vel >= velocity_cutoff))
        mean_fr = np.nan_to_num(n_spikes_running / total_duration, nan=0.0)

        if mean_fr == 0:
            info_p_spike = 0
        else:
            valid_mask = (normalized_time_in_position > 0) & (place_field > 0)
            rate_ratio = (place_field / mean_fr)[valid_mask]
            info_field = np.zeros_like(place_field, dtype=float)
            info_field[valid_mask] = (normalized_time_in_position[valid_mask]
                * rate_ratio
                * np.log2(rate_ratio)
            )
            info_field = np.nan_to_num(info_field, nan=0)
            info_p_spike = float(np.sum(info_field))

        binary_field = (place_field >= peak_fr * min_field_fraction).astype(int)
        labeled,n_raw = label(binary_field, structure=eight_conn)

        counts = np.bincount(labeled.ravel(), minlength=n_raw + 1)
        sizes = counts[1:n_raw+1]
        valid_ids = np.nonzero(sizes >= min_contiguous_bins)[0]
        field_sizes = sizes[valid_ids]
        n_fields = len(field_sizes)

        valid_mask = np.zeros(n_raw+1, dtype=labeled.dtype)
        valid_mask[valid_ids] = np.arange(1, n_fields+1)
        valid_mask = valid_mask[labeled]

        mean_infield_fr = mean_field_size = np.nan
        if n_fields > 0:
            infield_mask = valid_mask > 0
            mean_infield_fr = place_field[infield_mask].mean()
            mean_field_size = np.mean(field_sizes)

        if modality == 1:
            to_set = unimodal_cell_properties
        else:
            to_set = bimodal_cell_properties 

        to_set['mean_field_size'].append(mean_field_size)
        to_set['n_fields'].append(n_fields)
        to_set['peak_firing_rate'].append(peak_fr)
        to_set['mean_firing_rate'].append(mean_fr)
        to_set['information_per_spike'].append(info_p_spike)
        to_set['mean_infield_firing_rate'].append(mean_infield_fr)
        to_set['field_sizes'].append(field_sizes)

        place_cell_properties[id] = {
            'modality'                 : modality,
            'mean_field_size'          : mean_field_size,
            'n_fields'                 : n_fields,
            'peak_firing_rate'         : peak_fr,
            'mean_firing_rate'         : mean_fr,
            'information_per_spike'    : info_p_spike,
            'mean_infield_firing_rate' : mean_infield_fr,
            'field_sizes'              : field_sizes
        }

    return (
        place_cell_properties,
        unimodal_cell_properties,
        bimodal_cell_properties
    )

