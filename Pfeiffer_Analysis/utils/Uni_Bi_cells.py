
from scipy.signal import find_peaks
from scipy.stats import rayleigh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.stats import rayleightest
from scipy.ndimage import label

def classify_theta_modality(firing_rate_per_phase, phase_centers, spike_info_with_phase,
                            excitatory_neurons, phase_bin=10, rayleigh_p_cutoff=0.05):

    modality_results = {}
    min_width_bins   = 30.0 / phase_bin   # 3.0 bins at phase_bin=10

    for cell_id, data in firing_rate_per_phase.items():
        # Rayleigh test on all spike phases
        rec            = spike_info_with_phase[cell_id]
        cols           = list(rec.columns)
        all_phases_deg = rec.values[:, cols.index('theta_phase')].astype(float)
        all_phases_rad = np.deg2rad(all_phases_deg)

        rayleigh_p = rayleightest(all_phases_rad) if len(all_phases_rad) >= 1 else 1.0
        R          = (np.sqrt(np.mean(np.cos(all_phases_rad))**2 +
                                np.mean(np.sin(all_phases_rad))**2)
                    if len(all_phases_rad) > 0 else 0.0)

        # Finding thee peaks
        fri          = data['fri']
        n_bins       = len(fri)
        wrapped_fri  = np.concatenate([fri, fri])

        # candidate peaks via height + distance + prominence
        candidate_peaks, _ = find_peaks(
            wrapped_fri,
            height     = np.max(wrapped_fri) / 10,
            distance   = int(round(50 / phase_bin)),
            prominence = 0.05,
        )

        # apply 'halfheight' width filter manually
        if len(candidate_peaks) > 0:
            widths   = np.array([matlab_halfheight_width(wrapped_fri, p)
                                for p in candidate_peaks])
            keep     = widths >= min_width_bins
            peaks    = candidate_peaks[keep]
        else:
            peaks    = candidate_peaks

        # Unwrap: map every peak to [0, n_bins), then upper-edge phase
        peaks_in_range = peaks % n_bins
        peak_phases    = (peaks_in_range + 1) * phase_bin
        peak_heights   = wrapped_fri[peaks]

        unique_phases, unique_idx = np.unique(peak_phases, return_index=True)
        peak_data = np.column_stack([peak_heights[unique_idx], unique_phases])
        peak_data = peak_data[np.argsort(-peak_data[:, 0])]
        n_peaks   = len(peak_data)

        if n_peaks == 1 and rayleigh_p <= rayleigh_p_cutoff:
            modality = 1
        elif n_peaks == 2 and rayleigh_p <= rayleigh_p_cutoff:
            modality = 2
        elif n_peaks > 2 and rayleigh_p <= rayleigh_p_cutoff:
            modality = 3
        elif np.max(fri) == 0:
            modality = 0
        else:
            modality = -1

        modality_results[cell_id] = {
            'modality':     modality,
            'n_peaks':      n_peaks,
            'peak_heights': peak_data[:, 0] if n_peaks > 0 else np.array([]),
            'peak_phases':  peak_data[:, 1] if n_peaks > 0 else np.array([]),
            'rayleigh_p':   rayleigh_p,
            'rayleigh_R':   R,
            'n_spikes':     len(all_phases_rad),
        }

    # summary
    modality_names = {-1: 'Non-modal', 0: 'Too few spikes',
                        1: 'Unimodal', 2: 'Bimodal', 3: 'Multimodal'}
    print("\nModality Classification Summary:")
    for mod_val in [1, 2, 3, -1, 0]:
        count = sum(1 for r in modality_results.values() if r['modality'] == mod_val)
        if count > 0:
            print(f"  {modality_names[mod_val]}: {count}")

    population_stats = calculate_population_firing_rates(
        firing_rate_per_phase, modality_results, excitatory_neurons)

    return modality_results, population_stats

def calculate_population_firing_rates(firing_rate_per_phase, modality_results, excitatory_neurons):

    groups = {
        'all_excitatory': {'raw': [], 'smooth': [], 'fri': []},
        'unimodal':       {'raw': [], 'smooth': [], 'fri': []},
        'bimodal':        {'raw': [], 'smooth': [], 'fri': []},
        'inhibitory':     {'raw': [], 'smooth': [], 'fri': []},
    }

    # Coerce excitatory_neurons to a set for fast lookup; tolerate None
    if excitatory_neurons is None:
        exc_set = set(firing_rate_per_phase.keys())  # treat all as excitatory
    else:
        exc_set = set(excitatory_neurons)
        

    for cell_id, data in firing_rate_per_phase.items():
        is_excitatory = cell_id in exc_set
        modality      = modality_results[cell_id]['modality']

        if is_excitatory:
            groups['all_excitatory']['raw'].append(data['raw_rate'])
            groups['all_excitatory']['smooth'].append(data['smooth_rate'])
            groups['all_excitatory']['fri'].append(data['fri']) 

            if modality == 1:
                groups['unimodal']['raw'].append(data['raw_rate'])
                groups['unimodal']['smooth'].append(data['smooth_rate'])
                groups['unimodal']['fri'].append(data['fri'])
            elif modality == 2:
                groups['bimodal']['raw'].append(data['raw_rate'])
                groups['bimodal']['smooth'].append(data['smooth_rate'])
                groups['bimodal']['fri'].append(data['fri'])    
        else:
            groups['inhibitory']['raw'].append(data['raw_rate'])
            groups['inhibitory']['smooth'].append(data['smooth_rate'])
            groups['inhibitory']['fri'].append(data['fri'])    

    def compute_mean_sem(data_list):
        if len(data_list) == 0:
            return None, None
        arr  = np.array(data_list)
        mean = np.mean(arr, axis=0)
        if len(arr) > 1:
            # ddof=1 → sample std, matches MATLAB
            sem = np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr))
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

    print("\nPopulation Summary:")
    for group_name, stats in population_stats.items():
        print(f"  {group_name}: {stats['n_cells']} cells")

    return population_stats

def calculate_place_field_uni_bimodal(field_results, spike_info, modality_results,
                                    excitatory_neurons, velocity_cutoff,
                                    min_field_fraction, min_contiguous_bins):

    Field_Data       = field_results['Field_Data']
    Time_In_Position = field_results['Time_In_Position']
    cell_ids         = field_results['cell_ids']

    total_duration              = np.sum(Time_In_Position)
    Normalized_Time_In_Position = Time_In_Position / total_duration if total_duration > 0 else Time_In_Position

    cell_id_to_idx = {cid: k for k, cid in enumerate(cell_ids)}

    # 8-connectivity structure to match MATLAB's grayconnected
    eight_conn = np.ones((3, 3), dtype=int)

    place_field_properties = {}

    for cell_id in excitatory_neurons:
        if cell_id not in cell_id_to_idx:        continue
        if cell_id not in modality_results:      continue
        modality = modality_results[cell_id]['modality']
        if modality not in [1, 2]:               continue

        k           = cell_id_to_idx[cell_id]
        Place_Field = Field_Data[:, :, k]
        peak_fr     = np.max(Place_Field)
        if peak_fr <= 0:
            continue

        # Mean firing rate from velocity-filtered spikes (positional access)
        cell_spikes = spike_info[cell_id]
        cols        = list(cell_spikes.columns)
        spike_vel   = cell_spikes.values[:, cols.index('velocity')].astype(float)
        n_spikes_running = int((spike_vel >= velocity_cutoff).sum())
        mean_fr     = n_spikes_running / total_duration if total_duration > 0 else 0.0

        # Skaggs spatial information per spike
        if mean_fr > 0:
            rate_ratio  = Place_Field / mean_fr
            valid_mask  = (Normalized_Time_In_Position > 0) & (Place_Field > 0)
            info_field  = np.zeros_like(Place_Field, dtype=float)
            info_field[valid_mask] = (Normalized_Time_In_Position[valid_mask]
                                       * rate_ratio[valid_mask]
                                       * np.log2(rate_ratio[valid_mask]))
            info_field[np.isnan(info_field)] = 0
            info_per_spike = float(np.sum(info_field))
        else:
            info_per_spike = 0.0

        # Connected components on threshold-crossing binary map (8-connectivity)
        binary_field  = (Place_Field >= peak_fr * min_field_fraction).astype(int)
        labeled, n_raw = label(binary_field, structure=eight_conn)

        field_sizes     = []
        valid_mask_2d   = np.zeros_like(labeled)
        n_fields        = 0
        for fid in range(1, n_raw + 1):
            this_mask = (labeled == fid)
            sz        = int(this_mask.sum())
            if sz >= min_contiguous_bins:
                n_fields += 1
                valid_mask_2d[this_mask] = n_fields
                field_sizes.append(sz)

        # In-field stats — NaN when no fields (matches MATLAB mean([]) → NaN)
        if n_fields > 0:
            in_field_mask = valid_mask_2d > 0
            #  mean only over actual in-field bins
            mean_infield_firing_rate = float(Place_Field[in_field_mask].mean())

            mean_field_size = float(np.mean(field_sizes))
        else:
            mean_infield_firing_rate = np.nan
            mean_field_size          = np.nan

        place_field_properties[cell_id] = {
            'modality':                 modality,
            'mean_field_size':          mean_field_size,
            'n_fields':                 n_fields,
            'peak_firing_rate':         float(peak_fr),
            'mean_firing_rate':         mean_fr,
            'information_per_spike':    info_per_spike,
            'mean_infield_firing_rate': mean_infield_firing_rate,
            'field_sizes':              field_sizes,
        }

    # Separate by modality
    keys = ['mean_field_size', 'n_fields', 'peak_firing_rate', 'mean_firing_rate',
            'information_per_spike', 'mean_infield_firing_rate']
    unimodal_props = {k: [] for k in keys}
    bimodal_props  = {k: [] for k in keys}

    for cid, props in place_field_properties.items():
        target = unimodal_props if props['modality'] == 1 else bimodal_props
        for k in keys:
            target[k].append(props[k])

    for props in (unimodal_props, bimodal_props):
        for k in props:
            props[k] = np.array(props[k], dtype=float)

    print(f"\nPlace Field Properties Summary:")
    print(f"  Unimodal cells analyzed: {len(unimodal_props['mean_field_size'])}")
    print(f"  Bimodal cells analyzed:  {len(bimodal_props['mean_field_size'])}")

    return place_field_properties, unimodal_props, bimodal_props

