import numpy as np
import pynapple as nap
import warnings
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsepp

@dataclass 
class RawData:
    raw_position       : nap.TsdFrame 
    running_position   : nap.TsdFrame 
    raw_spikes         : nap.TsGroup 
    running_spikes     : nap.TsGroup
    running_spike_info : dict[int, nap.TsdFrame]
    ripple_periods     : nap.IntervalSet
    excitatory_neurons : np.ndarray
    inhibitory_neurons : np.ndarray
    lfp_data           : nap.TsdFrame
    environment_size   : tuple[int,...]

@dataclass 
class PlaceFields: 
    place_fields   : np.ndarray
    place_cell_ids : np.ndarray
    position_hist  : np.ndarray

@dataclass 
class Theta:
    true_trajectory : list[np.ndarray]
    spikes          : list[np.ndarray]
    lfp_data        : nap.TsdFrame
    trough_times    : nap.Ts
    trough_indices  : np.ndarray
    spikes_with_phase : dict[int, nap.TsdFrame]

@dataclass
class Replay:
    spikes : list[np.ndarray]

def load_and_preprocess(
        base_data_path: str,
        rat_name: str,
        session: int,
        track_type: str = 'Linear',
        environment_size: Optional[Tuple[int,...]] = None,
        bin_size_cm: int = 2,
        loading_kwargs: Dict[str, Any] = {
            'ripple_type': 'awake',
            'minimum_dt': np.inf
        },
        placefield_kwargs: Dict[str, Any] = {
            'place_field_posterior': True,
            'place_field_gaussian_sd_cm': 2.0,
            'prior_mean_sps': 1.0,
            'prior_beta_s': 0.01,
            'min_spikerate': 1.0,
            'velocity_cutoff': 5.0
        },
        theta_kwargs: Dict[str, Any] = {
            'time_window_ms': 60,
            'time_window_advance_ms': None,
            'limit_analysis_by_theta_length': True,
            'theta_length_s': (0.08, 0.16),
            'max_cycle_duration_s': 1.0
        },
        ripple_kwargs: Dict[str, Any] = {
            'time_window_ms': 5.0,
            'time_window_advance_ms': None
        }
    ) -> Tuple[RawData, PlaceFields, Theta, Replay]:

    # Load raw data and raw data segmented into the running period.
    begin = time.time()
    (
        raw_position,
        running_position,
        raw_spikes,
        running_spike_info,
        running_spikes,
        ripple_periods,
        lfp_data,
        excitatory_neurons,
        inhibitory_neurons
    ) = hseio.load_clean_data(
        base_data_path,
        rat_name,
        session,
        track_type,
        ripple_type = loading_kwargs.get('ripple_type', 'awake'),
        minimum_dt  = loading_kwargs.get('minimum_dt', np.inf)
    )
    print(f"Loading took {time.time() - begin} seconds")

    warnings.warn("Dropping LFP metadata. If you want to keep it, you can call `hseio.load_clean_data` manually.")
    lfp_data = lfp_data['LFP']



    # Generate place fields based on the data
    begin = time.time()
    (
        place_fields,
        place_cell_ids,
        position_histogram,
        environment_size
    ) = hsepp.calculate_placefields(
        running_position,
        running_spike_info,
        excitatory_neurons,
        environment_size   = environment_size,
        track_type         = track_type,
        posterior          = placefield_kwargs.get('place_field_posterior', True),
        bin_size_cm        = bin_size_cm, 
        place_field_gaussian_sd_cm = placefield_kwargs.get('place_field_gaussian_sd_cm', 2.0),
        prior_mean_rat_sps = placefield_kwargs.get('prior_mean_rat_sps', 1.0),
        prior_beta_s       = placefield_kwargs.get('prior_beta_s', .01),
        min_spike_rate     = placefield_kwargs.get('min_spikerate', 1.0), 
        velocity_cutoff    = placefield_kwargs.get('velocity_cutoff', 5.0)
    )
    print(f"Place field calculation took {time.time() - begin} seconds")

    # Process theta and theta LFP
    begin = time.time()
    (
        true_trajectories,
        theta_spikemats 
    ) = hsepp.extract_theta_segments(
        running_position,
        running_spikes,
        place_cell_ids,
        velocity_cutoff            = placefield_kwargs.get('velocity_cutoff', 5.0),
        # Ignore these scaling factors for non-simulated data.
        place_field_scaling_factor = 1.0,
        velocity_scaling_factor    = 1.0,
        time_window_ms             = theta_kwargs.get('time_window_ms', 60),
        time_window_advance_ms     = theta_kwargs.get('time_window_advance_ms', None)
    )
    print(f"Theta segment extraction took {time.time() - begin} seconds")

    begin = time.time()
    (
        theta_lfp_data,
        theta_trough_times,
        theta_trough_indices
    ) = hsepp.detect_theta_cycles(
        lfp_data,
        limit_analysis_by_theta_length = theta_kwargs.get('limit_analysis_by_theta_length', True),
        theta_length_s                 = theta_kwargs.get('theta_length_s', (0.08, 0.16)),
        max_cycle_duration_s           = theta_kwargs.get('max_cycle_duration_s', 1.0)
    )

    spike_info_with_phase = hsepp.assign_spikes_theta_phase(
        running_spike_info,
        theta_lfp_data
    )
    print(f"Theta cycle detection took {time.time() - begin} seconds")

    # Process replay data.
    begin = time.time()
    ripple_spikemats = hsepp.process_ripples(
        ripple_periods,
        running_spikes,
        place_cell_ids,
        time_window_ms         = ripple_kwargs.get('time_window_ms', 5.0),
        time_window_advance_ms = ripple_kwargs.get('time_window_advance_ms', None),
    )
    print(f"Ripple processing took {time.time() - begin} seconds")

    raw_data = RawData(
        raw_position,
        running_position,
        raw_spikes,
        running_spikes,
        running_spike_info,
        ripple_periods,
        excitatory_neurons,
        inhibitory_neurons,
        lfp_data,
        environment_size
    )

    place_field_data = PlaceFields(
        place_fields, 
        place_cell_ids, 
        position_histogram
    )

    theta = Theta(
        true_trajectories, 
        theta_spikemats,
        theta_lfp_data,
        theta_trough_times,
        theta_trough_indices,
        spike_info_with_phase
    )

    ripples = Replay(ripple_spikemats)

    return (
        raw_data,
        place_field_data,
        theta,
        ripples
    )