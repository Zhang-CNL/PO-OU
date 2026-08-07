import numpy as np
import pynapple as nap
import warnings
import time
from dataclasses import dataclass
from typing import Any

import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsep

@hseio.register_type
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
    environment_size   : list[tuple[int,...]]

@hseio.register_type
@dataclass 
class PlaceFields: 
    place_fields   : np.ndarray
    place_cell_ids : np.ndarray
    position_hist  : np.ndarray

@hseio.register_type
@dataclass 
class Theta:
    ground_truth      : list[nap.TsdFrame]
    spikes            : list[np.ndarray]
    lfp_data          : nap.TsdFrame
    trough_times      : nap.Ts
    trough_indices    : np.ndarray
    spikes_with_phase : dict[int, nap.TsdFrame]

@hseio.register_type
@dataclass
class Replay:
    spikes : list[np.ndarray]

def load_raw_data(
        base_data_path: str,
        rat_name: str,
        session: int,
        track_type: str = 'Linear',
        bin_size_cm: int = 2,
        environment_size: list[tuple[int,...]]|None = None,
        loading_kwargs: dict[str, Any] = {
            'ripple_type': 'awake',
            'minimum_dt': np.inf
        },
        placefield_kwargs: dict[str, Any] = {
            'place_field_posterior': True,
            'place_field_gaussian_sd_cm': 2.0,
            'prior_mean_sps': 1.0,
            'prior_beta_s': 0.01,
            'min_spikerate': 1.0,
            'velocity_cutoff': 10.0,
            'flatten_linear': True
        }
    ) -> tuple[RawData,PlaceFields]:
    start = time.time()
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
    print(f"Loading data took {time.time() - start}s")

    start = time.time()
    (
        place_fields,
        place_cell_ids,
        position_histogram,
        environment_size
    ) = hsep.calculate_placefields(
        running_position,
        running_spike_info,
        excitatory_neurons,
        track_type       = track_type,
        environment_size = environment_size,
        bin_size_cm      = bin_size_cm,
        posterior        = placefield_kwargs.get('place_field_posterior', True),
        place_field_gaussian_sd_cm = placefield_kwargs.get('place_field_gaussian_sd_cm', 2.0),
        prior_mean_rat_sps = placefield_kwargs.get('prior_mean_rat_sps', 1.0),
        prior_beta_s       = placefield_kwargs.get('prior_beta_s', .01),
        min_spike_rate     = placefield_kwargs.get('min_spikerate', 1.0), 
        velocity_cutoff    = placefield_kwargs.get('velocity_cutoff', 10.0),
        flatten_linear     = placefield_kwargs.get('flatten_linear', True)
    )
    print(f"Calculating place fields took {time.time() - start}s")

    raw_data = RawData(
        raw_position,
        running_position,
        raw_spikes,
        running_spikes,
        running_spike_info,
        ripple_periods,
        excitatory_neurons,
        inhibitory_neurons,
        lfp_data['LFP'], #Warn about this
        environment_size
    )
    print(f"Dropping LFP metadata. If you want it, call the functions yourself")

    place_fields = PlaceFields(
        place_fields,
        place_cell_ids,
        position_histogram,
    )
    return (
        raw_data,
        place_fields
    )

def process_theta(
        raw_data: RawData,
        placefield_data: PlaceFields,
        velocity_cutoff: float = 10.0,
        theta_kwargs: dict[str, Any] = {
            'time_window_ms': 60,
            'time_window_advance_ms': None,
            'theta_length_s': (0.08, 0.16),
            'max_cycle_duration_s': 1.0,
            'run_period_threshold': 2.0
        },
    ) -> Theta:
    start = time.time()
    (
        theta_lfp_data,
        theta_trough_times,
        theta_trough_indices
    ) = hsep.detect_theta_cycles(
        raw_data.lfp_data,
        theta_length_s       = theta_kwargs.get('theta_length_s', (0.08, 0.16)),
        max_cycle_duration_s = theta_kwargs.get('max_cycle_duration_s', 1.0)
    ) 
    print(f"Detecting theta cycles took {time.time() - start}s")

    start = time.time()
    spike_info_with_phase = hsep.assign_spikes_theta_phase(
        raw_data.running_spike_info,
        theta_lfp_data
    )
    print(f"Aligning spikes to theta phase took {time.time() - start}")

    start = time.time()
    (
        ground_truth,
        spikemats
    ) = hsep.extract_theta_segments(
        raw_data.running_position,
        raw_data.running_spikes,
        theta_lfp_data,
        placefield_data.place_cell_ids,
        time_window_s         = theta_kwargs.get('time_window_ms', 60) / 1000,
        time_window_advance_s = theta_kwargs.get('time_window_advance_s', None),
        velocity_cutoff       = velocity_cutoff,
        run_period_threshold  = theta_kwargs.get('run_period_threshold', 2.0)
    )
    print(f"Extracting theta run sequences took {time.time() - start}")

    return Theta(
        ground_truth,
        spikemats,
        theta_lfp_data,
        theta_trough_times,
        theta_trough_indices,
        spike_info_with_phase
    )

def process_replay(

    ):
    pass
