import numpy as np
import pynapple as nap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsepp

@dataclass 
class RawData:
    raw_position       : nap.TsdFrame 
    running_position   : nap.TsdFrame 
    raw_spikes         : nap.TsGroup 
    running_spikes     : nap.TsGroup
    ripple_periods     : nap.IntervalSet
    excitatory_neurons : np.ndarray
    inhibitory_neurons : np.ndarray

@dataclass 
class PlaceFields: 
    place_fields   : np.ndarray
    place_cell_ids : np.ndarray

@dataclass 
class Theta:
    true_trajectory : List[np.ndarray]
    theta_spikes    : List[np.ndarray]

@dataclass
class Replay:
    ripple_spikes : List[np.ndarray]

def load_and_preprocess(
        base_data_path: str,
        rat_name: str,
        session: int,
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        minimum_dt: float = np.inf,
        environment_size: Optional[Tuple[int,...]] = None,
        place_field_posterior: bool = True,
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        min_spikerate: float = 1.0,
        velocity_cutoff: float = 5.0,
        theta_time_window_ms: float = 60,
        theta_time_window_advance_ms: Optional[float] = None,
        ripple_time_window_ms: float = 5.0,
        ripple_time_window_advance_ms: Optional[float] = None
    ):

    # Load raw data and raw data segmented into the running period.
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
        ripple_type,
        minimum_dt
    )

    raw_data = RawData(
        raw_position,
        running_position,
        raw_spikes,
        running_spikes,
        ripple_periods,
        excitatory_neurons,
        inhibitory_neurons
    )

    # Generate place fields based on the data
    (
        place_fields,
        place_cell_ids
    ) = hsepp.calculate_placefields(
        running_position,
        running_spike_info,
        running_spikes,
        excitatory_neurons,
        environment_size   = environment_size,
        track_type         = track_type,
        posterior          = place_field_posterior,
        bin_size_cm        = bin_size_cm, 
        place_field_gaussian_sd_cm = place_field_gaussian_sd_cm,
        prior_mean_rat_sps = prior_mean_rat_sps,
        prior_beta_s       = prior_beta_s,
        min_spike_rate     = min_spikerate, 
        velocity_cutoff    = velocity_cutoff
    )
    place_field_data = PlaceFields(place_fields, place_cell_ids)

    # Process theta and theta LFP
    (
        true_trajectories,
        theta_spikemats 
    ) = hsepp.process_theta(
        running_position,
        running_spikes,
        place_cell_ids,
        velocity_cutoff            = velocity_cutoff,
        # Ignore these scaling factors for non-simulated data.
        place_field_scaling_factor = 1.0,
        velocity_scaling_factor    = 1.0,
        time_window_ms             = theta_time_window_ms,
        time_window_advance_ms     = theta_time_window_advance_ms
    )
    theta = Theta(true_trajectories, theta_spikemats)

    # Process replay data.
    ripple_spikemats = hsepp.process_ripples(
        ripple_periods,
        running_spikes,
        place_cell_ids,
        ripple_time_window_ms,
        ripple_time_window_advance_ms,
    )
    ripples = Replay(ripple_spikemats)

    return (
        raw_data,
        place_field_data,
        theta,
        ripples
    )