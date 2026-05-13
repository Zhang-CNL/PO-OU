from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .load_data import *
from .metadata import *
from .placefields import *
from .theta import *
from .ripples import *

@dataclass
class RawData:
    raw_position       : nap.TsdFrame
    running_position   : nap.TsdFrame
    raw_spikes         : nap.TsGroup
    running_spike_info : Dict[int, nap.TsdFrame]
    running_spikes     : nap.TsGroup
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
class Ripples:
    ripple_spikes: List[np.ndarray]

def preprocess_data(
        rat_name: str,
        session: int,
        data_path: str,
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        minimum_dt: float = 0.1,
        environment_size: Optional[Tuple[int]] = None,
        place_field_posterior: bool = True,
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        posterior: bool = True,
        min_spike_rate: float = 1.0,
        velocity_cutoff = 5.0,
        theta_time_window_ms: float = 250.0,
        theta_time_window_advance_ms: Optional[float] = None,
        ripple_time_window_ms: float = 5.0,
        ripple_time_window_advance_ms: Optional[float] = None
    ):

    (
        raw_position,
        running_position,
        raw_spikes,
        running_spike_info,
        running_spikes,
        ripple_intervals,
        excitatory_neurons,
        inhibitory_neurons,
    ) = load_clean_data(
        data_path   = data_path,
        rat_name    = rat_name,
        session     = session,
        track_type  = track_type,
        ripple_type = ripple_type,
        minimum_dt  = minimum_dt
    )

    place_fields, place_cell_ids = calculate_placefields(
        running_position,
        running_spike_info,
        running_spikes,
        excitatory_neurons,
        environment_size           = environment_size,
        track_type                 = track_type,
        posterior                  = place_field_posterior,
        bin_size_cm                = bin_size_cm,
        place_field_gaussian_sd_cm = place_field_gaussian_sd_cm,
        prior_mean_rat_sps         = prior_mean_rat_sps,
        prior_beta_s               = prior_beta_s,
        min_spike_rate             = min_spike_rate,
        velocity_cutoff            = velocity_cutoff
    )

    true_trajectories, tspikemats = process_theta(
        running_position,
        running_spikes,
        place_cell_ids,
        time_window_ms         = theta_time_window_ms,
        time_window_advance_ms = theta_time_window_advance_ms 
    )

    rspikemats = process_ripples(
        ripple_intervals,
        running_spikes,
        place_cell_ids,
        ripple_time_window_ms,
        ripple_time_window_advance_ms
    )

    raw_data = RawData(
        raw_position       = raw_position,
        running_position   = running_position,
        raw_spikes         = raw_spikes,
        running_spike_info = running_spike_info,
        running_spikes     = running_spikes,
        excitatory_neurons = excitatory_neurons,
        inhibitory_neurons = inhibitory_neurons
    )
    place_field_data = PlaceFields(
        place_fields   = place_fields,
        place_cell_ids = place_cell_ids
    )
    theta_data = Theta(
        true_trajectory = true_trajectories,
        theta_spikes    = tspikemats
    )
    ripple_data = Ripples(
        ripple_spikes = rspikemats
    )

    return raw_data, place_field_data, theta_data, ripple_data