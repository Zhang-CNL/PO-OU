from typing import Optional, Tuple, List
from dataclasses import dataclass

from .load_data import *
from .metadata import *
from .placefields import *
from .theta import *

@dataclass 
class PlaceFields:
    place_fields: np.ndarray
    place_cell_ids: np.ndarray

@dataclass 
class Theta:
    true_trajectory: List[np.ndarray]
    theta_spikes: List[np.ndarray]


def preprocess_data(
        rat_name: str,
        session: int,
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
        theta_time_window_advance_ms: Optional[float] = None
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
        posterior                  = place_field_posterior,
        min_spike_rate             = min_spike_rate,
        velocity_cutoff            = velocity_cutoff
    )

    true_trajectories, spikemats = process_theta(
        running_position,
        running_spikes,
        place_cell_ids,
        time_window_ms         = theta_time_window_ms,
        time_window_advance_ms = theta_time_window_advance_ms 
    )

    place_field_data = PlaceFields(
        place_fields   = place_fields,
        place_cell_ids = place_cell_ids
    )
    theta_data = Theta(
        true_trajectory = true_trajectories,
        theta_spikes    = spikemats
    )