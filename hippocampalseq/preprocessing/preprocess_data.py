import copy
import numpy as np
from typing import Optional

import hippocampalseq.utils as hseu
from .load_data import *
from .placefields import * 
from .ripples import *
from .theta import *


def load_and_preprocess(
        data_path: str,
        rat_name: str, 
        session: int, 
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        time_window_ms: float = 3.0,
        time_window_advance_ms: Optional[float] = None,
        avg_fr_smoothing_convolution: np.ndarray = np.array([.25, .25, .25, .25]),
        avg_spikes_per_s_threshold: int = 2,
        min_popburst_duration_ms: int = 30,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        seed: int|None = 42
    ) -> hseu.RatData:
    """Runs all preprocessing steps on the given rat data.

    Args:
        data_path (str): Path to data directory
        rat_name (str): Name of the rat
        session (int): Session number (1 or 2)
        position_gap_threshold_s (float): Threshold for position gaps in seconds. Defaults to 0.25.
        velocity_run_threshold_s (float): Threshold for determining if the rat is running in centimeters per second. Defaults to 5.0.
        bin_size_cm (int): Bin size for place field calculation in centimeters. Defaults to 4.
        place_field_gaussian_sd_cm (float): Standard deviation of Gaussian for place field calculation in centimeters. Defaults to 4.0.

    Returns:
        hseu.RatData: Dictionary containing rat data with additional fields for velocity and place field data
    """
    #rat_data = load_rat(data_path, rat_name, session, sweep_type)
    #rat_data = clean_data(rat_data, position_gap_threshold_s)
    #rat_data = calculate_velocity(rat_data, velocity_run_threshold_s)
    rat_data = load_clean_data(
        data_path, 
        rat_name, 
        session, 
        track_type,
        ripple_type,
        position_gap_threshold_s, 
        velocity_run_threshold_s
    )
    rat_data.place_field_data = calculate_placefields(
        rat_data,
        bin_size_cm, 
        place_field_gaussian_sd_cm,
        position_gap_threshold_s
    )
    rat_data.ripple_data = process_ripples(
        rat_data,
        time_window_ms,
        time_window_advance_ms,
        avg_fr_smoothing_convolution,
        avg_spikes_per_s_threshold,
        min_popburst_duration_ms
    )

    rat_data.theta_data = process_theta(
        rat_data, 
        run_period_threshold, 
        place_field_scaling_factor,
        velocity_scaling_factor,
        time_window_ms,
        time_window_advance_ms,
        seed
    )

    return rat_data