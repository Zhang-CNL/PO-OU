import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter

from .metadata import * 
from .segment_runs import *
import hippocampalseq.utils as hseu


def __get_run_data(
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        time: np.ndarray,
        run_starts: np.ndarray,
        run_ends: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts spikes and positions from the given rat data in the given runs.

    Args:
        rat_data (hseu.RatData): Dictionary containing rat data

    Returns:
        hseu.AttrDict: Dictionary containing extracted spike and position data
    """
    _spike_times = []
    _spike_ids   = []
    _run_x_pos   = []
    _run_y_pos   = []
    _run_time    = []
    for start,end in zip(run_starts,run_ends):
        spike_window = hseu.restrict_indices(spike_times, start, end)
        pos_window   = hseu.restrict_indices(time, start, end)

        window_spike_times = spike_times[spike_window]
        window_spike_ids   = spike_ids[spike_window] 
        window_x_pos       = x[pos_window]
        window_y_pos       = y[pos_window]
        
        _spike_times.append(window_spike_times)
        _spike_ids.append(window_spike_ids)
        _run_x_pos.append(window_x_pos)
        _run_y_pos.append(window_y_pos)
        _run_time.append(time[pos_window])
    
    _spike_times = np.hstack(_spike_times)
    _spike_ids   = np.hstack(_spike_ids)
    _x           = np.hstack(_run_x_pos)
    _y           = np.hstack(_run_y_pos)
    _t           = np.hstack(_run_time)
    return _spike_times, _spike_ids, _x, _y, _t

def __calculate_one_placefield(
        position_hist: np.ndarray,
        spike_hist: np.ndarray,
        place_field_sd_gaussian: float,
        prior_alpha_s: float,
        prior_beta_s: float,
        bin_size_cm: float = 2.0,
        posterior: bool = True
    ) -> np.ndarray:
    """
    Calculate the place field given the spike histogram and position histogram.

    Args:
        position_hist (np.ndarray): Position histogram
        spike_hist (np.ndarray): Spike histogram
        place_field_sd_gaussian (float): Standard deviation of the place field

    Returns:
        np.ndarray: Smoothed place field
    """
    if posterior:
        spike_hist_with_prior = spike_hist + prior_alpha_s - 1
        pos_hist_with_prior_s = position_hist + prior_beta_s
        place_field_raw = spike_hist_with_prior / pos_hist_with_prior_s
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            place_field_raw = np.nan_to_num(spike_hist / position_hist, posinf=0, neginf=0)
    pf_gaussian_sd_bins = hseu.cm_to_bins(place_field_sd_gaussian, bin_size_cm)
    place_field_smoothed = gaussian_filter(
        place_field_raw, sigma=pf_gaussian_sd_bins
    )
    return place_field_smoothed


def calculate_placefields(
        rat_data: hseu.RatData,
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        position_gap_threshold_s: float = 0.25,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        posterior: bool = True
    ) -> hseu.PlacefieldData:
    """Calculate the place fields of a given rat.

    Args:
        rat_data (hseu.RatData): Data of the rat
        bin_size_cm (int): Bin size in cm
        place_field_gaussian_sd_cm (float): Standard deviation of the place field in cm
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        hseu.PlacefieldData: Place field data

    Notes:
        The function first calculates the position histogram and the spike histograms.
        Then it calculates the place fields for each cell.
        Finally, it calculates the mean firing rate and the maximum firing rate for each cell.
    """
    prior_alpha_s = prior_beta_s * prior_mean_rat_sps + 1

    t_all       = rat_data.raw_data.time
    x_all       = rat_data.raw_data.x
    y_all       = rat_data.raw_data.y

    spike_times = rat_data.run_data.spike_times
    spike_ids   = rat_data.run_data.spike_ids
    x_run = rat_data.run_data.x
    y_run = rat_data.run_data.y
    t_run = rat_data.run_data.time

    nbinsx = int(PFEIFFER_ENV_WIDTH_CM / bin_size_cm)
    nbinsy = int(PFEIFFER_ENV_HEIGHT_CM / bin_size_cm)
    spatial_grid_x = np.linspace(0, PFEIFFER_ENV_HEIGHT_CM, nbinsx + 1)
    spatial_grid_y = np.linspace(0, PFEIFFER_ENV_WIDTH_CM, nbinsy + 1)

    # Calculate time occupancy in each spatial position bin
    position_hist,_,_ = np.histogram2d(
        x_all[:-1], y_all[:-1],
        bins=(spatial_grid_x,spatial_grid_y),
        weights=np.diff(t_all)
    )
    position_hist = position_hist.T 

    place_fields = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    spike_histograms = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    for cell_id in range(rat_data.n_cells):
        cell_idx = np.where(spike_ids == cell_id)[0]

        cell_x = x_run[cell_idx]
        cell_y = y_run[cell_idx]

        if len(cell_idx) > 0:
            spike_hist, _, _ = np.histogram2d(
                cell_x,
                cell_y,
                bins=(spatial_grid_x, spatial_grid_y),
            )
            spike_histograms[cell_id] = spike_hist.T

        place_fields[cell_id] = __calculate_one_placefield(
                position_hist,
                spike_histograms[cell_id],
                place_field_gaussian_sd_cm,
                prior_alpha_s,
                prior_beta_s,
                bin_size_cm,
                posterior
            )

    mean_fr_array = np.sum(spike_histograms, axis=(1, 2)) / np.sum(position_hist)
    max_fr_array = np.max(place_fields, axis=(1, 2))

    # Identify place cells using a threshold set by Brad Pfeiffer
    max_tuning_curve_above_thresh = np.squeeze(
        np.argwhere(max_fr_array > PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC)
    )
    place_cell_ids = np.intersect1d(rat_data.excitatory_neurons, max_tuning_curve_above_thresh)

    place_field_data = hseu.PlacefieldData(
        place_fields     = place_fields[place_cell_ids],
        mean_firing_rate = mean_fr_array,
        max_firing_rate  = max_fr_array,
        place_cell_ids   = place_cell_ids,
        n_place_cells    = len(place_cell_ids),
    )

    return place_field_data