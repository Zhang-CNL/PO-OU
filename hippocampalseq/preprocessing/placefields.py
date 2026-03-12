import numpy as np
from scipy.ndimage import gaussian_filter
import hippocampalseq.utils as hseu
from .data import *



def __get_run_data(rat_data: RatData) -> hseu.AttrDict:
    """
    Extracts spikes and positions from the given rat data in the given runs.

    Parameters:
        rat_data (RatData): Dictionary containing rat data

    Returns:
        hseu.AttrDict: Dictionary containing extracted spike and position data
    """
    spike_ids   = rat_data.spike_ids
    spike_times = rat_data.spike_times_sec
    pos_xy      = rat_data.pos_xy_cm
    pos_times   = rat_data.pos_times_sec
    run_starts  = rat_data.run_times_start
    run_ends    = rat_data.run_times_end

    _spike_times = []#np.array([])
    _spike_ids = []#np.array([])
    _run_x_pos = []#np.array([])
    _run_y_pos = []#np.array([])
    for epoch in range(len(run_starts)):
        start = run_starts[epoch]
        end   = run_ends[epoch]
        # extract window indices
        spike_window_bool = hseu.times_to_bool(spike_times, start, end)
        pos_window_bool   = hseu.times_to_bool(pos_times, start, end)
        # extract spikes and positions in this window
        window_spike_times = spike_times[spike_window_bool]
        window_spike_ids   = spike_ids[spike_window_bool]
        window_x_pos       = pos_xy[:, 0][pos_window_bool]
        window_y_pos       = pos_xy[:, 1][pos_window_bool]
        # append to list
        _spike_times.append(window_spike_times)
        _spike_ids.append(window_spike_ids)
        _run_x_pos.append(window_x_pos)
        _run_y_pos.append(window_y_pos)
        #_spike_times = np.append(_spike_times, window_spike_times)
        #_spike_ids   = np.append(_spike_ids, window_spike_ids).astype(int)
        #_run_x_pos   = np.append(_run_x_pos, window_x_pos)
        #_run_y_pos   = np.append(_run_y_pos, window_y_pos)
    
    _spike_times = np.hstack(_spike_times)
    _spike_ids   = np.hstack(_spike_ids)
    pos_xy_cm    = np.vstack((np.hstack(_run_x_pos), np.hstack(_run_y_pos))).T
    #pos_xy_cm = np.array((_run_x_pos, _run_y_pos)).T
    return hseu.AttrDict({
        "spike_times": _spike_times,
        "spike_ids": _spike_ids,
        "run_pos_xy_cm": pos_xy_cm
    })

def __calculate_one_placefield(
        position_hist: np.ndarray,
        spike_hist: np.ndarray,
        place_field_sd_gaussian: float,
        prior_alpha_s: float,
        prior_beta_s: float,
        posterior: bool = True
    ) -> np.ndarray:
    """
    Calculate the place field given the spike histogram and position histogram.

    Parameters:
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
        place_field_raw = np.nan_to_num(spike_hist / position_hist)
    #if self.params.rotate_placefields:
    #    place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=0)
    #    place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=1)
    pf_gaussian_sd_bins = hseu.cm_to_bins(place_field_sd_gaussian)
    place_field_smoothed = gaussian_filter(
        place_field_raw, sigma=pf_gaussian_sd_bins
    )
    return place_field_smoothed

def __get_spike_positions(
        cell_spike_times: np.ndarray,
        pos_xy: np.ndarray, 
        pos_times: np.ndarray,
        position_gap_threshold_s: float
    ) -> np.ndarray:
    """Extracts the positions corresponding to the given spike times.

    Parameters:
        cell_spike_times (np.ndarray): Spike times of a cell
        pos_xy (np.ndarray): Position data
        pos_times (np.ndarray): Position times
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        np.ndarray: Positions corresponding to the given spike times
    """
    cell_spike_pos_xy = np.array(
        [
            __find_position_during_spike(pos_xy, pos_times, time, position_gap_threshold_s)
            for time in cell_spike_times
        ]
    )
    return cell_spike_pos_xy

def __find_position_during_spike(
        pos_xy: np.ndarray, 
        pos_times: np.ndarray, 
        spike_time: float,
        position_gap_threshold_s: float
    ) -> np.ndarray:
    """
    Finds the position corresponding to the given spike time.

    Parameters:
        pos_xy (np.ndarray): Position data
        pos_times (np.ndarray): Position times
        spike_time (float): Spike time
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        np.ndarray: Positions corresponding to the given spike times

    Notes:
        If the spike time does not have a position within the given threshold,
        the function prints a message with the minimum difference and its index.
    """
    abs_diff = np.abs(pos_times - spike_time)
    min_diff = np.min(abs_diff)
    if min_diff > position_gap_threshold_s:
        print(
            "find_pos_ind_nearest_spike() returning value larger than gap "
            f"threshold: {min_diff, np.where(abs_diff == min_diff)}"
        )
    nearest_pos_xy = pos_xy[abs_diff == min_diff][0]
    if nearest_pos_xy.shape != (2,):
        nearest_pos_xy = nearest_pos_xy[0]
    return nearest_pos_xy

def __mean_firing_rate(position_histogram: np.ndarray, spiking_histograms: np.ndarray) -> float:
    """
    Calculate the mean firing rate for a given position histogram and a set of spiking histograms.

    Parameters:
        position_histogram (np.ndarray): Position histogram
        spiking_histograms (np.ndarray): Spiking histograms

    Returns:
        np.array: Mean firing rate (spikes per second)
    """
    total_run_time = np.sum(position_histogram)
    total_spikes = np.sum(spiking_histograms, axis=(1, 2))
    return total_spikes / total_run_time

def calculate_placefields(
        rat_data: RatData,
        bin_size_cm: int = 4,
        place_field_gaussian_sd_cm: float = 4,
        position_gap_threshold_s: float = 0.25,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        posterior: bool = True
    ) -> PlacefieldData:
    """Calculate the place fields of a given rat.

    Parameters:
        rat_data (RatData): Data of the rat
        bin_size_cm (int): Bin size in cm
        place_field_gaussian_sd_cm (float): Standard deviation of the place field in cm
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        PlacefieldData: Place field data

    Notes:
        The function first calculates the position histogram and the spike histograms.
        Then it calculates the place fields for each cell.
        Finally, it calculates the mean firing rate and the maximum firing rate for each cell.
    """
    prior_alpha_s = prior_beta_s * prior_mean_rat_sps + 1
    run_data = __get_run_data(rat_data)

    nbinsx = int(PFEIFFER_ENV_WIDTH_CM / bin_size_cm)
    nbinsy = int(PFEIFFER_ENV_HEIGHT_CM / bin_size_cm)
    spatial_grid_y = np.linspace(0, 200, nbinsy + 1)
    spatial_grid_x = np.linspace(0, 200, nbinsx + 1)

    position_hist,_,_ = np.histogram2d(
            run_data.run_pos_xy_cm[:,0],
            run_data.run_pos_xy_cm[:,1],
            bins=(spatial_grid_x,spatial_grid_y)
        )
    position_hist = position_hist.T * PFEIFFER_RECORDING_FPS


    spike_times = run_data.spike_times #rat_data.spike_times_sec
    spike_ids   = run_data.spike_ids #rat_data.spike_ids
    pos_xy      = rat_data.pos_xy_cm
    pos_times   = rat_data.pos_times_sec

    spike_histograms = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    for cell_id in range(rat_data.n_cells):
        cell_spike_times = spike_times[spike_ids == cell_id]
        cell_spike_pos_xy = __get_spike_positions(
                cell_spike_times,
                pos_xy, 
                pos_times, 
                position_gap_threshold_s
            )
        if len(cell_spike_times) > 0:
            spike_hist, _, _ = np.histogram2d(
                    cell_spike_pos_xy[:, 0],
                    cell_spike_pos_xy[:, 1],
                    bins=(spatial_grid_x, spatial_grid_y),
                )
            spike_histograms[cell_id] = spike_hist.T
        #else:
        #    spike_histograms[cell_id] = np.zeros((nbinsx, nbinsy))

    place_fields = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    for i in range(rat_data.n_cells):
        place_fields[i] = __calculate_one_placefield(
                position_hist,
                spike_histograms[i],
                place_field_gaussian_sd_cm,
                prior_alpha_s,
                prior_beta_s,
                posterior
            )

    mean_fr_array = __mean_firing_rate(position_hist, spike_histograms)

    max_fr_array = np.max(place_fields, axis=(1, 2))

    # Identify place cells using a threshold set by Brad Pfeiffer
    max_tuning_curve_above_thresh = np.squeeze(
        np.argwhere(max_fr_array > PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC)
    )
    place_cell_ids = np.intersect1d(rat_data.excitatory_neurons, max_tuning_curve_above_thresh)

    place_field_data = PlacefieldData(
            place_fields,
            hseu.placefield_matrix(place_fields, place_cell_ids),
            mean_fr_array,
            max_fr_array,
            place_cell_ids,
            len(place_cell_ids)
        )

    return place_field_data