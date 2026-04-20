import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter

import hippocampalseq.utils as hseu

def fix_alignment(spike_times, pos_times, spike_ids, x, y, min_time_diff=np.inf):
    """Fixes alignment between spike times and position times by removing spikes
     that are not within the position recording period and position points that 
     are not within the spike recording period.

    Args:
        spike_times (np.ndarray): Times of the spikes
        pos_times (np.ndarray): Times of the position data
        spike_ids (np.ndarray): IDs of the spikes
        x (np.ndarray): x position data
        y (np.ndarray): y position data

    Returns:
        tuple: spike_times, pos_times, spike_ids, x, y
    """

    _st = []
    _id = []
    _x  = []
    _y  = []
    _t  = []
    uids = np.unique(spike_ids)
    for uid in uids:
        st = spike_times[spike_ids == uid]
        idx = np.searchsorted(pos_times, st)

        idp = np.clip(idx - 1, 0, len(pos_times) - 1)
        idc = np.clip(idx, 0, len(pos_times) - 1)

        dtp = np.abs(st - pos_times[idp])
        dtc = np.abs(st - pos_times[idc])
        prev = dtp <= dtc

        nn = np.where(prev, idp, idc)
        td = np.where(prev, dtp, dtc)

        valid = td <= min_time_diff
        selectioni = nn[valid]
        selectiont = st[valid]

        _x.append(x[selectioni])
        _y.append(y[selectioni])
        _t.append(pos_times[selectioni])
        _st.append(selectiont)
        _id.append(np.full(len(selectiont), uid))

    x           = np.concatenate(_x)
    y           = np.concatenate(_y)
    pos_times   = np.concatenate(_t)
    spike_times = np.concatenate(_st)
    spike_ids   = np.concatenate(_id)

    idx         = np.argsort(pos_times)
    x           = x[idx]
    y           = y[idx]
    pos_times   = pos_times[idx]
    spike_times = spike_times[idx]
    spike_ids   = spike_ids[idx]

    return spike_times, pos_times, spike_ids, x, y

def clean_gaps(x: np.ndarray, y: np.ndarray, time: np.ndarray, position_gap_threshold_s: float = 0.25):
    """Cleans gaps in position data by removing points that are more than
    position_gap_threshold_s seconds apart from the previous point.
    Gaps are filled with NaN so as to ignore them in place-field and velocity calculations

    Args:
        x (np.ndarray): x position data
        y (np.ndarray): y position data
        time (np.ndarray): Times of the position data
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        x (np.ndarray): Cleaned x position data
        y (np.ndarray): Cleaned y position data
    """
    
    pdiff = np.diff(time)
    gaps  = np.where(pdiff > position_gap_threshold_s)[0]
    if len(gaps) > 0:
        print(f"Removing {len(gaps)} position gaps")
        for i in gaps:
            x[i - 5 : i + 5] = np.nan
            y[i - 5 : i + 5] = np.nan
            print(f"\t(x,y) = ({x[i]}, {y[i]})")
    return x,y

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
    for epoch in range(len(run_starts)):
        start = run_starts[epoch]
        end   = run_ends[epoch]

        spike_window_start = np.searchsorted(spike_times, start, side='left')
        spike_window_end   = np.searchsorted(spike_times, end)
        pos_window_start   = np.searchsorted(time, start, side='left')
        pos_window_end     = np.searchsorted(time, end)

        window_spike_times = spike_times[spike_window_start:spike_window_end]
        window_spike_ids   = spike_ids[spike_window_start:spike_window_end] 
        window_x_pos       = x[pos_window_start:pos_window_end]
        window_y_pos       = y[pos_window_start:pos_window_end]
        
        _spike_times.append(window_spike_times)
        _spike_ids.append(window_spike_ids)
        _run_x_pos.append(window_x_pos)
        _run_y_pos.append(window_y_pos)
    
    _spike_times = np.hstack(_spike_times)
    _spike_ids   = np.hstack(_spike_ids)
    _x = np.hstack(_run_x_pos)
    _y = np.hstack(_run_y_pos)
    return _spike_times, _spike_ids, _x, _y

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
        place_field_raw = np.nan_to_num(spike_hist / position_hist)
    pf_gaussian_sd_bins = hseu.cm_to_bins(place_field_sd_gaussian, bin_size_cm)
    place_field_smoothed = gaussian_filter(
        place_field_raw, sigma=pf_gaussian_sd_bins
    )
    return place_field_smoothed

def __get_spike_positions(
        cell_spike_times: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        pos_times: np.ndarray, 
        position_gap_threshold_s: float = 0.25
    ):

    """Finds the positions of a given set of spike times.

    Args:
        cell_spike_times (np.ndarray): Times of the spikes
        x (np.ndarray): x position data
        y (np.ndarray): y position data
        pos_times (np.ndarray): Time of the position data
        position_gap_threshold_s (float, optional): Threshold in seconds for which spikes are not considered. Defaults to 0.25.

    Returns:
        tuple: x and y positions of the spikes
    """
    # Find the indices where spike_times would be inserted in pos_times
    # This assumes pos_times is sorted (which it usually is for time-series)
    idx = np.searchsorted(pos_times, cell_spike_times)

    # Clip indices to stay within array bounds
    idx_right = np.clip(idx, 0, len(pos_times) - 1)
    idx_left = np.clip(idx - 1, 0, len(pos_times) - 1)

    # Determine which is closer: the index to the left or the right?
    diff_right = np.abs(pos_times[idx_right] - cell_spike_times)
    diff_left = np.abs(pos_times[idx_left] - cell_spike_times)
    
    # Use the index that gives the minimum difference
    closer_to_right = diff_right < diff_left
    final_idx = np.where(closer_to_right, idx_right, idx_left)
    min_diffs = np.where(closer_to_right, diff_right, diff_left)

    if np.any(min_diffs > position_gap_threshold_s):
        over_threshold = min_diffs[min_diffs > position_gap_threshold_s]
        print(f"Warning: {len(over_threshold)} spikes exceeded the gap threshold.")

    return x[final_idx], y[final_idx]

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

    spike_times  = rat_data.raw_data.spike_times
    time         = rat_data.raw_data.time
    spike_ids    = rat_data.raw_data.spike_ids
    x            = rat_data.raw_data.x
    y            = rat_data.raw_data.y

    spike_times, time_aligned, spike_ids, x_aligned, y_aligned = fix_alignment(
        spike_times,
        time,
        spike_ids,
        x,y,0.1
    )
    # spike_times = rat_data.run_data.spike_times
    # spike_ids = rat_data.run_data.spike_ids
    # time_aligned = rat_data.raw_data.time
    # x_aligned = rat_data.run_data.x
    # y_aligned = rat_data.run_data.y

    spike_times, spike_ids, x, y = __get_run_data(
        spike_ids,
        spike_times,
        rat_data.raw_data.x, 
        rat_data.raw_data.y,
        rat_data.raw_data.time,
        rat_data.run_data.run_starts,
        rat_data.run_data.run_ends
    )

    nbinsx = int(hseu.PFEIFFER_ENV_WIDTH_CM / bin_size_cm)
    nbinsy = int(hseu.PFEIFFER_ENV_HEIGHT_CM / bin_size_cm)
    spatial_grid_x = np.linspace(0, hseu.PFEIFFER_ENV_HEIGHT_CM, nbinsx + 1)
    spatial_grid_y = np.linspace(0, hseu.PFEIFFER_ENV_WIDTH_CM, nbinsy + 1)

    position_hist,_,_ = np.histogram2d(
        x, y,
        bins=(spatial_grid_x,spatial_grid_y)
    )
    position_hist = position_hist.T * hseu.PFEIFFER_RECORDING_FPS


    spike_histograms = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    for cell_id in range(rat_data.n_cells):
        cell_spike_times = spike_times[spike_ids == cell_id]
        cell_x,cell_y = __get_spike_positions(
            cell_spike_times,
            x_aligned,
            y_aligned, 
            time_aligned, 
            position_gap_threshold_s
        )

        if len(cell_spike_times) > 0:
            spike_hist, _, _ = np.histogram2d(
                cell_x,
                cell_y,
                bins=(spatial_grid_x, spatial_grid_y),
            )
            spike_histograms[cell_id] = spike_hist.T


    place_fields = np.zeros((rat_data.n_cells, nbinsx, nbinsy))
    for i in range(rat_data.n_cells):
        place_fields[i] = __calculate_one_placefield(
                position_hist,
                spike_histograms[i],
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
        np.argwhere(max_fr_array > hseu.PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC)
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