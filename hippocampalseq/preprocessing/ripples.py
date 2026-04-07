import numpy as np 
import scipy.stats as sp
from typing import Optional
import hippocampalseq.utils as hseu

def __select_population_burst(
        spikemat_fullripple: np.ndarray, 
        ripple_start: float,
        ripple_end: float,
        n_place_cells: int,
        time_window_s: float,
        avg_fr_smoothing_convolution: np.ndarray,
        avg_spikes_per_s_threshold: float,
        min_popburst_n_time_windows: int
    ) -> tuple:
    """
    Selects a population burst from the given spike data.

    A population burst is defined as a sequence of timebins where the average firing rate of the population
    is above the given threshold for at least min_popburst_n_time_windows timebins.

    Args:
        spikemat_fullripple (np.ndarray): Spike matrix of a full ripple
        ripple_start (float): Start time of the ripple
        ripple_end (float): End time of the ripple
        n_place_cells (int): Number of place cells
        time_window_s (float): Length of time window in seconds
        avg_fr_smoothing_convolution (np.ndarray): Smoothing convolution for average firing rate
        avg_spikes_per_s_threshold (float): Threshold for average firing rate in spikes per second
        min_popburst_n_time_windows (int): Minimum number of time windows for a population burst

    Returns:
        tuple: A tuple containing the spike matrix of the population burst, the start and end times of the population burst,
            and the smoothed average firing rate of the population.

    """
    spikes_per_timebin = spikemat_fullripple.sum(axis=1)
    avg_spikes_per_s_smoothed = np.convolve(
        spikes_per_timebin / (n_place_cells * time_window_s), avg_fr_smoothing_convolution, mode="same"
    )
    timebins_above_threshold = avg_spikes_per_s_smoothed > avg_spikes_per_s_threshold
    if timebins_above_threshold.sum() > 1:
        start_timebin = np.argwhere(timebins_above_threshold)[0,0]
        end_timebin   = np.argwhere(timebins_above_threshold)[-1,0]
        if (end_timebin - start_timebin) >= min_popburst_n_time_windows:
            spikemat_popburst = spikemat_fullripple[start_timebin:end_timebin]
            spikemat_popburst_start = ripple_start + start_timebin * time_window_s
            spikemat_popburst_end = (
                ripple_end
                - (spikemat_fullripple.shape[0] - end_timebin)
                * time_window_s
            )
        else:
            spikemat_popburst = None
            spikemat_popburst_start, spikemat_popburst_end = [np.nan, np.nan]
    else:
        spikemat_popburst = None
        spikemat_popburst_start, spikemat_popburst_end = [np.nan, np.nan]
    return (
        spikemat_popburst,
        [spikemat_popburst_start, spikemat_popburst_end],
        avg_spikes_per_s_smoothed,
    )

def __calc_spikemats(
        rat_data: hseu.RatData, 
        time_window_s: float, 
        time_window_advance_s: float,
        avg_fr_smoothing_convolution: np.ndarray,
        avg_spikes_per_s_threshold: int,
        min_popburst_n_time_windows: int
    ) -> tuple:
    """
    Calculates the spike matrices for the full ripple and population burst
    for all ripples in the given rat data.

    Args:
        rat_data (hseu.RatData): Data of the rat
        time_window_s (float): Length of time window in seconds
        time_window_advance_s (float): Advance of time window in seconds
        avg_fr_smoothing_convolution (np.ndarray): Convolution to smooth the average firing rate
        avg_spikes_per_s_threshold (int): Threshold for the average firing rate
        min_popburst_n_time_windows (int): Minimum number of time windows for a population burst

    Returns:
        tuple: Contains the spike matrices for the full ripple and population burst,
               the start and end times of the population bursts, and the smoothed average firing rate
    """
    spike_ids      = rat_data.raw_data.spike_ids
    spike_times    = rat_data.raw_data.spike_times
    place_cell_ids = rat_data.place_field_data.place_cell_ids
    place_fields   = rat_data.place_field_data.place_fields
    ripple_times   = rat_data.raw_data.raw_ripples[:,:2]
    n_ripples      = rat_data.n_ripples

    spike_ripples = []
    spikemat_times = np.zeros((n_ripples, 2))
    for i in range(n_ripples):
        ripple_start = ripple_times[i,0]
        ripple_end   = ripple_times[i,1]
        spikemats_fullripple = hseu.extract_spikemat(
            spike_ids,
            spike_times,
            place_cell_ids,
            ripple_start,
            ripple_end,
            time_window_s,
            time_window_advance_s,
        )
        (
            spikemats_popburst,
            spikemat_times[i],
            avg_spikes_per_s_smoothed,
        ) = __select_population_burst(
            spikemats_fullripple, 
            ripple_start, 
            ripple_end,
            rat_data.place_field_data.n_place_cells,
            time_window_s,
            avg_fr_smoothing_convolution,
            avg_spikes_per_s_threshold,
            min_popburst_n_time_windows
        )

        spike_ripple = hseu.RippleSpike(
            spikemats_fullripple,
            spikemats_popburst,
            avg_spikes_per_s_smoothed
        )
        spike_ripples.append(spike_ripple)

    return (spike_ripples, spikemat_times)

def __calc_popburst_firing_rate_array(spike_data, n_place_cells: int, time_window_s: float) -> np.ndarray:
    """
    Calculate the firing rate of the population bursts.

    The firing rate is calculated as the total number of spikes divided by the total time of the population bursts.

    Args:
        spikemats (dict): Spikemats of the population bursts
        n_place_cells (int): Number of place cells
        time_window_s (float): Time window in seconds

    Returns:
        np.ndarray: Firing rate of the population bursts (spikes per second)
    """
    firing_rate_matrix = np.full((n_place_cells, len(spike_data)), np.nan)
    total_spikes = np.zeros(n_place_cells)
    total_time = 0
    for i in range(len(spike_data)):
        popburst = spike_data[i].spikemat_popburst
        if popburst is not None:
            total_spikes += popburst.sum(axis=0)
            total_time   += popburst.shape[0] * time_window_s
            firing_rate_matrix[:, i] = total_spikes / total_time
    firing_rate_array = total_spikes / total_time
    return firing_rate_array, firing_rate_matrix

def __calc_firing_rate_scaling(run_mean_frs: np.ndarray, ripple_mean_frs: np.ndarray) -> dict:
    """
    Calculate the scaling factors for the firing rate of the ripples.

    The scaling factors are calculated as the ratio of the ripple mean firing rates to the run mean firing rates.
    The gamma distribution is then fitted to the scaling factors to obtain the alpha and beta parameters.

    Args:
        run_mean_frs (np.ndarray): Mean firing rates of the runs.
        ripple_mean_frs (np.ndarray): Mean firing rates of the ripples.

    Returns:
        dict: A dictionary containing the scaling factors, alpha and beta parameters.
    """
    scaling_factors = ripple_mean_frs / run_mean_frs
    scaling_factors = scaling_factors[scaling_factors > 0]
    k, _, scale = sp.gamma.fit(scaling_factors, floc=0)
    return {"scaling_factors": scaling_factors, "alpha": k, "beta": 1 / scale}
    #return {"scaling_factors": scaling_factors, "alpha": 1, "beta": 1}

def process_ripples(
        rat_data: hseu.RatData, 
        time_window_ms: float = 3.0, 
        time_window_advance_ms: Optional[float] = None,
        avg_fr_smoothing_convolution: np.ndarray = np.array([.25, .25, .25, .25]),
        avg_spikes_per_s_threshold: int = 2,
        min_popburst_duration_ms: int = 30
    ) -> hseu.RippleData:
    """
    Process ripple data.

    Process the ripple data by calculating the spikemats for the ripples and population bursts,
    and then calculating the firing rate of the population bursts and the scaling factors.

    Args:
        rat_data (hseu.RatData): Data of the rat
        time_window_ms (float): Time window in milliseconds. Defaults to 3.0.
        time_window_advance_ms (Optional[float]): Time window advance in milliseconds. Defaults to None.
        avg_fr_smoothing_convolution (np.ndarray): Smoothing convolution for average firing rate. Defaults to np.array([.25, .25, .25, .25]).
        avg_spikes_per_s_threshold (int): Threshold for average firing rate in spikes per second. Defaults to 2.
        min_popburst_duration_ms (int): Minimum duration of population bursts in milliseconds. Defaults to 30.

    Returns:
        hseu.RippleData: Dictionary containing ripple data with additional fields for population burst firing rate and scaling factors.
    """
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_s = time_window_s
    else:
        time_window_advance_s = time_window_advance_ms / 1000
    min_popburst_n_time_windows = int(np.ceil(min_popburst_duration_ms / time_window_ms))

    spike_data, spike_times = __calc_spikemats(
        rat_data, 
        time_window_s, 
        time_window_advance_s,
        avg_fr_smoothing_convolution,
        avg_spikes_per_s_threshold,
        min_popburst_n_time_windows
    )
    firing_rate_array, firing_rate_matrix = __calc_popburst_firing_rate_array(
        spike_data,
        rat_data.place_field_data.n_place_cells,
        time_window_s
    )
    firing_rate_scale = __calc_firing_rate_scaling(
        rat_data.place_field_data.mean_firing_rate[rat_data.place_field_data.place_cell_ids],
        firing_rate_array
    )

    ripple_info = hseu.RippleData(
        popburst_time_s    = spike_times,
        mean_popburst_arr  = firing_rate_array,
        mean_popburst_mat  = firing_rate_matrix,
        firing_rate_scale  = firing_rate_scale,
        ripple_spikes      = spike_data
    )
    return ripple_info