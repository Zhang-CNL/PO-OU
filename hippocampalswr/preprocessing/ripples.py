import numpy as np 
import scipy.stats as sp
import hippocampalswr.utils as hswu
from typing import Optional
from .data import *

def __get_spikemat(
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        place_cell_ids: np.ndarray,
        start_time: float,
        end_time: float,
        time_window_s: float,
        time_window_advance_s: float,
    ) -> np.ndarray:
    spikemat = np.empty(shape=(0, len(place_cell_ids)), dtype=int)
    timebin_start_time = start_time
    timebin_end_time = start_time + time_window_s
    while timebin_end_time < end_time:
        spikes_after_start = spike_times >= timebin_start_time
        spikes_before_end = spike_times < timebin_end_time
        timebin_bool = spikes_after_start == spikes_before_end
        spike_ids_in_window = spike_ids[timebin_bool]
        spikevector = np.array(
            [[np.sum(spike_ids_in_window == cell_id) for cell_id in place_cell_ids]]
        )
        spikemat = np.append(spikemat, spikevector, axis=0)
        timebin_start_time = timebin_start_time + time_window_advance_s
        timebin_end_time = timebin_end_time + time_window_advance_s
    return np.array(spikemat, dtype=int)

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
    spikes_per_timebin = spikemat_fullripple.sum(axis=1)
    avg_spikes_per_s = (
        spikes_per_timebin / n_place_cells / time_window_s
    )
    avg_spikes_per_s_smoothed = np.convolve(
        avg_spikes_per_s, avg_fr_smoothing_convolution, mode="same"
    )
    timebins_above_threshold = (
        avg_spikes_per_s_smoothed > avg_spikes_per_s_threshold
    )
    if (timebins_above_threshold).sum() > 1:
        start_timebin = np.argwhere(timebins_above_threshold)[0][0]
        end_timebin = np.argwhere(timebins_above_threshold)[-1][0]
        if (end_timebin - start_timebin) >= min_popburst_n_time_windows:
            spikemat_popburst = spikemat_fullripple[start_timebin:end_timebin]
            spikemat_popburst_start = (
                ripple_start + start_timebin * time_window_s
            )
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
        rat_data: RatData, 
        time_window_s: float, 
        time_window_advance_s: float,
        avg_fr_smoothing_convolution: np.ndarray,
        avg_spikes_per_s_threshold: int,
        min_popburst_n_time_windows: int
    ) -> tuple:
    spike_ids      = rat_data.spike_ids
    spike_times    = rat_data.spike_times_sec
    place_cell_ids = rat_data.place_field_data.place_cell_ids
    place_fields   = rat_data.place_field_data.place_fields
    ripple_times   = rat_data.ripple_times_sec
    n_ripples      = rat_data.n_ripples

    spikemats_fullripple = dict()
    spikemats_popburst = dict()
    avg_spikes_per_s_smoothed = dict()
    spikemat_times = np.zeros((n_ripples, 2))
    for ripple_num in range(n_ripples):
        ripple_start = ripple_times[ripple_num][0]
        ripple_end = ripple_times[ripple_num][1]
        spikemats_fullripple[ripple_num] = __get_spikemat(
            spike_ids,
            spike_times,
            place_cell_ids,
            ripple_start,
            ripple_end,
            time_window_s,
            time_window_advance_s,
        )
        (
            spikemats_popburst[ripple_num],
            spikemat_times[ripple_num],
            avg_spikes_per_s_smoothed[ripple_num],
        ) = __select_population_burst(
                spikemats_fullripple[ripple_num], 
                ripple_start, 
                ripple_end,
                rat_data.place_field_data.n_place_cells,
                time_window_s,
                avg_fr_smoothing_convolution,
                avg_spikes_per_s_threshold,
                min_popburst_n_time_windows
            )
    return (
        spikemats_fullripple,
        spikemats_popburst,
        spikemat_times,
        avg_spikes_per_s_smoothed
    )

def __calc_popburst_firing_rate_array(spikemats: dict, n_place_cells: int, time_window_s: float) -> np.ndarray:
    firing_rate_matrix = np.full((n_place_cells, len(spikemats)), np.nan)
    total_spikes = np.zeros(n_place_cells)
    total_time = 0
    for i in range(len(spikemats)):
        if spikemats[i] is not None:
            total_spikes += spikemats[i].sum(axis=0)
            total_time += spikemats[i].shape[0] * time_window_s
            firing_rate_matrix[:, i] = total_spikes / total_time
    firing_rate_array = total_spikes / total_time
    return firing_rate_array, firing_rate_matrix

def __calc_firing_rate_scaling(run_mean_frs: np.ndarray, ripple_mean_frs: np.ndarray) -> dict:
        scaling_factors = ripple_mean_frs / run_mean_frs
        scaling_factors = scaling_factors[scaling_factors > 0]
        k, _, scale = sp.gamma.fit(scaling_factors, floc=0)
        return {"scaling_factors": scaling_factors, "alpha": k, "beta": 1 / scale}
        #return {"scaling_factors": scaling_factors, "alpha": 1, "beta": 1}

def process_ripples(
        rat_data: RatData, 
        time_window_ms: float = 3.0, 
        time_window_advance_ms: Optional[float] = None,
        avg_fr_smoothing_convolution: np.ndarray = np.array([.25, .25, .25, .25]),
        avg_spikes_per_s_threshold: int = 2,
        min_popburst_duration_ms: int = 30
    ) -> RippleData:
    time_window_s = time_window_ms / 1000
    time_window_advance_s = time_window_s if time_window_advance_ms is None else time_window_advance_ms / 1000
    min_popburst_n_time_windows = int(np.ceil(min_popburst_duration_ms / time_window_ms))

    place_field_mat = hswu.placefield_matrix(
            rat_data.place_field_data.place_fields,
            rat_data.place_field_data.place_cell_ids
        )

    fullripple,popburst,times,avg = __calc_spikemats(
            rat_data, 
            time_window_s, 
            time_window_advance_s,
            avg_fr_smoothing_convolution,
            avg_spikes_per_s_threshold,
            min_popburst_n_time_windows
        )
    firing_rate_array, firing_rate_matrix = __calc_popburst_firing_rate_array(
            popburst,
            rat_data.place_field_data.n_place_cells,
            time_window_s
        )
    firing_rate_scale = __calc_firing_rate_scaling(
            rat_data.place_field_data.mean_firing_rate[rat_data.place_field_data.place_cell_ids],
            firing_rate_array
        )

    ripple_info = RippleData(
            spikemats_ripple   = fullripple,
            spikemats_popburst = popburst,
            popburst_time_s    = times,
            avg_sps_smoothed   = avg,
            mean_popburst_arr  = firing_rate_array,
            mean_popburst_mat  = firing_rate_matrix,
            firing_rate_scale  = firing_rate_scale
        )
    return ripple_info