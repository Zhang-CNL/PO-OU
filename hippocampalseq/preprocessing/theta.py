import numpy as np 
import hippocampalseq.utils as hseu
from typing import Optional
from .data import *

def __run_snippet_durations(rat_data: RatData, duration_scaling_factor: float):
    ripple_spike_durations = rat_data.ripple_data.popburst_time_s[:,1] - rat_data.ripple_data.popburst_time_s[:,0]
    return ripple_spike_durations * duration_scaling_factor

def __run_period_times(rat_data: RatData, run_period_threshold: float):
    run_lengths = rat_data.run_times_end - rat_data.run_times_start
    run_periods = run_lengths > run_period_threshold
    run_starts = rat_data.run_times_start[run_periods]
    run_ends = rat_data.run_times_end[run_periods]
    return np.vstack((run_starts, run_ends)).T

def __run_snippet_times(run_period_times: np.ndarray, run_snippet_durations: np.ndarray) -> np.ndarray:
    run_snippet_times = np.zeros((len(run_snippet_durations), 2))

    for i, snippet_duration in enumerate(run_snippet_durations):
        sample = run_period_times[:,1] - run_period_times[:,0]
        snippet_times_sample = run_period_times[sample > snippet_duration]
        if len(snippet_times_sample) > 0:
            run_ind_sample = np.random.choice(len(snippet_times_sample))
            run_samples = snippet_times_sample[run_ind_sample]
            snippet_start = np.random.uniform(run_samples[0], run_samples[1] - snippet_duration)
            snippet_end = snippet_start + snippet_duration
            run_snippet_times[i] = [snippet_start, snippet_end]
        else:
            run_snippet_times[i] = [np.nan, np.nan]
    return run_snippet_times[~np.isnan(run_snippet_times).any(axis=1)]

def __trajectories(rat_data: RatData, run_times_start: np.ndarray, run_times_end: np.ndarray) -> dict:
    trajectories = dict()
    for i in range(len(run_times_start)):
        start = np.argwhere(rat_data.pos_times_sec > run_times_start[i])[0,0]
        end = np.argwhere(rat_data.pos_times_sec < run_times_end[i])[-1,0]
        trajectories[i] = np.array(
            [
                rat_data.pos_xy_cm[start:end,0],
                rat_data.pos_xy_cm[start:end,1]
            ]
        ).T
    return trajectories

def __spikemats(
        rat_data: RatData, 
        run_snippet_times: np.ndarray, 
        time_window_s: float,
        time_window_advance_s: float
    ) -> dict:
    spikemats = dict()
    for i in range(len(run_snippet_times)):
        start = run_snippet_times[i,0]
        end = run_snippet_times[i,1]
        spikemats[i] = hseu.extract_spikemat(
            rat_data.spike_ids, 
            rat_data.spike_times_sec, 
            rat_data.place_field_data.place_cell_ids,
            start, 
            end, 
            time_window_s, 
            time_window_advance_s
        )
    return spikemats

def __select_run_snippets(
        rat_data: RatData, 
        run_period_threshold: float,
        duration_scaling_factor: float
    ) -> tuple[np.ndarray, np.ndarray]:
    run_snippet_durations = __run_snippet_durations(rat_data, duration_scaling_factor)
    run_period_times = __run_period_times(rat_data, run_period_threshold)
    run_snippet_times = __run_snippet_times(run_period_times, run_snippet_durations)
    true_trajectories = __trajectories(rat_data, rat_data.run_times_start, rat_data.run_times_end)
    return run_snippet_times, true_trajectories

def process_theta(
        rat_data: RatData,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 3.0,
        time_window_advance_ms: Optional[float] = None,
        seed: int|None = 42
    ) -> Theta:
    np.random.seed(seed)
    time_window_s = time_window_ms / 1000
    time_window_advance_s = time_window_s if time_window_advance_ms is None else time_window_advance_ms / 1000
    duration_scaling_factor = velocity_scaling_factor * place_field_scaling_factor

    place_field_mat = hseu.placefield_matrix(
        rat_data.place_field_data.place_fields, 
        rat_data.place_field_data.place_cell_ids
    )
    run_snippet_times, true_trajectories = __select_run_snippets(
        rat_data, 
        run_period_threshold,
        duration_scaling_factor
    )
    spikemats = __spikemats(
        rat_data, 
        run_snippet_times,
        time_window_s,
        time_window_advance_s
    )
    return Theta(
        run_snippet_times,
        true_trajectories,
        spikemats
    )