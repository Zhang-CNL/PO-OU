import numpy as np 
from typing import Optional
import hippocampalseq.utils as hseu

def __run_snippet_durations(rat_data: hseu.RatData, duration_scaling_factor: float):
    ripple_spike_durations = rat_data.ripple_data.popburst_time_s[:,1] - rat_data.ripple_data.popburst_time_s[:,0]
    return ripple_spike_durations * duration_scaling_factor

def __run_period_times(rat_data: hseu.RatData, run_period_threshold: float):
    run_lengths = rat_data.raw_data.run_ends - rat_data.raw_data.run_starts
    run_periods = run_lengths > run_period_threshold
    run_starts  = rat_data.raw_data.run_starts[run_periods]
    run_ends    = rat_data.raw_data.run_ends[run_periods]
    return np.vstack((run_starts, run_ends)).T

def __run_snippet_times(run_period_times: np.ndarray, run_snippet_durations: np.ndarray) -> np.ndarray:
    run_snippet_times = np.full((len(run_snippet_durations), 2), np.nan)
    sample = run_period_times[:,1] - run_period_times[:,0]
    mask = run_snippet_durations[:, None] <= sample

    valid = np.any(mask, axis=1)

    for i in np.where(valid)[0]:
        indices = np.where(mask[i])[0]
        run_ind_sample = np.random.choice(indices)
        period = run_period_times[run_ind_sample]
        duration = run_snippet_durations[i]
        start = np.random.uniform(period[0], period[1] - duration)
        end   = start + duration
        run_snippet_times[i] = [start, end]

    return run_snippet_times[~np.isnan(run_snippet_times).any(axis=1)]

def __spikemats(
        rat_data: hseu.RatData, 
        run_snippet_times: np.ndarray, 
        time_window_s: float,
        time_window_advance_s: float
    ) -> dict:
    spikemats = []
    for i in range(len(run_snippet_times)):
        start = run_snippet_times[i,0]
        end = run_snippet_times[i,1]
        spikemat = hseu.extract_spikemat(
            rat_data.raw_data.spike_ids, 
            rat_data.raw_data.spike_times, 
            rat_data.place_field_data.place_cell_ids,
            start, 
            end, 
            time_window_s, 
            time_window_advance_s
        )
        spikemats.append(spikemat)
    return spikemats

def __select_run_snippets(
        rat_data: hseu.RatData, 
        run_period_threshold: float,
        duration_scaling_factor: float
    ) -> tuple[np.ndarray, np.ndarray]:
    run_snippet_durations = __run_snippet_durations(rat_data, duration_scaling_factor)
    run_period_times      = __run_period_times(rat_data, run_period_threshold)
    run_snippet_times     = __run_snippet_times(run_period_times, run_snippet_durations)
    true_trajectories     = extract_trajectories(rat_data, rat_data.raw_data.run_starts, rat_data.raw_data.run_ends)
    return run_snippet_times, true_trajectories

def extract_trajectories(rat_data: hseu.RatData, run_starts: np.ndarray, run_ends: np.ndarray):
    trajectories = []
    for i in range(len(run_starts)):
        start = np.searchsorted(rat_data.raw_data.time, run_starts[i], side='left')
        end = np.searchsorted(rat_data.raw_data.time, run_ends[i])
        trajectory = np.array([
            rat_data.raw_data.x[start:end-1],
            rat_data.raw_data.y[start:end-1]
        ]).T 
        trajectories.append(trajectory)
    return trajectories

def select_run_snippets(
        rat_data: hseu.RatData, 
        run_period_threshold: float = 2.0,
        duration_scaling_factor: float = 2.9 * 6.75,
    ): 
    starts,ends = rat_data.raw_data.run_starts, rat_data.raw_data.run_ends
    lengths = ends - starts
    periods = lengths > run_period_threshold
    starts,ends = starts[periods],ends[periods]

    true_trajectories = extract_trajectories(rat_data, starts, ends)
    run_snippet_times = np.vstack((starts,ends)).T
    return run_snippet_times, true_trajectories

def process_theta(
        rat_data: hseu.RatData,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 3.0,
        time_window_advance_ms: Optional[float] = None,
        seed: int|None = 42
    ) -> hseu.Theta:
    np.random.seed(seed)
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_s = time_window_s
    else:
        time_window_advance_s = time_window_advance_ms / 1000
    duration_scaling_factor = velocity_scaling_factor * place_field_scaling_factor

    run_snippet_times, true_trajectories = select_run_snippets(
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
    theta_spikes = [hseu.ThetaSpikes(trajectory, spikemat) for trajectory,spikemat in zip(true_trajectories, spikemats)]
    return hseu.Theta(
        run_snippet_times,
        theta_spikes
    )