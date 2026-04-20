import numpy as np 
from typing import Optional
from .placefields import fix_alignment
import hippocampalseq.utils as hseu

def extract_spikemats(
        rat_data: hseu.RatData, 
        starts: np.ndarray, 
        ends: np.ndarray,
        time_window_s: float,
        time_window_advance_s: float
    ) -> dict:
    spikemats = []

    spike_times, time_aligned, spike_ids, x_aligned, y_aligned = fix_alignment(
        rat_data.raw_data.spike_times,
        rat_data.raw_data.time,
        rat_data.raw_data.spike_ids,
        rat_data.raw_data.x,
        rat_data.raw_data.y,
        0.1
    )

    for start,end in zip(starts,ends):
        spikemat = hseu.extract_spikemat(
            spike_ids, 
            spike_times, 
            rat_data.place_field_data.place_cell_ids,
            start, 
            end, 
            time_window_s, 
            time_window_advance_s
        )
        spikemats.append(spikemat)
    return spikemats

def extract_trajectories(rat_data: hseu.RatData, run_starts: np.ndarray, run_ends: np.ndarray):
    trajectories = []
    for rstart,rend in zip(run_starts,run_ends):
        tslice = hseu.restrict_indices(rat_data.raw_data.time, rstart, rend)
        trajectory = np.array([
            rat_data.raw_data.x[tslice],
            rat_data.raw_data.y[tslice]
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
    return starts, ends, true_trajectories

def process_theta(
        rat_data: hseu.RatData,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 250.0,
        time_window_advance_ms: Optional[float] = None,
    ) -> hseu.Theta:
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_s = time_window_s
    else:
        time_window_advance_s = time_window_advance_ms / 1000
    duration_scaling_factor = velocity_scaling_factor * place_field_scaling_factor

    starts, ends, true_trajectories = select_run_snippets(
        rat_data, 
        run_period_threshold,
        duration_scaling_factor
    )
    spikemats = extract_spikemats(
        rat_data, 
        starts,
        ends,
        time_window_s,
        time_window_advance_s
    )
    return hseu.Theta(
        starts,
        ends,
        true_trajectories,
        spikemats
    )