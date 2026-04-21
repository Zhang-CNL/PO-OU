import numpy as np
import pynapple as nap
from typing import Optional, List, Tuple
import numpy.typing as npt

import hippocampalseq.utils as hseu

def extract_trajectories(run_position_data: nap.TsdFrame, starts: npt.ArrayLike, ends: npt.ArrayLike) -> List[np.ndarray]:
    """Extract ground truth trajectories from position data.
    Args:
        run_position_data (nap.TsdFrame): Position data.
        starts (npt.ArrayLike): List of start times.
        ends (npt.ArrayLike): List of end times.

    Returns:
        (List[np.ndarray]): List of ground-truth trajectories.
    """
    true_trajectories = []
    for start,end in zip(starts,ends):
        run_subset = run_position_data.restrict(nap.IntervalSet(start,end))
        trajectory = run_subset[['x','y']].values
        true_trajectories.append(trajectory)
    return true_trajectories

def select_run_snippets(
        run_position_data: nap.TsdFrame, 
        velocity_cutoff: float = 5.0,
        run_period_threshold: float = 2.0, 
        duration_scaling_factor: float = 2.9 * 6.75
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Select snippets of runs that are at least `run_period_threshold` seconds long.
    Args:
        run_position_data (nap.TsdFrame): Position data.
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        run_period_threshold (float): Minimum run period in seconds. Defaults to 2.0.
        duration_scaling_factor (float): Duration scaling factor. Defaults to 2.9*6.75.

    Returns:
        (np.ndarray): Start times of runs.
        (np.ndarray): End times of runs.
        (List[np.ndarray]): List of ground-truth trajectories.
    """
    mask = run_position_data['velocity'].values >= velocity_cutoff 
    run_starts,run_ends = hseu.extract_times_from_boolean(mask, run_position_data.index.values)
    lengths = run_ends - run_starts
    periods = lengths > run_period_threshold
    starts,ends = run_starts[periods],run_ends[periods]

    true_trajectories = extract_trajectories(run_position_data, starts, ends)
    return starts, ends, true_trajectories

def process_theta(
        run_position_data: nap.TsdFrame,
        run_spikes: nap.TsGroup,
        place_cell_ids: np.ndarray,
        velocity_cutoff: float = 5.0,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 250.0,
        time_window_advance_ms: Optional[float] = None,
    ):
    """Extract theta snippets and corresponding spiking matrices.
    Args:
        run_position_data (nap.TsdFrame): Position data.
        run_spikes (nap.TsGroup): Spikes.
        place_cell_ids (np.ndarray): Place cell ids.
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        run_period_threshold (float): Minimum run period in seconds. Defaults to 2.0.
        place_field_scaling_factor (float): Place field scaling factor. Defaults to 2.9.
        velocity_scaling_factor (float): Velocity scaling factor. Defaults to 6.75.
        time_window_ms (float): Time window in ms. Defaults to 250.0.
        time_window_advance_ms (Optional[float]): Time window advance in ms. Defaults to None.

    Returns:
        (List[np.ndarray]): List of ground-truth trajectories.
        (List[np.ndarray]): List of spiking matrices.
    """
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_ms = time_window_ms
    time_window_advance_s = time_window_advance_ms / 1000
    duration_scaling_factor = velocity_scaling_factor * place_field_scaling_factor

    (
        starts, 
        ends,
        true_trajectories
    ) = select_run_snippets(
        run_position_data, 
        velocity_cutoff,
        run_period_threshold,
        duration_scaling_factor
    )
    ncells = len(run_spikes)
    spikemats = []
    for start,end in zip(starts,ends):
        spikemat = hseu.extract_spikemat(
            run_spikes,
            start,
            end,
            time_window_s,
            time_window_advance_s
        )
        spikemats.append(spikemat[:,place_cell_ids].astype(int))

    return true_trajectories,spikemats
            