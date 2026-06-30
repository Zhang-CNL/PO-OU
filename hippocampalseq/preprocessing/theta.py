import numpy as np
import pynapple as nap
import warnings
import numpy.typing as npt
from typing import Optional, List, Tuple

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
    mask = run_position_data['Velocity'].values >= velocity_cutoff 
    run_starts,run_ends = hseu.extract_times_from_boolean(mask, run_position_data.index.values)
    lengths = run_ends - run_starts
    periods = lengths > run_period_threshold
    starts,ends = run_starts[periods],run_ends[periods]

    true_trajectories = extract_trajectories(run_position_data, starts, ends)
    return starts, ends, true_trajectories

def detect_theta_cycles(
        lfp_data: dict,
        limit_analysis_by_theta_length: bool = True,
        theta_length_s: Tuple[float, float] = (0.08, 0.16),
        max_cycle_duration_s: float = 1.0
    ) -> Tuple[nap.TsdFrame, np.ndarray, np.ndarray]:
    """Filter LFP data for individual theta cycle numbers.

    Args:
        lfp_data (dict): Raw LFP data from `hippocampalseq.io.load_lfp_data`
        limit_analysis_by_theta_length (bool): Use `theta_length_s` to filter out LFP segments that are too short. Defaults to True. 
        theta_length_s (Tuple[float, float]): Minimum and maximum lengths allowed for a cycle to be used in analysis. Defaults to (0.08,0.16).
        max_cycle_duration (float): Maximum length of time a cycle can be 
    """
    phase_times = lfp_data.index.values
    phase_deg = np.degrees(lfp_data['Phase Rad'].values) % 360

    # Detect troughs via phase resets (360 -> 0)
    phase_diff = np.diff(phase_deg)
    troughs = np.where(phase_diff < -345)[0] + 1

    n_samples = len(phase_deg)
    cycle_duration = np.zeros(n_samples)
    monotonic_increasing = np.zeros(n_samples)
    cycle_id = np.zeros(n_samples)

    n_valid            = 0
    n_skipped_boundary = 0
    n_skipped_length   = 0

    for i in range(len(troughs) - 1):
        start = troughs[i]
        end   = troughs[i+1]
        duration = phase_times[end] - phase_times[start]

        # TODO: This seems superfluous given the next if-statement
        if duration > max_cycle_duration_s:
            n_skppied_boundary += 1 
            continue

        if limit_analysis_by_theta_length and (duration < theta_length_s[0] or duration > theta_length_s[1]):
            n_skipped_length += 1
            continue 
        
        cycle_phases = phase_deg[start:end] 
        is_monotonic = np.all(np.diff(cycle_phases) > 0)

        n_valid += 1
        cycle_duration[start:end] = duration 
        monotonic_increasing[start:end] = is_monotonic.astype(int)
        cycle_id[start:end] = n_valid

    lfp_cycles = nap.TsdFrame(
        t=phase_times,
        d=np.c_[
            lfp_data['Filtered LFP'].values,
            lfp_data['Amplitude'].values,
            lfp_data['Power'].values,
            lfp_data['Raw LFP'].values,
            lfp_data['Phase Rad'].values,
            phase_deg,
            cycle_duration,
            monotonic_increasing,
            cycle_id
        ],
        columns=[
            'Filtered LFP', 'Amplitude', 'Power'
            'Raw LFP', 'Phase Rad', 'Phase Deg',
            'Cycle Duration', 'Monotonic', 'Cycle ID'
        ],
        time_units='s'
    )
    warnings.warn(f"{n_skipped_boundary}/{n_valid} theta cycles boundary-rejected")
    warnings.warn(f"{n_skipped_length}/{n_valid} theta cycles length filtered")
    trough_times = nap.Ts(t=phase_times[troughs], time_units='s')
    return (
        lfp_cycles, 
        troughs_indices,
        trough_times
    )

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
            