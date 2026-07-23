import numpy as np
import pynapple as nap
import warnings

import hippocampalseq.utils as hseu

def extract_trajectories(
        run_position_data: nap.TsdFrame, 
        starts: np.ndarray, 
        ends: np.ndarray
    ) -> list[np.ndarray]:
    """Extract ground truth trajectories from position data.
    Args:
        run_position_data (nap.TsdFrame): Position data.
        starts (npt.ArrayLike): List of start times.
        ends (npt.ArrayLike): List of end times.

    Returns:
        (list[np.ndarray]): List of ground-truth trajectories.
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
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Select snippets of runs that are at least `run_period_threshold` seconds long.
    Args:
        run_position_data (nap.TsdFrame): Position data.
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        run_period_threshold (float): Minimum run period in seconds. Defaults to 2.0.
        duration_scaling_factor (float): Duration scaling factor. Defaults to 2.9*6.75.

    Returns:
        (np.ndarray): Start times of runs.
        (np.ndarray): End times of runs.
        (list[np.ndarray]): List of ground-truth trajectories.
    """
    mask = run_position_data['Velocity'].values >= velocity_cutoff 
    run_starts,run_ends = hseu.extract_times_from_boolean(mask, run_position_data.index.values)
    lengths = run_ends - run_starts
    periods = lengths > run_period_threshold
    starts,ends = run_starts[periods],run_ends[periods]

    true_trajectories = extract_trajectories(run_position_data, starts, ends)
    return starts, ends, true_trajectories

def extract_theta_segments(
        run_position_data: nap.TsdFrame,
        run_spikes: nap.TsGroup,
        place_cell_ids: np.ndarray,
        velocity_cutoff: float = 5.0,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 250.0,
        time_window_advance_ms: float|None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
        time_window_advance_ms (float|None): Time window advance in ms. Defaults to None.

    Returns:
        (list[np.ndarray]): List of ground-truth trajectories.
        (list[np.ndarray]): List of spiking matrices.
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

    spikemats = []
    snippets_to_keep = []
    for i,(start,end) in enumerate(zip(starts,ends)):
        spikemat = hseu.extract_spikemat(
            run_spikes,
            start,
            end,
            time_window_s,
            time_window_advance_s
        )
        spikemat = spikemat[:,place_cell_ids].astype(int)
        # Filter empty spiking matrices.
        if spikemat.shape[0] == 0 or np.sum(spikemat) == 0:
            continue
        spikemats.append(spikemat)
        snippets_to_keep.append(i)

    true_trajectories = [true_trajectories[i] for i in snippets_to_keep]

    return (
        true_trajectories,
        spikemats
    )

def detect_theta_cycles(
        lfp_data: nap.TsdFrame,
        theta_length_s: tuple[float, float]|None = (0.08, 0.16),
        max_cycle_duration_s: float = 6.0
    ) -> tuple[nap.TsdFrame, nap.Ts, np.ndarray]:
    """Filter LFP data for individual theta cycle numbers.

    Args:
        lfp_data (dict): Raw LFP data from `hippocampalseq.io.load_lfp_data`
        theta_length_s (tuple[float, float]|None): Minimum and maximum lengths allowed for a cycle to be used in analysis. If none, no filtering by length is done. Defaults to (0.08,0.16).
        max_cycle_duration (float): Maximum length of time a cycle can be in seconds. Defaults to 6.0.

    ReturnsL
        (nap.TsdFrame): LFP frame synced to the phase of theta.
        (nap.Ts): Time-series representation of the troughs of theta.
        (np.ndarray): Indices of the troughs of theta.
    """
    phase_times = lfp_data.index.values
    phase_deg = np.degrees(lfp_data['Phase Rad'].values) % 360

    # Detect troughs via phase resets (360 -> 0)
    phase_diff = np.diff(phase_deg)
    troughs    = np.where(phase_diff < -345)[0] + 1

    n_samples            = len(phase_deg)
    cycle_duration       = np.zeros(n_samples)
    monotonic_increasing = np.zeros(n_samples)
    cycle_id             = np.zeros(n_samples)

    n_valid,n_skipped_boundary,n_skipped_length = 0,0,0

    if len(troughs) >= 2:
        starts = troughs[:-1]
        ends   = troughs[1:]
        durations = phase_times[ends] - phase_times[starts]

        boundary_mask = durations > max_cycle_duration_s
        n_skipped_boundary = np.sum(boundary_mask)

        if theta_length_s is not None:
            length_mask = ~boundary_mask \
                & ((durations < theta_length_s[0]) 
                    | (durations > theta_length_s[1])
                )
        else:
            length_mask = np.zeros_like(durations)

        n_skipped_length = np.sum(length_mask)

        valid_mask = ~boundary_mask & ~length_mask
        n_valid = np.sum(valid_mask)

        starts = starts[valid_mask]
        ends   = ends[valid_mask]
        durations = durations[valid_mask]

        for cid, (start,end,duration) in enumerate(zip(starts,ends,durations),start=1):
            is_monotonic = np.all(phase_diff[start:end-1] > 0)
            cycle_duration[start:end]       = duration
            monotonic_increasing[start:end] = is_monotonic.astype(int)
            cycle_id[start:end]             = cid


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
            'Filtered LFP', 'Amplitude', 'Power',
            'Raw LFP', 'Phase Rad', 'Phase Deg',
            'Cycle Duration', 'Monotonic Increasing', 'Cycle ID'
        ],
        time_units='s'
    )
    warnings.warn(f"{n_skipped_boundary}/{n_valid} theta cycles boundary-rejected")
    warnings.warn(f"{n_skipped_length}/{n_valid} theta cycles length filtered")
    trough_times = nap.Ts(t=phase_times[troughs], time_units='s')
    return (
        lfp_cycles, 
        trough_times,
        troughs
    )

            