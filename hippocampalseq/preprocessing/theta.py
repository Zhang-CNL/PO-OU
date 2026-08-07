import numpy as np
import pynapple as nap
import warnings
from dataclasses import dataclass

import hippocampalseq.utils as hseu

@dataclass
class BimodalPhaseWindows:
    forward: tuple = (250, 60)
    reverse: tuple = (80, 230)
    major_peak: tuple = (200, 70)
    minor_peak: tuple = (80, 190)

def filter_nonmonotonic_runs(runs: list[nap.TsdFrame], theta_starting_phase: float=70.0) -> list[int]:
    """Filter out runs whose phase is non-monotonic.

    Args:
        runs (list[nap.TsdFrame]): List of segmented runs to filter. 
        theta_starting_phase (float, optional): Starting phase of theta sequence. Defaults to 70.0.

    Returns:
        (list[int]): List of indices of runs that are monotonic.
    """
    monotonic_run_idx = []
    for i,run in enumerate(runs):
        phase = run['Phase Deg'].values.copy()
        osc_nums = run['Oscillation Number'].values
        u_osc = np.unique(osc_nums)
        monotonic = True
        for i,osc in enumerate(u_osc):
            segment = phase[osc_nums == osc]
            segment[segment <= theta_starting_phase] += (i+1) * 360
            if not np.all(np.diff(segment) > 0):
                monotonic = False
                break
        if monotonic:
            monotonic_run_idx.append(i)
    return monotonic_run_idx

def extract_theta_segments(
        running_position: nap.TsdFrame,
        spike_data: nap.TsGroup,
        lfp_data: nap.TsdFrame,
        place_cell_ids: np.ndarray,
        time_window_s: float,
        time_window_advance_s: float|None = None,
        bimodal_windows: BimodalPhaseWindows = BimodalPhaseWindows(),
        theta_length_s: tuple[float,float] = (0.08, 0.16),
        velocity_cutoff: float = 5.0,
        run_period_threshold: float = 2.0
    ) -> tuple[list[nap.TsdFrame], list[np.ndarray]]:
    """Align theta segments to the LFP data and phase. 
    Numbers the oscillations, extracts trajectories for runs, and spikemats.

    Args:
        running_position (nap.TsdFrame): Dataframe containing coordinates, velocity, and head direction for run intervals.
        spike_data (nap.TsGroup): Group of time-series data for spikes during run periods.
        lfp_data (nap.TsdFrame): LFP data frame with ['Phase Deg', 'Power'] fields.
        place_cell_ids (np.ndarray): IDs of cells that are place cells.
        time_window_s (float): 
        time_window_advance_s (float|None):
        velocity_cutoff (float): Minimum velocity to be counted for a run segment.
        run_period_threshold (float): Minimum length in seconds to be included as a run segment.

    Returns:
        (list[nap.TsdFrame]): List of run segments with true positions and LFP data aligned.
        (list[np.ndarray]): List of (T,N) spike matrices.
    """
    if time_window_advance_s is None:
        time_window_advance_s = time_window_s
    theta_starting_phase = 70.0

    run_mask = running_position['Velocity'].values >= velocity_cutoff
    run_starts,run_ends = hseu.extract_times_from_boolean(
        run_mask, 
        running_position.index.values
    )
    run_mask = (run_ends - run_starts) >= run_period_threshold
    run_starts,run_ends = run_starts[run_mask],run_ends[run_mask]
    
    pos_t = running_position.index.values
    lfp_t = lfp_data.index.values
    phase = lfp_data['Phase Deg'].values
    power = lfp_data['Power']

    tsdframes = []
    spikemats = []

    oscillation_number = 0

    for start,end in zip(run_starts, run_ends):
        if (end - start) < time_window_s:
            continue

        decoding_times = np.arange(
            start + time_window_s / 2,
            end - time_window_s / 2 + 1e-12,
            time_window_advance_s
        )
        if decoding_times.size == 0:
            continue

        spikemat = hseu.extract_spikemat(
            spike_data,
            start,
            end,
            time_window_s,
            time_window_advance_s
        )
        spikemat = spikemat[:,place_cell_ids].astype(int)
        
        if spikemat.size == 0 or np.sum(spikemat) == 0:
            continue

        # Match decoding windows to nearest samples
        pos_idx = np.searchsorted(pos_t, decoding_times)
        pos_idx = np.clip(pos_idx, 1, len(pos_t) - 1)
        left = pos_idx - 1
        pos_idx -= (
            np.abs(decoding_times - pos_t[left])
            < np.abs(pos_t[pos_idx] - decoding_times)
        )

        lfp_idx = np.searchsorted(lfp_t, decoding_times)
        lfp_idx = np.clip(lfp_idx, 1, len(lfp_t) - 1)
        left = lfp_idx - 1
        lfp_idx -= (
            np.abs(decoding_times - lfp_t[left])
            < np.abs(lfp_t[lfp_idx] - decoding_times)
        )

        # Number the theta oscillations
        phase_segment = phase[lfp_idx].copy()
        crossings = (
            (phase_segment[:-1] < theta_starting_phase)
            & (phase_segment[1:] >= theta_starting_phase)
        )
        osc = np.zeros(decoding_times.size, dtype=int)
        crossing_idx = np.flatnonzero(crossings) + 1

        last = 0
        for c in crossing_idx:
            osc[last:c] = oscillation_number
            oscillation_number += 1
            last = c

        osc[last:] = oscillation_number
        oscillation_number += 1

        # Check monotonicity of the phase segment
        # last = 0
        # monotonic = True
        # for c in crossing_idx:
        #     segment = phase_segment[last:c]
        #     segment[segment <= theta_starting_phase] += 360
        #     monotonic = np.all(np.diff(segment) >= 0)
        #     if not monotonic:
        #         break
        #     last = c
        
        #phase_segment[phase_segment <= theta_starting_phase] += 360
        #monotonic = np.all(np.diff(phase_segment) >= 0)

        #if not monotonic:
        #    continue

        df = nap.TsdFrame(
            t=decoding_times,
            d=np.c_[
                running_position['x'].values[pos_idx],
                running_position['y'].values[pos_idx],
                running_position['Head direction'].values[pos_idx],
                running_position['Velocity'].values[pos_idx],
                phase_segment,
                power[lfp_idx],
                osc
            ],
            columns=[
                'x', 'y', 'Head direction',
                'Velocity', 'Phase Deg', 'Power', 
                'Oscillation Number'
            ]
        )

        tsdframes.append(df)
        spikemats.append(spikemat)

    return tsdframes, spikemats  

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

            