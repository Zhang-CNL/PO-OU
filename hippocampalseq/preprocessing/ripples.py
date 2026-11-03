import numpy as np
import pynapple as nap

import hippocampalseq.utils as hseu

def extract_ripple_segments(
        position_data: nap.TsdFrame,
        lfp_data: nap.TsdFrame,
        ripple_intervals: nap.IntervalSet,
        spiking_data: nap.TsGroup,
        place_cell_ids: np.ndarray,
        time_window_s: float = 5.0 / 1000,
        time_window_advance_s: float|None = None
    ):
    """Extract ripple spiking matrices aligned to the 
    position and LFP data for each ripple event.

    Args:
        position_data (nap.TsdFrame): Dataframe containing coordinates, velocity, and head direction for run intervals.
        lfp_data (nap.TsdFrame): LFP data frame with ['Phase Deg', 'Power'] fields.
        ripple_intervals (nap.IntervalSet): Intervals of ripple events.
        spiking_data (nap.TsGroup): Group of time-series data for spikes during run periods.
        place_cell_ids (np.ndarray): Indices of place cells.
        time_window_s (float): Bin size in seconds for the spiking matrices.
        time_window_advance_s (float|None): Overlap between bins in seconds for the spiking matrices. If None, defaults to time_window_s.

    Returns:
        (list[nap.TsdFrame]): List of ripple segments with true positions and LFP data aligned.
        (list[np.ndarray]): List of (T,N) spike matrices for each ripple
    """
    if time_window_advance_s is None:
        time_window_advance_s = time_window_s

    pos_t = position_data.index.values
    lfp_t = lfp_data.index.values
    phase = lfp_data['Phase Deg'].values
    power = lfp_data['Power'].values
    
    starts    = ripple_intervals.start
    ends      = ripple_intervals.end
    tsdframes = []
    spikemats = []

    for start,end in zip(starts,ends):
        if end - start < time_window_s:
            continue
        decoding_times = np.arange(
            start + time_window_s / 2,
            end - time_window_s / 2, #+ 1e-12,
            time_window_advance_s
        )
        if decoding_times.size == 0:
            continue

        spikemat = hseu.extract_spikemat(
            spiking_data,
            start,
            end,
            time_window_s,
            time_window_advance_s
        )
        if spikemat.shape[0] == 0 or np.sum(spikemat) == 0:
            continue
        
        pos_idx = np.searchsorted(spiking_data.timestamps, decoding_times)
        pos_idx = np.clip(pos_idx, 1, len(pos_t) - 1)
        left = pos_idx - 1
        pos_idx -= (
            np.abs(decoding_times - pos_t[left])
            <= np.abs(pos_t[pos_idx] - decoding_times)
        )
        lfp_idx = np.searchsorted(lfp_t, decoding_times)
        lfp_idx = np.clip(lfp_idx, 1, len(lfp_t) - 1)
        left = lfp_idx - 1
        lfp_idx -= (
            np.abs(decoding_times - lfp_t[left])
            <= np.abs(lfp_t[lfp_idx] - decoding_times)
        )

        df = nap.TsdFrame(
            t=decoding_times,
            d=np.c_[
                position_data['x'].values[pos_idx],
                position_data['y'].values[pos_idx],
                position_data['Head direction'].values[pos_idx],
                position_data['Velocity'].values[pos_idx],
                position_data['V_x'].values[pos_idx],
                position_data['V_y'].values[pos_idx],
                phase[lfp_idx],
                power[lfp_idx]
            ],
            columns=[
                'x',
                'y',
                'Head direction',
                'Velocity',
                'V_x',
                'V_y',
                'Phase Deg',
                'Power'
            ]
        )

        tsdframes.append(df)
        spikemats.append(spikemat[:,place_cell_ids])

    return tsdframes, spikemats
        