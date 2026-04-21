from typing import Optional
import pynapple as nap

import hippocampalseq.utils as hseu

def process_ripples(
        ripple_intervals: nap.IntervalSet,
        spiking_data: nap.TsGroup,
        place_cell_ids: np.ndarray,
        time_window_ms: float = 5.0,
        time_window_advance_ms: Optional[float] = None
    ):
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_ms = time_window_ms
    time_window_advance_s = time_window_advance_ms / 1000
    
    starts = ripple_intervals.start
    ends = ripple_intervals.end
    spikemats = []

    for start,end in zip(starts,ends):
        spikemat = hseu.extract_spikemat(
            spiking_data,
            start,
            end,
            time_window_s,
            time_window_advance_s
        )
        if spikemat.shape[0] == 0:
            continue
        spikemats.append(spikemat[:,place_cell_ids])

    return spikemats
        