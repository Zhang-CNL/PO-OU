from typing import Optional
import numpy as np
import pynapple as nap

import hippocampalseq.utils as hseu

def process_ripples(
        ripple_intervals: nap.IntervalSet,
        spiking_data: nap.TsGroup,
        place_cell_ids: np.ndarray,
        time_window_s: float = 5.0 / 1000,
        time_window_advance_s: Optional[float] = None
    ):
    if time_window_advance_s is None:
        time_window_advance_s = time_window_s
    
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
        if spikemat.shape[0] == 0 or np.sum(spikemat) == 0:
            continue
        spikemats.append(spikemat[:,place_cell_ids])

    return spikemats
        