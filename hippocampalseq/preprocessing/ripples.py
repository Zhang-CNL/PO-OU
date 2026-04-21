from typing import Optional
import pynapple as nap

def process_ripples(
        ripple_intervals: nap.IntervalSet,
        spiking_data: nap.TsGroup,
        time_window_ms: float = 5.0,
        time_window_advance_ms: Optional[float] = None
    ):
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_ms = time_window_ms
    time_window_advance_s = time_window_advance_ms / 1000