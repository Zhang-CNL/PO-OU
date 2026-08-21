import numpy as np

def create_interval_mask(length: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Helper to create a boolean mask from start/end indices."""
    mask = np.zeros(length + 1, dtype=int)
    np.add.at(mask, starts, 1)
    np.add.at(mask, ends, -1)
    return np.cumsum(mask)[:-1] > 0


def extract_times_from_boolean(mask: np.ndarray, run_times: np.ndarray):
    b = np.asarray(mask, dtype=bool)
    run_times = np.asarray(run_times)

    if b.size == 0:
        return np.array([]), np.array([])

    # +1 where False->True (rising edge), -1 where True->False (falling edge)
    edges = np.diff(b.astype(np.int8))
    rising_idx  = np.flatnonzero(edges == 1) + 1
    falling_idx = np.flatnonzero(edges == -1) + 1

    start_times = run_times[rising_idx]
    end_times   = run_times[falling_idx]

    if b[0]:
        start_times = np.concatenate(([run_times[0]], start_times))
    if b[-1]:
        end_times = np.concatenate((end_times, [run_times[-1]]))

    return start_times, end_times

def restrict_indices(t_array: np.ndarray, start: float, end: float) -> slice:
    start_ind = int(np.searchsorted(t_array, start, side='left'))
    end_ind   = int(np.searchsorted(t_array, end, side='right'))
    return slice(start_ind, end_ind)

def times_to_bool(data_times, start_time, end_time):
    times_after_start = data_times >= start_time
    times_before_end = data_times <= end_time
    window_ind = times_after_start & times_before_end
    return window_ind