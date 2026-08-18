import numpy as np
import pynapple as nap
import torch
from typing import Any 
from collections.abc import Callable, Iterable

type NDArray = np.ndarray|torch.Tensor

class AttrDict(dict):
    def __init__(self, dct):
        super().__init__(dct)
        self.__dict__ = dct

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.__dict__[k] = v

    def __copy__(self):
        return AttrDict(self)

def spike_info_to_tsg(spike_info: dict[int, nap.TsdFrame]) -> nap.TsGroup:
    return nap.TsGroup({
        cell: nap.Ts(t=spike_info[cell].index.values) for cell in spike_info
    })

def changeover_functions(_type: type, *args: Iterable[str]) -> Callable[...,Any]|list[Callable[...,Any]]:
    """Given a type of an array, return an arbritrary list of functions from the proper module
    (torch or numpy) corresponding to the arguments.

    Args:
        _type (type): The type of the array.
        *args (tuple[str,...]): The names of the functions to return.

    Returns:
        Callable[...,Any]|list[Callable[...,Any]]: The functions from the proper module.
    """
    module = torch if _type == torch.Tensor else np
    attrs = [getattr(module, arg) for arg in args]
    if len(attrs) == 1:
        return attrs[0]
    return attrs

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

def cm_to_bins(array_in_cm: float|np.ndarray, bin_size_cm: int = 2):
    return np.floor(array_in_cm / bin_size_cm)  # cm to bins


def extract_spikemat(
        spiking_data: nap.TsGroup,
        run_start: float,
        run_end: float,
        time_window_s: float,
        time_window_step_s: float
    ) -> np.ndarray:
    """Extract a discretized spiking matrix from a group of spike times.

    Args:
        spiking_data (nap.TsGroup): Group of spike times. N cells with spikes.
        run_start (float): Start time of the run.
        run_end (float): End time of the run.
        time_window_s (float): Size of the time window in seconds.
        time_window_step_s (float): Step size of the time window in seconds.

    Returns:
        (np.ndarray): Discretized spiking matrix. Will have shape (T,N)
    """
    ncells = len(spiking_data)
    
    bins = np.arange(run_start, run_end, time_window_step_s)
    epoch = nap.IntervalSet(run_start, run_end)
    spikes = spiking_data.restrict(epoch)

    if len(spikes) == 0:
        return np.zeros((0, len(spikes)), dtype=int)

    if np.isclose(time_window_s, time_window_step_s):
        spikemat = spikes.count(time_window_s, ep=epoch).values
    else:
        t,uids = [],[]
        for uid, ts in spikes.items():
            times = ts.index.values
            t.append(times)
            uids.append(np.full(len(times), uid))

        t = np.concatenate(t)
        uids = np.concatenate(uids)

        sidx = np.argsort(t)
        t = t[sidx]
        uids = uids[sidx]

        spikemat = np.zeros((len(bins), ncells), dtype=int)
       
        start_idx = np.searchsorted(t, bins, side='left')
        end_idx   = np.searchsorted(t, bins + time_window_s, side='right')
        for i in range(len(bins)):
            wuid = uids[start_idx[i]:end_idx[i]]
            if len(wuid) > 0:
                counts = np.bincount(wuid, minlength=ncells)
                spikemat[i,:] = counts
    return spikemat
    
def make_ndgrid(bounds: list[tuple[float,...]], bin_size: int|list[int], indexing='xy') -> torch.Tensor:
    n_dims = len(bounds)
    if isinstance(bin_size, int) or isinstance(bin_size, float):
        bin_size = [bin_size] * n_dims
    if len(bin_size) != n_dims:
        raise ValueError(f"Number of bins ({len(n_bins)}) does not match number of dimensions ({n_dims})")

    edges = [
        torch.arange(lo,hi,nb) + nb / 2 for (lo,hi), nb in zip(bounds, bin_size)
    ]
    coords = torch.meshgrid(*edges, indexing=indexing)
    return torch.stack([c.ravel() for c in coords]).T
    
def atleast_2d(x: NDArray) -> NDArray:
    """Ensure that the input array has at least 2 dimensions.
    Differs from np.atleast_2d in that it appends the dimension instead of prepending it

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Input array with at least 2 dimensions.
    """
    if x.ndim < 2:
        x = x[...,None]
        return atleast_2d(x)
    return x

def atleast_3d(x: NDArray) -> NDArray:
    if x.ndim < 3:
        x = x[...,None]
        return atleast_3d(x)
    return x
