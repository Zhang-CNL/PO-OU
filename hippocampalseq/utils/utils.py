import numpy as np
import pynapple as nap
import torch
from typing import Any 
from collections.abc import Callable, Iterable
from scipy.signal import butter, filtfilt

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

def calculate_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate velocity from position and time data.

    Args:
        x (np.ndarray): x position data.
        y (np.ndarray): y position data.
        t (np.ndarray): time data.

    Returns:
        (np.ndarray): velocity data.
        (np.ndarray): time data.
    """
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.concatenate([dt, [dt[-1]]]) 
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])
    
    median_dt = np.median(dt)
    dt[dt > 10 * median_dt] = median_dt
    b, a = butter(2, 0.02)
    dt_filtered = filtfilt(b, a, dt)
    dt_filtered[dt_filtered <= 0] = np.min(dt_filtered[dt_filtered > 0]) / 10
    
    b, a = butter(2, 0.2)
    dx_filtered = filtfilt(b, a, dx)
    dy_filtered = filtfilt(b, a, dy)
    
    distance = np.sqrt(dx_filtered**2 + dy_filtered**2)
    velocity = np.abs(distance / dt_filtered)
    vx = np.abs(dx_filtered / dt_filtered)
    vy = np.abs(dy_filtered / dt_filtered)
    return (
        velocity, 
        dt_filtered,
        vx,
        vy
    )

def calculate_velocity_dt(x: np.ndarray, dt: float|np.ndarray):
    dx = np.diff(x)
    dx = np.concatenate([dx, [dx[-1]]])
    b,a = butter(2, .2)
    dx = filtfilt(b,a,dx)
    return np.abs(dx / dt)
