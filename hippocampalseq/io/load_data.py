import os
import numpy as np
import pynapple as nap 
import warnings
from scipy.signal import butter, filtfilt
from typing import Tuple, Dict

from .metadata import *
from .lfp import * 
from .spikes import *


def calc_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    velocity[velocity < 0] = 0
    return velocity, dt_filtered

def align_spikes_to_position(
        spikeframe: nap.TsGroup, 
        posframe: nap.TsdFrame, 
        minimum_dt: float = 0.1
    ) -> Tuple[nap.TsdFrame, Dict[int, nap.TsdFrame]]:
    """Align spikes to position data using nearest neighbor.
    Eliminates spikes that occur when no position data is available.

    Args:
        spikeframe (nap.TsGroup): Spike data.
        posframe (nap.TsdFrame): Position data.
        minimum_dt (float, optional): Minimum allowed time difference between spikes and position data. Spikes are
            discarded if the time difference is greater than this value. If set to np.inf, all spikes are kept.
            Defaults to 0.1.

    Returns:
        (nap.TsdFrame): Place cell spike times aligned to positions
        (Dict[int, nap.TsdFrame]): Dictionary of position information for each place cell's spikes.
    """
    spike_info = {}
    spike_times_filt = {}

    pos_times = posframe.index.values
    for uid in spikeframe.keys():
        spike_times = spikeframe[uid].index.values
        idx = np.searchsorted(pos_times, spike_times)

        idp = np.clip(idx - 1, 0, len(pos_times) - 1)
        idc = np.clip(idx, 0, len(pos_times) - 1)

        dtp = np.abs(spike_times - pos_times[idp])
        dtc = np.abs(spike_times- pos_times[idc])
        prev = dtp <= dtc

        nn = np.where(prev, idp, idc)
        td = np.where(prev, dtp, dtc)

        valid = td <= minimum_dt
        selectioni = nn[valid]

        spike_info[uid] = nap.TsdFrame(
            t=pos_times[selectioni],
            d=np.c_[
                posframe['x'].values[selectioni],
                posframe['y'].values[selectioni],
                posframe['Velocity'].values[selectioni],
                #posframe['delta t'].values[valid], 
                td[valid]
            ],
            columns=['x', 'y', 'Velocity', 'Delta t'],
        )
        st = spike_times[valid]
        spike_times_filt[uid] = nap.Ts(t=st)

    return nap.TsGroup(spike_times_filt), spike_info

def filter_noisy_epochs(
        rat_name: str, 
        session: int,
        track_type: str,
        raw_position_data,
        raw_spike_data,
        running_spike_data,
        running_spike_info,
    ):
    if rat_name in PFEIFFER_NOISY_EPOCHS:
        rat_values = PFEIFFER_NOISY_EPOCHS[rat_name]
        session = f"{track_type}{session}"
        if session in rat_values:
            starts = rat_values[session]['starts']
            ends   = rat_values[session]['ends']

            print(f"Removing {len(starts)} noisy epoch{'s' if len(starts) > 1 else ''}")

            ts = raw_position_data.time_support
            cleaned = ts.set_diff(nap.IntervalSet(starts, ends))

            raw_position_data  = raw_position_data.restrict(cleaned)
            raw_spike_data     = raw_spike_data.restrict(cleaned)
            running_spike_data = running_spike_data.restrict(cleaned)
            for uid in running_spike_info:
                running_spike_info[uid] = running_spike_info[uid].restrict(cleaned)

    return (
        raw_position_data,
        raw_spike_data,
        running_spike_data,
        running_spike_info
    )

def load_clean_data(
        data_path: str,
        rat_name: str, 
        session: int,
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        minimum_dt: float = 0.1
    ):
    """Load and segment data.

    Args:
        data_path (str): Path to data directory.
        rat_name (str): Rat name.
        session (int): Session number.
        track_type (str, optional): Track type. Can be one of ("Open", "Linear") Defaults to 'Linear'.
        ripple_type (str, optional): Ripple epoch to be extracted. Can be one of ("awake", "rem", "sleep", "sleep_immobile"). Defaults to 'awake'.
        minimum_dt (float, optional): Minimum allowed time difference between spikes and position data. Spikes are
            discarded if the time difference is greater than this value. If set to np.inf, all spikes are kept.
            Defaults to 0.1.

    Returns:
        (nap.TsdFrame): Raw, unfiltered position data.
        (nap.TsdFrame): Running position data. Filtered out according to `ripple_type`.
        (nap.TsGroup): Raw spiking data.
        (Dict[int, nap.TsdFrame]): Dictionary of position information for each cell's spikes.
        (nap.TsGroup): Filtered spiking data aligned to positions and run times according to `ripple_type`.
        (nap.IntervalSet): Start and end periods of sharp-wave ripples.
        (dict): Lfp data such as phase and amplitude in a dict.
        (np.ndarray): Indices of excitatory neurons.
        (np.ndarray): Indices of inhibitory neurons.
    """

    # TODO: Where in here do I use the function to filter noisy epochs

    (
        time,
        x, y, hd,
        epoch_starts,
        epoch_ends,
        spike_data,
        ripple_periods,
        excitatory_neurons,
        inhibitory_neurons
    ) = load_spiking_data(
        data_path,
        rat_name,
        session,
        track_type,
        ripple_type
    )

    v, dt = calc_velocity(x, y, time)
    dt[dt > 60] = 0

    epoch = nap.IntervalSet(start=epoch_starts, end=epoch_ends)

    raw_position = nap.TsdFrame(
        t=time,
        d=np.c_[
            x, 
            y,
            hd,
            v,
            dt
        ],
        columns=['x','y','Head direction','Velocity','Delta t']
    )
    running_position = raw_position.restrict(epoch)
    running_spikes   = spike_data.restrict(epoch)

    running_spikes,running_spike_info = align_spikes_to_position(
        running_spikes, 
        running_position, 
        minimum_dt
    )

    (
        raw_position,
        spike_data,
        running_spikes,
        running_spike_info    
    ) = filter_noisy_epochs(
        rat_name, 
        session, 
        track_type, 
        raw_position, 
        spike_data,
        running_spikes, 
        running_spike_info
    )

    path = os.path.join(data_path, rat_name, f"{track_type}{session}")
    lfp = load_lfp_data(raw_position, spike_data, path)

    return (
        raw_position,
        running_position,
        spike_data,
        running_spike_info,
        running_spikes,
        ripple_periods,
        lfp,
        excitatory_neurons,
        inhibitory_neurons
    )