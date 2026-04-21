import os
import numpy as np
import pynapple as nap 
import warnings
from scipy.signal import butter, filtfilt

from .metadata import *
import hippocampalseq.utils as hseu

def calc_velocity(x, y, t):
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

def align_spikes_to_position(spikeframe, posframe, minimum_dt=0.1):
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
                posframe['velocity'].values[selectioni],
                #posframe['delta t'].values[valid], 
                td[valid]
            ],
            columns=['x', 'y', 'velocity', 'delta t'],
        )
        st = spike_times[valid]
        spike_times_filt[uid] = nap.Ts(t=st)

    return nap.TsGroup(spike_times_filt), spike_info

def filter_noisy_epochs(
        rat_name: str, 
        session: int,
        track_type: str,
        position_data,
        spike_data,
        spike_info
    ):
    if rat_name in PFEIFFER_NOISY_EPOCHS:
        rs = PFEIFFER_NOISY_EPOCHS[rat_name]
        session = f"{track_type}{session}"
        if session in rs:
            print("Removing noisy epochs")
            starts = rs[session]['starts']
            ends = rs[session]['ends']
            ts = position_data.time_support
            cleaned = ts.set_diff(starts, ends)
            position_data = position_data.restrict(cleaned)
            spike_data = spike_data.restrict(cleaned)
            spike_info_clean = {}
            for uid,ts in spike_info.items():
                clean = ts.restrict(cleaned)
                if len(clean) > 0:
                    spike_info_clean[uid] = clean
            spike_info = np.TsGroup(spike_info_clean)

    return position_data, spike_data, spike_info

def load_clean_data(
        data_path: str,
        rat_name: str, 
        session: int,
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        minimum_dt: float = 0.1
    ):
    assert rat_name in RAT_NAMES, f"{rat_name} not in {RAT_NAMES}"
    assert ripple_type in ['awake', 'rem', 'sleep', 'sleep_immobile']
    assert track_type in ['Linear', 'Open']

    session = f"{track_type}{session}"
    path = os.path.join(data_path, rat_name, session)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    pos_mat = hseu.read_mat(os.path.join(path, 'Position_Data.mat'))
    raw_pos = pos_mat['Position_Data']
    
    time = raw_pos[:,0] # (Npos, 1)
    x    = raw_pos[:,1]
    y    = raw_pos[:,2]
    hd   = raw_pos[:,3]

    epoch_mat = hseu.read_mat(os.path.join(path, 'Epochs.mat'))
    if ripple_type == 'awake':
        rt = np.squeeze(epoch_mat['Run_Times']).astype(float)
    elif ripple_type == 'rem':
        rt = np.squeeze(epoch_mat['REM_Times']).astype(float)
    elif ripple_type == 'sleep':
        rt = np.squeeze(epoch_mat['Sleep_Times']).astype(float)
    elif ripple_type == 'sleep_immobile':
        rt = np.squeeze(epoch_mat['Sleep_Box_Immobile_Times']).astype(float)
    
    warnings.warn("Check the number of epochs in rt")
    rt = np.atleast_2d(np.squeeze(rt))
    starts = rt[:,0]
    ends   = rt[:,1]

    spike_mat = hseu.read_mat(os.path.join(path, 'Spike_Data.mat'))
    spikes = spike_mat['Spike_Data']

    # (Nspikes,1)
    spike_ids   = spikes[:,1].astype(int) - 1
    spike_times = spikes[:,0]


    excitatory_neurons = spike_mat['Excitatory_Neurons'].astype(int) - 1
    inhibitory_neurons = spike_mat['Inhibitory_Neurons'].astype(int) - 1

    ripple_mat = hseu.read_mat(os.path.join(path, 'Ripple_Events.mat'))
    ripples    = ripple_mat['Ripple_Events']


    v, dt = calc_velocity(x, y, time)
    dt[dt > 60] = 0

    epoch = nap.IntervalSet(start=starts, end=ends)

    raw_position = nap.TsdFrame(
        t=time,
        d=np.c_[
            x, 
            y,
            v,
            dt
        ],
        columns=['x','y','velocity','delta t']
    )
    running_position = raw_position.restrict(epoch)

    unique_cells = np.unique(spike_ids)
    cell_spikes = {}
    for cell in unique_cells:
        spikes = spike_times[spike_ids == cell]
        cell_spikes[cell] = nap.Ts(t=np.sort(spikes))

    spike_data = nap.TsGroup(cell_spikes)
    running_spikes = spike_data.restrict(epoch)

    running_spikes,running_spike_info = align_spikes_to_position(running_spikes, running_position, minimum_dt)

    return (
        raw_position,
        running_position,
        spike_data,
        running_spike_info,
        running_spikes,
        excitatory_neurons,
        inhibitory_neurons
    )