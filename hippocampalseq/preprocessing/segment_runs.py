import numpy as np
from scipy.signal import butter, filtfilt

import hippocampalseq.utils as hseu


def fix_alignment(spike_times, pos_times, spike_ids, x, y, min_time_diff=np.inf):
    """Fixes alignment between spike times and position times by removing spikes
     that are not within the position recording period and position points that 
     are not within the spike recording period.

    Args:
        spike_times (np.ndarray): Times of the spikes
        pos_times (np.ndarray): Times of the position data
        spike_ids (np.ndarray): IDs of the spikes
        x (np.ndarray): x position data
        y (np.ndarray): y position data

    Returns:
        tuple: spike_times, pos_times, spike_ids, x, y
    """

    _st = []
    _id = []
    _x  = []
    _y  = []
    _t  = []
    uids = np.unique(spike_ids)
    for uid in uids:
        st = spike_times[spike_ids == uid]
        idx = np.searchsorted(pos_times, st)

        idp = np.clip(idx - 1, 0, len(pos_times) - 1)
        idc = np.clip(idx, 0, len(pos_times) - 1)

        dtp = np.abs(st - pos_times[idp])
        dtc = np.abs(st - pos_times[idc])
        prev = dtp <= dtc

        nn = np.where(prev, idp, idc)
        td = np.where(prev, dtp, dtc)

        valid = td <= min_time_diff
        selectioni = nn[valid]

        _x.append(x[selectioni])
        _y.append(y[selectioni])
        _t.append(pos_times[selectioni])
        _st.append(st[valid])
        _id.append(np.full(len(valid), uid))

    x           = np.concatenate(_x)
    y           = np.concatenate(_y)
    pos_times   = np.concatenate(_t)
    spike_times = np.concatenate(_st)
    spike_ids   = np.concatenate(_id)

    idx         = np.argsort(pos_times)
    x           = x[idx]
    y           = y[idx]
    pos_times   = pos_times[idx]
    spike_times = spike_times[idx]
    spike_ids   = spike_ids[idx]

    return spike_times, pos_times, spike_ids, x, y

def calculate_velocity(x: np.ndarray, y: np.ndarray, time: np.ndarray):
    """
    Calculates the velocity of the rat given position data.

    Args:
        x (np.ndarray): x position data
        y (np.ndarray): y position data
        time (np.ndarray): Times of the position data

    Returns:
        velocity_t (np.ndarray): Times of the velocity
        velocity (np.ndarray): Velocity
    """
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])
    dt = np.concatenate([dt, [dt[-1]]])

    median_dt = np.median(dt)
    dt[dt > 10 * median_dt] = median_dt
    #This filtering removes jittery jumps from the camera 
    # obscuring the LED sometimes
    b, a = butter(2, 0.02)
    dt = filtfilt(b, a, dt)
    dt[dt <= 0] = np.min(dt[dt > 0]) / 10
    
    # Filter position changes - same reasoning
    b, a = butter(2, 0.2)
    dx = filtfilt(b, a, dx)
    dy = filtfilt(b, a, dy)

    distance = np.sqrt(dx**2 + dy**2)
    velocity = np.abs(distance / dt)
    #velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / hseu.PFEIFFER_RECORDING_FPS
    velocity_t = (time[1:] + time[:-1]) / 2

    nv = np.isnan(velocity)
    if np.any(nv):
        velocity_t = velocity_t[~nv]
        velocity   = velocity[~nv]
    return velocity_t, velocity

def calculate_run_periods(velocity: np.ndarray, velocity_times: np.ndarray, velocity_run_threshold: float = 5.0):
    run_periods = velocity >= velocity_run_threshold
    run_start,run_end = hseu.extract_times_from_boolean(run_periods, velocity_times)
    return run_start,run_end

def segment_rundata(raw_data: hseu.RawData, velocity_run_threshold: float = 5.0, alignment_min_dt: float = 0.1):
    velocity_t,velocity = calculate_velocity(raw_data.x, raw_data.y, raw_data.time)
    run_starts,run_ends = calculate_run_periods(velocity, velocity_t, velocity_run_threshold)

    spikes_aligned,t_aligned,sids_aligned,x_aligned,y_aligned = fix_alignment(
        raw_data.spike_times,
        raw_data.time,
        raw_data.spike_ids,
        raw_data.x,
        raw_data.y,
        alignment_min_dt
    )

    tsidx = np.searchsorted(t_aligned, run_starts)
    ssidx = np.searchsorted(spikes_aligned, run_starts)
    teidx = np.searchsorted(t_aligned, run_ends)
    seidx = np.searchsorted(spikes_aligned, run_ends)

    tidx = hseu.create_interval_mask(len(t_aligned), tsidx, teidx)
    sidx = hseu.create_interval_mask(len(spikes_aligned), ssidx, seidx)

    run_spikes = {}
    for cell_id in raw_data.unique_cells:
        run_spikes[cell_id] = spikes_aligned[sidx]

    return hseu.RunningData(
        run_starts  = run_starts,
        run_ends    = run_ends,
        x           = x_aligned[tidx],
        y           = y_aligned[tidx],
        time        = t_aligned[tidx],
        spike_ids   = sids_aligned[sidx],
        spike_times = spikes_aligned[sidx],
        velocity    = velocity,
        velocity_t  = velocity_t,
        cell_spikes = run_spikes,
        unique_cells= raw_data.unique_cells
    )