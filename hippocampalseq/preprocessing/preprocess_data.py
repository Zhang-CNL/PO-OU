import os
import copy
import numpy as np
from typing import Optional
import warnings

import hippocampalseq.utils as hseu
from .placefields import * 
from .ripples import *
from .theta import *


def filter_noisy(raw_data: hseu.RatData, rat_name: str, session: str) -> hseu.RatData:
    if rat_name in hseu.PFEIFFER_NOISY_EPOCHS:
        rat_sessions = hseu.PFEIFFER_NOISY_EPOCHS[rat_name]
        if session in rat_sessions:
            starts = rat_sessions[session]['starts']
            ends = rat_sessions[session]['ends']
            for start,end in zip(starts,ends):
                spike_mask = (raw_data.spike_times < start) & (raw_data.spike_times > end)
                raw_data.spike_times = raw_data.spike_times[spike_mask]
                raw_data.spike_ids = raw_data.spike_ids[spike_mask]

                pos_mask = (raw_data.time < start) & (raw_data.time > end)
                raw_data.time = raw_data.time[pos_mask]
                raw_data.x = raw_data.x[pos_mask]
                raw_data.y = raw_data.y[pos_mask]

    return raw_data


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
    #dx = np.concatenate((dx, dx[-1]))
    #dy = np.concatenate((dy, dy[-1]))
    #dt = np.concatenate((dt, dt[-1]))

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

def segment_rundata(raw_data: hseu.RawData, velocity_run_threshold: float = 5.0):
    velocity_t,velocity = calculate_velocity(raw_data.x, raw_data.y, raw_data.time)
    run_starts,run_ends = calculate_run_periods(velocity, velocity_t, velocity_run_threshold)

    spike_times, time_aligned, spike_ids, x_aligned, y_aligned = fix_alignment(
        raw_data.spike_times,
        raw_data.time,
        raw_data.spike_ids,
        raw_data.x,
        raw_data.y
    )
    tsidx = np.searchsorted(time_aligned, run_starts)
    ssidx = np.searchsorted(spike_times, run_starts)
    teidx = np.searchsorted(time_aligned, run_ends)
    seidx = np.searchsorted(spike_times, run_ends)

    tidx = np.zeros_like(time_aligned, dtype=bool)
    sidx = np.zeros_like(spike_times, dtype=bool)
    for i in range(len(tsidx)):
        tidx[tsidx[i]:teidx[i]] = True
        sidx[ssidx[i]:seidx[i]] = True

    return hseu.RunningData(
        run_starts  = run_starts,
        run_ends    = run_ends,
        x           = x_aligned[tidx],
        y           = y_aligned[tidx],
        time        = time_aligned[tidx],
        spike_ids   = spike_ids[sidx],
        spike_times = spike_times[sidx],
        velocity    = velocity,
        velocity_t  = velocity_t
    )

def load_clean_data(
        data_path: str,
        rat_name: str, 
        session: int,
        ripple_type: str = 'awake',
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        drop_misaligned: bool = False
    ) -> hseu.RatData:
    assert rat_name in hseu.RAT_NAMES, f"{rat_name} not in {hseu.RAT_NAMES}"
    assert session in [1,2]
    assert ripple_type in ['awake', 'rem', 'sleep', 'sleep_immobile']

    session = f"Linear{session}"
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
    start = rt[0]
    end   = rt[1]

    spike_mat = hseu.read_mat(os.path.join(path, 'Spike_Data.mat'))
    spikes = spike_mat['Spike_Data']

    # (Nspikes,1)
    spike_ids   = spikes[:,1].astype(int) - 1
    spike_times = spikes[:,0]

    # Restrict to desired epoch
    spike_slice = hseu.restrict_indices(spike_times, start, end)
    spike_ids   = spike_ids[spike_slice]
    spike_times = spike_times[spike_slice]

    pos_slice = hseu.restrict_indices(time, start, end)
    time      = time[pos_slice]
    x         = x[pos_slice]
    y         = y[pos_slice]
    hd        = hd[pos_slice]

    excitatory_neurons = spike_mat['Excitatory_Neurons'].astype(int) - 1
    inhibitory_neurons = spike_mat['Inhibitory_Neurons'].astype(int) - 1

    ripple_mat = hseu.read_mat(os.path.join(path, 'Ripple_Events.mat'))
    ripples = ripple_mat['Ripple_Events']


    try:
        well_mat = hseu.read_mat(os.path.join(path, 'Well_Sequence.mat'))
        well_seq = well_mat['Well_Sequence']
    except FileNotFoundError:
        print("No well sequence found, who cares")
        well_seq = []

    # Align spike and position data, remove large gaps, and calculate velocity.
    # Each spike should have a corresponding position, otherwise we just remove it.
    # Other methods interpolate, but Krause et al chose to simply drop it
    #if drop_misaligned:
    #    spike_times,time,spike_ids,x,y = fix_alignment(spike_times, time, spike_ids, x, y)
    #    x,y = clean_gaps(x, y, time, position_gap_threshold_s)
    #velocity_t,velocity = calculate_velocity(x, y, time)
    #run_starts,run_ends = calculate_run_periods(velocity, velocity_t, velocity_run_threshold_s)

    unique_cells = np.unique(spike_ids)
    cell_spikes = []
    for cell in unique_cells:
        spikes = spike_times[spike_ids == cell]
        cell_spikes.append(spikes)

    raw_data = hseu.RawData(
        time           = time,
        x              = x,
        y              = y,
        head_direction = hd,
        spike_ids      = spike_ids,
        spike_times    = spike_times,
        raw_ripples    = ripples,
        unique_cells   = unique_cells,
        cell_spikes    = cell_spikes
    )

    raw_data = filter_noisy(raw_data, rat_name, session)

    run_data = segment_rundata(raw_data, velocity_run_threshold_s)

    return hseu.RatData(
        rat_name           = rat_name,
        session            = session,
        raw_data           = raw_data,
        run_data           = run_data,
        excitatory_neurons = excitatory_neurons,
        inhibitory_neurons = inhibitory_neurons,
        well_sequence      = well_seq,
        n_ripples          = len(ripples),
        n_cells            = np.max(spike_ids) + 1
    )

def load_and_preprocess(
        data_path: str,
        rat_name: str, 
        session: int, 
        ripple_type: str = 'awake',
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        time_window_ms: float = 3.0,
        time_window_advance_ms: Optional[float] = None,
        avg_fr_smoothing_convolution: np.ndarray = np.array([.25, .25, .25, .25]),
        avg_spikes_per_s_threshold: int = 2,
        min_popburst_duration_ms: int = 30,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        seed: int|None = 42
    ) -> hseu.RatData:
    """Runs all preprocessing steps on the given rat data.

    Args:
        data_path (str): Path to data directory
        rat_name (str): Name of the rat
        session (int): Session number (1 or 2)
        position_gap_threshold_s (float): Threshold for position gaps in seconds. Defaults to 0.25.
        velocity_run_threshold_s (float): Threshold for determining if the rat is running in centimeters per second. Defaults to 5.0.
        bin_size_cm (int): Bin size for place field calculation in centimeters. Defaults to 4.
        place_field_gaussian_sd_cm (float): Standard deviation of Gaussian for place field calculation in centimeters. Defaults to 4.0.

    Returns:
        hseu.RatData: Dictionary containing rat data with additional fields for velocity and place field data
    """
    #rat_data = load_rat(data_path, rat_name, session, sweep_type)
    #rat_data = clean_data(rat_data, position_gap_threshold_s)
    #rat_data = calculate_velocity(rat_data, velocity_run_threshold_s)
    rat_data = load_clean_data(
        data_path, 
        rat_name, 
        session, 
        ripple_type,
        position_gap_threshold_s, 
        velocity_run_threshold_s
    )
    rat_data.place_field_data = calculate_placefields(
        rat_data,
        bin_size_cm, 
        place_field_gaussian_sd_cm,
        position_gap_threshold_s
    )
    rat_data.ripple_data = process_ripples(
        rat_data,
        time_window_ms,
        time_window_advance_ms,
        avg_fr_smoothing_convolution,
        avg_spikes_per_s_threshold,
        min_popburst_duration_ms
    )

    rat_data.theta_data = process_theta(
        rat_data, 
        run_period_threshold, 
        place_field_scaling_factor,
        velocity_scaling_factor,
        time_window_ms,
        time_window_advance_ms,
        seed
    )

    return rat_data