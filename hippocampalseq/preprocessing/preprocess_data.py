import os
import copy
import numpy as np
import hippocampalseq.utils as hseu
from typing import Optional
from .placefields import * 
from .ripples import *
from .theta import *

def fix_alignment(spike_times, pos_times, spike_ids, x, y):
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
    spikes_before = spike_times < pos_times[0]
    spikes_after  = spike_times > pos_times[-1]
    spike_conj    = ~spikes_before & ~spikes_after
    spike_times   = spike_times[spike_conj]
    spike_ids     = spike_ids[spike_conj]

    pos_before = pos_times < spike_times[0]
    pos_after  = pos_times > spike_times[-1]
    pos_conj   = ~pos_before & ~pos_after
    pos_times  = pos_times[pos_conj]
    x          = x[pos_conj]
    y          = y[pos_conj]

    return spike_times, pos_times, spike_ids, x, y

def clean_gaps(x: np.ndarray, y: np.ndarray, time: np.ndarray, position_gap_threshold_s: float = 0.25):
    """Cleans gaps in position data by removing points that are more than
    position_gap_threshold_s seconds apart from the previous point.
    Gaps are filled with NaN so as to ignore them in place-field and velocity calculations

    Args:
        x (np.ndarray): x position data
        y (np.ndarray): y position data
        time (np.ndarray): Times of the position data
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        x (np.ndarray): Cleaned x position data
        y (np.ndarray): Cleaned y position data
    """
    
    pdiff = np.diff(time)
    gaps  = np.where(pdiff > position_gap_threshold_s)[0]
    if len(gaps) > 0:
        print(f"Removing {len(gaps)} position gaps")
        for i in gaps:
            x[i - 5 : i + 5] = np.nan
            y[i - 5 : i + 5] = np.nan
            print(f"\t(x,y) = ({x[i]}, {y[i]})")
    return x,y

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
    velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / hseu.PFEIFFER_RECORDING_FPS
    velocity_t = (time[1:] + time[:-1]) / 2

    nv = np.isnan(velocity)
    if np.any(nv):
        velocity_t = velocity_t[~nv]
        velocity   = velocity[~nv]
    return velocity_t, velocity

def calculate_run_periods(velocity: np.ndarray, velocity_times: np.ndarray, velocity_run_threshold: float = 5.0):
    run_periods = velocity > velocity_run_threshold
    run_start,run_end = hseu.extract_times_from_boolean(run_periods, velocity_times)
    return run_start,run_end

def load_clean_data(
        data_path: str,
        rat_name: str, 
        session: int,
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0
    ) -> hseu.RatData:
    assert rat_name in hseu.RAT_NAMES, f"{rat_name} not in {hseu.RAT_NAMES}"
    assert session in [1,2]

    path = os.path.join(data_path, rat_name, f"Open{session}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    pos_mat = hseu.read_mat(os.path.join(path, 'Position_Data.mat'))
    raw_pos = pos_mat['Position_Data']
    
    time = raw_pos[:,0]
    x    = raw_pos[:,1]
    y    = raw_pos[:,2]
    hd   = raw_pos[:,3]

    # It looks like we don't really need this since
    # we can just use the velocity to parse it all out, but we'll keep 
    # it for now
    epoch_mat = hseu.read_mat(os.path.join(path, 'Epochs.mat'))
    rt = np.squeeze(epoch_mat['Run_Times']).astype(float)

    start = rt[0]
    end   = rt[1]

    spike_mat = hseu.read_mat(os.path.join(path, 'Spike_Data.mat'))
    spikes = spike_mat['Spike_Data']
    
    spike_ids = spikes[:,1].astype(int) - 1
    spike_times = spikes[:,0]

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
    spike_times,time,spike_ids,x,y = fix_alignment(spike_times, time, spike_ids, x, y)
    x,y = clean_gaps(x, y, time, position_gap_threshold_s)
    velocity_t,velocity = calculate_velocity(x, y, time)
    run_starts,run_ends = calculate_run_periods(velocity, velocity_t, velocity_run_threshold_s)

    raw_data = hseu.RawData(
        time           = time,
        x              = x,
        y              = y,
        head_direction = hd,
        velocity       = velocity,
        velocity_time  = velocity_t,
        run_starts     = run_starts,
        run_ends       = run_ends,
        spike_ids      = spike_ids,
        spike_times    = spike_times,
        raw_ripples    = ripples
    )
    return hseu.RatData(
        rat_name           = rat_name,
        session            = session,
        raw_data           = raw_data,
        excitatory_neurons = excitatory_neurons,
        inhibitory_neurons = inhibitory_neurons,
        well_sequence      = well_seq,
        n_ripples          = len(ripples),
        n_cells            = np.max(spike_ids)
    )

def load_and_preprocess(
        data_path: str,
        rat_name: str, 
        session: int, 
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        bin_size_cm: int = 4,
        place_field_gaussian_sd_cm: float = 4.0,
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