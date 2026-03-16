import os
import copy
import numpy as np
import hippocampalseq.utils as hseu
from typing import Optional
from .placefields import * 
from .ripples import *
from .data import *



def __extract_runs(
        epochs_dict: dict, 
        pos_dict: dict, 
        spike_dict: dict, 
        ripple_dict: dict, 
        sweep_type: str 
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract position, spike, and ripple data for a run from the given dictionaries.

    Args:
        epochs_dict (dict): Dictionary containing run times
        pos_dict (dict): Dictionary containing position data
        spike_dict (dict):  Dictionary containing spike data
        ripple_dict (dict): Dictionary containing ripple data
        sweep_type (str): Type of sweep ("theta" or "replay")

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing position, spike, and ripple data for the run
    """
    run_times = epochs_dict["Run_Times"]
    spikes = spike_dict['Spike_Data']
    spike_ind = hseu.times_to_bool(spikes[:,0], run_times[0], run_times[1])

    ripples = ripple_dict['Ripple_Events']
    rip_idx = hseu.times_to_bool(ripples[:,0], run_times[0], run_times[1])

    pos = pos_dict['Position_Data']
    pos_idx = hseu.times_to_bool(pos[:,0], run_times[0], run_times[1])

    if sweep_type == "theta":
        spikes  = spikes[spike_ind,:]
        ripples = ripples[rip_idx,:]
        pos     = pos[pos_idx,:]
    elif sweep_type == "replay":
        spikes  = spikes[~spike_ind,:]    
        ripples = ripples[~rip_idx,:]
        pos     = pos[~pos_idx,:]
    return pos,spikes,ripples



def __calculate_velocity(pos_xy: np.ndarray) -> np.ndarray:
    """Calculates the velocity of a given position data.

    Args:
        pos_xy (np.ndarray): Position data

    Returns:
        np.ndarray: Velocity data
    """
    xdiff = np.diff(pos_xy[:,0])
    ydiff = np.diff(pos_xy[:,1])
    return np.sqrt(xdiff**2 + ydiff**2) / PFEIFFER_RECORDING_FPS

def __align_spikes_to_pos(rat_data: RatData) -> RatData:
    """Align spike and position data by removing spikes and positions outside of each other's recording times.

    Args:
        rat_data (RatData): Data of the rat

    Returns:
        RatData: Aligned data
    """
    # Remove spikes outside of position recording
    spike_times = rat_data.spike_times_sec
    pos_times   = rat_data.pos_times_sec
    spikes_before_pos = spike_times < pos_times[0]
    spikes_after_pos  = spike_times > pos_times[-1]
    spike_conj = ~spikes_before_pos & ~spikes_after_pos
    rat_data.spike_ids = rat_data.spike_ids[spike_conj]
    rat_data.spike_times_sec = rat_data.spike_times_sec[spike_conj]    

    # Remove positions outside of spike recording
    pos_before_spikes = pos_times < spike_times[0]
    pos_after_spikes  = pos_times > spike_times[-1]
    pos_conj = ~pos_before_spikes & ~pos_after_spikes
    rat_data.pos_xy_cm = rat_data.pos_xy_cm[pos_conj]
    rat_data.pos_times_sec = rat_data.pos_times_sec[pos_conj]
    return rat_data

def __clean_gaps(rat_data: RatData, position_gap_threshold_s: float) -> RatData:
    """Cleans gaps in position data.

    Removes any points in the position data that are more than
    position_gap_threshold_s seconds apart from the previous point.

    Args:
        rat_data (RatData): Dictionary containing rat data
        position_gap_threshold_s (float): Threshold for position gaps in seconds

    Returns:
        RatData: Dictionary containing cleaned position data
    """
    pos_diff = np.diff(rat_data.pos_times_sec)
    pos_gaps = np.where(pos_diff > position_gap_threshold_s)[0]
    pos_xy   = rat_data.pos_xy_cm[:]
    if len(pos_gaps) > 0:
        for ind in pos_gaps:
            pos_xy[ind - 5 : ind + 5] = np.nan
    rat_data.pos_xy_cm           = pos_xy
    rat_data.large_position_gaps = pos_gaps
    return rat_data

def load_rat(data_path: str, rat_name: str, session: int, sweep_type: str = "theta") -> RatData:
    """Loads data for a given rat and session.

    Args:
        data_path (str): Path to data directory
        rat_name (str): Name of the rat
        session (int): Session number (1 or 2)
        sweep_type (str): Type of sweep ("theta" or "replay")

    Returns:
        hse.AttrDict: Dictionary containing rat data
    """
    assert rat_name in RAT_NAMES, f"{rat_name} not in {RAT_NAMES}"
    assert session in [1,2]
    assert sweep_type in ["theta", "replay"]
    dir_path = os.path.join(data_path, rat_name, f"Open{session}")
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"{dir_path} not found")

    spike_file  = hseu.read_mat(os.path.join(dir_path, "Spike_Data.mat"))
    pos_file    = hseu.read_mat(os.path.join(dir_path, "Position_Data.mat"))
    ripple_file = hseu.read_mat(os.path.join(dir_path, "Ripple_Events.mat"))
    epochs      = hseu.read_mat(os.path.join(dir_path, "Epochs.mat"))
    well_seq    = hseu.read_mat(os.path.join(dir_path, "Well_Sequence.mat"))

    pos,spikes,ripples = __extract_runs(epochs, pos_file, spike_file, ripple_file, sweep_type)

    # Where are the well sequence data used
    # Where is significant_ripples found and where is it used
    # It looks like both are used when calculating descriptive stats and not before. Nignore for now.
    rat_data = RatData(
            rat_name,
            session,
            pos[:,0], 
            np.squeeze(pos[:,1:-1]),
            ripples,
            ripples[:,:2],
            np.array(spikes[:,1], dtype=int) - 1,
            spikes[:,0],
            np.array(spike_file['Excitatory_Neurons'], dtype=int) - 1,
            np.array(spike_file['Inhibitory_Neurons'], dtype=int) - 1,
            well_seq["Well_Sequence"],
            len(ripples),
            int(np.max(spikes[:,1])),
        )
    return rat_data


def clean_data(rat_data: RatData, position_gap_threshold_s: float = 0.25) -> RatData:
    """Cleans the given rat data by aligning spikes to positions and removing gaps in positions.

    Args:
        rat_data (RatData): Dictionary containing rat data
        position_gap_threshold_s (float): Threshold for position gaps in seconds. Defaults to 0.25.

    Returns:
        RatData: Dictionary containing cleaned rat data
    """
    rat_data = copy.deepcopy(rat_data)
    rat_data = __align_spikes_to_pos(rat_data)
    rat_data = __clean_gaps(rat_data, position_gap_threshold_s)
    return rat_data


def calculate_velocity(rat_data: RatData, velocity_run_threshold_s: float = 5.0) -> RatData:
    """Calculates the velocity of the rat and determines when the rat is running.

    Args:
        rat_data (RatData): Dictionary containing rat data
        velocity_run_threshold_s (float): Threshold for determining if the rat is running in meters per second. Defaults to 5.0.

    Returns:
        RatData: Dictionary containing rat data with additional fields for velocity and run times
    """
    pos_times = rat_data.pos_times_sec
    velocity_t = (pos_times[1:] + pos_times[:-1]) / 2
    velocity   = __calculate_velocity(rat_data.pos_xy_cm)

    nan_vel = np.isnan(velocity)
    if np.any(nan_vel):
        velocity_t = velocity_t[~nan_vel]
        velocity   = velocity[~nan_vel]
    run_b = velocity > velocity_run_threshold_s
    run_starts,run_ends = hseu.extract_times_from_boolean(run_b, velocity_t)

    rat_data.velocity_time_sec = velocity_t
    rat_data.velocity          = velocity
    rat_data.run_times_start   = run_starts
    rat_data.run_times_end     = run_ends

    return rat_data

def load_and_preprocess(
        data_path: str,
        rat_name: str, 
        session: int, 
        sweep_type: str = "theta", 
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        bin_size_cm: int = 4,
        place_field_gaussian_sd_cm: float = 4.0,
        time_window_ms: float = 3.0,
        time_window_advance_ms: Optional[float] = None,
        avg_fr_smoothing_convolution: np.ndarray = np.array([.25, .25, .25, .25]),
        avg_spikes_per_s_threshold: int = 2,
        min_popburst_duration_ms: int = 30
    ) -> RatData:
    """Runs all preprocessing steps on the given rat data.

    Args:
        data_path (str): Path to data directory
        rat_name (str): Name of the rat
        session (int): Session number (1 or 2)
        sweep_type (str): Type of sweep ("theta" or "replay")
        position_gap_threshold_s (float): Threshold for position gaps in seconds. Defaults to 0.25.
        velocity_run_threshold_s (float): Threshold for determining if the rat is running in meters per second. Defaults to 5.0.
        bin_size_cm (int): Bin size for place field calculation in centimeters. Defaults to 4.
        place_field_gaussian_sd_cm (float): Standard deviation of Gaussian for place field calculation in centimeters. Defaults to 4.0.

    Returns:
        RatData: Dictionary containing rat data with additional fields for velocity and place field data
    """
    rat_data = load_rat(data_path, rat_name, session, sweep_type)
    rat_data = clean_data(rat_data, position_gap_threshold_s)
    rat_data = calculate_velocity(rat_data, velocity_run_threshold_s)
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

    return rat_data