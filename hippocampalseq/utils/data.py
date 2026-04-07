import numpy as np
from dataclasses import dataclass
from typing import List

RAT_NAMES = ['Harpy', 'Imp', 'Janni', 'Naga']
PFEIFFER_ENV_WIDTH_CM  = 200
PFEIFFER_ENV_HEIGHT_CM = 200
PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC = 2
PFEIFFER_RECORDING_FPS = 1 / 30

@dataclass
class RunData:
    spike_times : np.ndarray = None 
    spike_ids   : np.ndarray = None
    x           : np.ndarray = None 
    y           : np.ndarray = None 

@dataclass
class PlacefieldData:
    place_fields    : np.ndarray = None # (n_place_cells, K, K): Number of cells and each position corresponds to excitation strength in real environment.
    place_field_mat : np.ndarray = None # (n_place_cells, K^2): Flattened place fields
    mean_firing_rate: np.ndarray = None # (n_place_cells, 1): Mean firing rate for each cell
    max_firing_rate : np.ndarray = None # (n_place_cells, 1): Max firing rate for each cell
    place_cell_ids  : np.ndarray = None # (n_place_cells, 1): IDs of place cells
    n_place_cells   : int        = None
    run_data        : RunData    = None

@dataclass
class RippleSpike:
    spikemat         : np.ndarray = None # (T, n_place_cells): Spikes in ripple time window for each place cell
    spikemat_popburst: np.ndarray = None # (T_sub, n_place_cells): Spikes in population burst time window for each place cell
    avg_sps_smoothed : np.ndarray = None # (T,): Average spikes per second for each time bin across all cells

@dataclass
class RippleData:
    popburst_time_s  : np.ndarray        = None # (Nripples, 2): Start and end time of population burst
    mean_popburst_arr: np.ndarray        = None # (n_place_cells, 1): Mean firing rate for each place cell across all time
    mean_popburst_mat: np.ndarray        = None # (n_place_cells, Nripples): Mean firing rate for each place cell for each ripple
    firing_rate_scale: dict              = None # Gamma scaling factors considered for each place cell's gamma prior.
    ripple_spikes    : List[RippleSpike] = None # (Nripples): List of RippleSpike objects

@dataclass
class ThetaSpikes:
    true_trajectory: np.ndarray = None # (T, 2): (x,y) position animal was in at corresponding time in pos_times_sec
    spikemat       : np.ndarray = None # (T, n_place_cells): Spikes in ripple time window for each place cell

@dataclass
class Theta:
    run_times_s : np.ndarray 
    theta_spikes: List[ThetaSpikes]

@dataclass 
class RawData:
    time           : np.ndarray # (T, 1): time that the animal was at corresponding position in (x,y)
    x              : np.ndarray # (T,1)
    y              : np.ndarray # (T,1)
    head_direction : np.ndarray
    velocity       : np.ndarray # (cm/sec)
    velocity_time  : np.ndarray
    run_starts     : np.ndarray
    run_ends       : np.ndarray
    spike_ids      : np.ndarray # (T, 1): id of the cell that spiked at corresponding time in spike_times
    spike_times    : np.ndarray # (T, 1): time that the cell spiked at corresponding position in (x,y)
    raw_ripples    : np.ndarray # (Nripples, 6): Ripple start time, ripple end time, ripple peak time, raw ripple power, z-scored ripple power across whole recording, z-scored ripple power across epoch

@dataclass
class RunSpikes:
    time        : np.ndarray
    x           : np.ndarray
    y           : np.ndarray 
    spike_times : np.ndarray
    spike_ids   : np.ndarray
    run_starts  : np.ndarray
    run_ends    : np.ndarray

@dataclass 
class RatData:
    rat_name           : str            = None
    session            : int            = None
    raw_data           : RawData        = None
    run_spikes         : RunSpikes      = None
    excitatory_neurons : np.ndarray     = None # List of IDs corresponding to excitatory neurons in spike_ids
    inhibitory_neurons : np.ndarray     = None # List of IDs corresponding to inhibitory neurons in spike_ids
    well_sequence      : np.ndarray     = None
    n_ripples          : int            = None
    n_cells            : int            = None
    large_position_gaps: np.ndarray     = None
    run_times_start    : np.ndarray     = None # (?, 1): time at which rat hit a velocity above the set threshold
    run_times_end      : np.ndarray     = None # (?, 1): time at which rat dropped below velocity at set threshold
    place_field_data   : PlacefieldData = None 
    ripple_data        : RippleData     = None
    theta_data         : Theta          = None