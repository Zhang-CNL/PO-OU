import numpy as np
from dataclasses import dataclass

RAT_NAMES = ['Harpy', 'Imp', 'Janni', 'Naga']
PFEIFFER_ENV_WIDTH_CM  = 200
PFEIFFER_ENV_HEIGHT_CM = 200
PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC = 2
PFEIFFER_RECORDING_FPS = 1 / 30


@dataclass
class PlacefieldData:
    place_fields    : np.ndarray = None # (N, K, K): Number of cells and each position corresponds to excitation strength in real environment.
    place_field_mat : np.ndarray = None # (N, K^2): Flattened place fields
    mean_firing_rate: np.ndarray = None # (N, 1): Mean firing rate for each cell
    max_firing_rate : np.ndarray = None # (N, 1): Max firing rate for each cell
    place_cell_ids  : np.ndarray = None # (n_place_cells, 1): IDs of place cells
    n_place_cells   : int        = None

@dataclass
class RippleData:
    spikemats_ripple  : dict[int,np.ndarray] = None # (Nripples, T, n_place_cells): Spikes in ripple time window for each place cell
    spikemats_popburst: dict[int,np.ndarray] = None # (Nripples, T_sub, n_place_cells): Spikes in population burst time window for each place cell
    popburst_time_s   : np.ndarray           = None # (Nripples, 2): Start and end time of population burst
    avg_sps_smoothed  : dict[int,np.ndarray] = None # (Nripples, T): Average spikes per second for each time bin across all cells
    mean_popburst_arr : np.ndarray           = None # (n_place_cells, 1): Mean firing rate for each place cell across all time
    mean_popburst_mat : np.ndarray           = None # (n_place_cells, Nripples): Mean firing rate for each place cell for each ripple
    firing_rate_scale : dict                 = None # Gamma scaling factors considered for each place cell's gamma prior.

@dataclass
class Theta:
    run_times_s: np.ndarray 
    true_trajectories: dict[int,np.ndarray]
    spikemats: dict[int,np.ndarray]

@dataclass 
class RatData:
    rat_name           : str            = None
    session            : int            = None
    pos_times_sec      : np.ndarray     = None # (T, 1): time that the animal was at corresponding position in pos_xy_cm
    pos_xy_cm          : np.ndarray     = None # (T, 2): (x,y) position animal was in at corresponding time in pos_times_sec
    ripple_info        : np.ndarray     = None # (Nripples, 6): Ripple start time, ripple end time, ripple peak time, raw ripple power, z-scored ripple power across whole recording, z-scored ripple power across epoch
    ripple_times_sec   : np.ndarray     = None # (Nripples, 2): Ripple start and end times
    spike_ids          : np.ndarray     = None # (T, 1): id of the cell that spiked at corresponding time in spike_times_sec
    spike_times_sec    : np.ndarray     = None # (T, 1): time that the cell spiked at corresponding position in pos_xy_cm
    excitatory_neurons : np.ndarray     = None # List of IDs corresponding to excitatory neurons in spike_ids
    inhibitory_neurons : np.ndarray     = None # List of IDs corresponding to inhibitory neurons in spike_ids
    well_sequence      : np.ndarray     = None
    n_ripples          : int            = None
    n_cells            : int            = None
    large_position_gaps: np.ndarray     = None
    velocity_time_sec  : np.ndarray     = None # (T-1, 1): time that rat hit a certain velocity corresponding to velocity
    velocity           : np.ndarray     = None # (T-1, 1): velocity across x and y corresponding to time in velocity_time_sec
    run_times_start    : np.ndarray     = None # (?, 1): time at which rat hit a velocity above the set threshold
    run_times_end      : np.ndarray     = None # (?, 1): time at which rat dropped below velocity at set threshold
    place_field_data   : PlacefieldData = None 
    ripple_data        : RippleData     = None
    theta_data         : Theta          = None