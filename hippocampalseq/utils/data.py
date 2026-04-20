import numpy as np
from dataclasses import dataclass
from typing import List

RAT_NAMES = ['Harpy', 'Imp', 'Janni', 'Naga']
PFEIFFER_ENV_WIDTH_CM  = 200
PFEIFFER_ENV_HEIGHT_CM = 200
PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC = 1
PFEIFFER_RECORDING_FPS = 1 / 30

PFEIFFER_NOISY_EPOCHS = {
    'Janni': {
        "Linear2": {
            "starts": [18721, 23511],
            "ends"  : [22773, 29423]
        },
        "Linear3": {
            "starts": [11650, 16390],
            "ends"  : [15498, 20184]
        },
        "Open2": {
            "starts": [33756],
            "ends"  : [34007]
        }
    },
    "Harpy": {
        "Linear1": {
            "starts": [12850, 17880],
            "ends"  : [12956, 17929]
        },
        "Linear2": {
            "starts": [19307, 19476],
            "ends"  : [19322, 19489]
        },
        "Linear3": {
            "starts": [27025],
            "ends"  : [27035]
        },
        "Open1": {
            "starts": [27332],
            "ends"  : [27639]
        },
        "Open2": {
            "starts": [19528, 19582, 19701, 20802, 21607, 21690, 21701, 22141, 22258],
            "ends"  : [19539, 19592, 19722, 20815, 21621, 21696, 21702, 22180, 22265]
        }
    },
    "Imp": {
        "Linear1": {
            "starts": [25880, 30570, 33920],
            "ends"  : [29735, 33885, 33962]
        },
        "Open1": {
            "starts": [25160],
            "ends"  : [25275]
        },
        "Open2": {
            "starts": [20122, 20147],
            "ends"  : [20126, 20164]
        }       
    }
}

@dataclass
class PlacefieldData:
    place_fields    : np.ndarray = None # (n_place_cells, K, K): Number of cells and each position corresponds to excitation strength in real environment.
    mean_firing_rate: np.ndarray = None # (n_place_cells, 1): Mean firing rate for each cell
    max_firing_rate : np.ndarray = None # (n_place_cells, 1): Max firing rate for each cell
    place_cell_ids  : np.ndarray = None # (n_place_cells, 1): IDs of place cells
    n_place_cells   : int        = None

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
class Theta:
    run_starts        : np.ndarray
    run_ends          : np.ndarray
    true_trajectories : List[np.ndarray] # (T, 2): (x,y) position animal was in at corresponding time in pos_times_sec
    spikemats         : List[np.ndarray] # (T, n_place_cells): Spikes in ripple time window for each place cell

@dataclass 
class RawData:
    time           : np.ndarray # (T, 1): time that the animal was at corresponding position in (x,y)
    x              : np.ndarray # (T,1)
    y              : np.ndarray # (T,1)
    head_direction : np.ndarray
    spike_ids      : np.ndarray # (T, 1): id of the cell that spiked at corresponding time in spike_times
    spike_times    : np.ndarray # (T, 1): time that the cell spiked at corresponding position in (x,y)
    raw_ripples    : np.ndarray # (Nripples, 6): Ripple start time, ripple end time, ripple peak time, raw ripple power, z-scored ripple power across whole recording, z-scored ripple power across epoch
    unique_cells   : np.ndarray
    cell_spikes    : List[np.ndarray]

@dataclass
class RunningData:
    run_starts : np.ndarray
    run_ends   : np.ndarray
    x          : np.ndarray
    y          : np.ndarray
    time       : np.ndarray
    spike_ids  : np.ndarray
    spike_times: np.ndarray
    velocity   : np.ndarray
    velocity_t : np.ndarray

@dataclass 
class RatData:
    rat_name           : str            = None
    session            : str            = None
    raw_data           : RawData        = None
    run_data           : RunningData    = None
    excitatory_neurons : np.ndarray     = None # List of IDs corresponding to excitatory neurons in spike_ids
    inhibitory_neurons : np.ndarray     = None # List of IDs corresponding to inhibitory neurons in spike_ids
    well_sequence      : np.ndarray     = None
    n_ripples          : int            = None
    n_cells            : int            = None
    large_position_gaps: np.ndarray     = None
    place_field_data   : PlacefieldData = None 
    ripple_data        : RippleData     = None
    theta_data         : Theta          = None