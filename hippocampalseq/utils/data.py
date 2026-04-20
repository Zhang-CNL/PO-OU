import numpy as np
from dataclasses import dataclass
from typing import List



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
    """
    Data corresponding to rat running.
    Positions (x,y,t) should be in a 1-1 alignment with (spike_time, spike_id).
    """
    run_starts   : np.ndarray # (N,1): Time that the rat started running
    run_ends     : np.ndarray # (N,1): Time that the rat stopped running
    x            : np.ndarray # (T,1): x position of rat at corresponding time
    y            : np.ndarray # (T,1): y position of rat at corresponding time
    time         : np.ndarray # (T,1): Time that the rat was at corresponding position
    spike_ids    : np.ndarray # (T,1): id of the cell that spiked at corresponding time
    spike_times  : np.ndarray # (T,1): time that the cell spiked at corresponding position
    velocity     : np.ndarray
    velocity_t   : np.ndarray
    cell_spikes  : Dict[int, np.ndarray]
    unique_cells : np.ndarray

@dataclass 
class RatData:
    rat_name           : str            = None
    session            : str            = None
    track_type         : str            = None
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