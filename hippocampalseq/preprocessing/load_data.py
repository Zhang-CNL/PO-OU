import os
import numpy as np
import warnings

from .metadata import *
from .segment_runs import *
import hippocampalseq.utils as hseu

def filter_noisy(raw_data: hseu.RatData, rat_name: str, session: str) -> hseu.RatData:
    if rat_name in PFEIFFER_NOISY_EPOCHS:
        rat_sessions = PFEIFFER_NOISY_EPOCHS[rat_name]
        if session in rat_sessions:
            starts = rat_sessions[session]['starts']
            ends = rat_sessions[session]['ends']

            spike_mask = ~hseu.create_interval_mask(len(raw_data.spike_times), starts, ends)
            pos_mask   = ~hseu.create_interval_mask(len(raw_data.time), starts, ends)

            raw_data.time = raw_data.time[pos_mask]
            raw_data.x = raw_data.x[pos_mask]
            raw_data.y = raw_data.y[pos_mask]
            raw_data.spike_times = raw_data.spike_times[spike_mask]
            raw_data.spike_ids = raw_data.spike_ids[spike_mask]

    return raw_data

def load_clean_data(
        data_path: str,
        rat_name: str, 
        session: int,
        track_type: str = 'Linear',
        ripple_type: str = 'awake',
        position_gap_threshold_s: float = 0.25,
        velocity_run_threshold_s: float = 5.0,
        drop_misaligned: bool = False
    ) -> hseu.RatData:
    assert rat_name in RAT_NAMES, f"{rat_name} not in {hseu.RAT_NAMES}"
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
    print(rt)
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
    cell_spikes = {}
    for cell in unique_cells:
        spikes = spike_times[spike_ids == cell]
        cell_spikes[cell] = np.sort(spikes)

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
        track_type         = track_type,
        raw_data           = raw_data,
        run_data           = run_data,
        excitatory_neurons = excitatory_neurons,
        inhibitory_neurons = inhibitory_neurons,
        well_sequence      = well_seq,
        n_ripples          = len(ripples),
        n_cells            = np.max(spike_ids) + 1
    )
