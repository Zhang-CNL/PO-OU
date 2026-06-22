import os
import numpy as np
import pynapple as nap

import hippocampalseq.utils as hseu
from .metadata import *

def load_spiking_data(
        data_path: str,
        rat_name: str,
        session: int,
        track_type: str = 'Linear',
        ripple_type: str = 'awake'
    ) -> Tuple[np.ndarray,...,nap.TsGroup]:
    """Load spiking data from rat folder.

    Args:
        data_path (str): Path to data directory.
        rat_name (str): Rat name.
        session (int): Session number.
        track_type (str, optional): Track type. Can be one of ("Open", "Linear") Defaults to 'Linear'.
        ripple_type (str, optional): Ripple epoch to be extracted. Can be one of ("awake", "rem", "sleep", "sleep_immobile"). Defaults to 'awake'.

    Returns:
        (np.ndarray): Time array.
        (np.ndarray): X-Position.
        (np.ndarray): Y-Position.
        (np.ndarray): Head-direction.
        (np.ndarray): Epoch starts selected from ripple type.
        (np.ndarray): Epoch ends selected from ripple type.
        (nap.TsGroup): Raw spike data.
    """
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

    ripple_starts = ripples[:,0]
    ripple_ends   = ripples[:,1]
    ripple_periods = nap.IntervalSet(start=ripple_starts, end=ripple_ends)

    unique_cells = np.unique(spike_ids)
    cell_spikes = {}
    for cell in unique_cells:
        spikes = spike_times[spike_ids == cell]
        cell_spikes[cell] = nap.Ts(t=np.sort(spikes))

    spike_data = nap.TsGroup(cell_spikes)

    return (
        time, 
        x, y, hd,
        starts,
        ends,
        spike_data
    )
