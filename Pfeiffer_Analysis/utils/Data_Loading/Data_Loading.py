from utils.Data_Loading.load_data import load_lfp_data, load_position_data, split_multi_session_data

def load_all_data(ani, expt, basepath_position, base_path_lfp, csc, split_sessions=False):
    """Load position, spike, and LFP data.
    Added a flag to change if you want to split multi-session recordings into their own trajectories or now.
    This will give fewer spikes per cell which may have less statistical power or less clear decoding."""
    
    position_data, spike_data, excitatory_neurons, inhibitory_neurons, running_epoch = load_position_data(
        basepath=basepath_position,
        animal=ani,
        experiment=expt
    )
    
    lfp_data = load_lfp_data(
        position_data=position_data,
        spike_data=spike_data,
        basepath=base_path_lfp,
        animal=ani,
        experiment=expt,
        csc_name=csc
    )
    
    data = {
        'position':      position_data,
        'spikes':        spike_data,
        'lfp':           lfp_data,
        'excitatory':    excitatory_neurons,
        'inhibitory':    inhibitory_neurons,
        'running_epoch': running_epoch,
    }

    if split_sessions:
        return split_multi_session_data(data, expt)
    return data
