from utils.PreProcessing.clean_data import remove_noisy_epochs, restrict_to_movement, integrate_spike_position

def preprocess_data(data, ani, expt, timepoints_to_remove, velocity_cutoff=5):
    """Clean data and restrict to movement periods."""
    
    # Integrate spikes with position
    spike_info, spike_times_filtered = integrate_spike_position(
        data['position'], data['spikes'], 
        data['excitatory'], data['inhibitory'],
        minimum_time_difference=0.1
    )
    
    # Remove noisy epochs
    position_clean, spikes_clean, spike_info_clean, lfp_clean, total_duration = remove_noisy_epochs(
        data['position'], spike_times_filtered, spike_info, 
        data['lfp'], ani, expt, timepoints_to_remove
    )
    
    # Restrict to movement
    spike_info_moving, lfp_moving = restrict_to_movement(
        spike_info_clean, lfp_clean, velocity_cutoff=velocity_cutoff
    )
    
    return {
        'position': position_clean,
        'spikes': spikes_clean,
        'spike_info': spike_info_clean,
        'spike_info_moving': spike_info_moving,
        'spike_times_filtered': spike_times_filtered,
        'lfp': lfp_clean,
        'lfp_moving': lfp_moving,
        'total_duration': total_duration,
        'spike_info_original': spike_info
    }
