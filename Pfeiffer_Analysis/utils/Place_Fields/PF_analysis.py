from utils.Place_Fields.get_place_fields import  find_PFs 
from utils.Place_Fields.calculate_decoding import find_decoding_error

def analyze_place_fields(initial_variables, data, preprocessed, 
                        bin_size=2, velocity_cutoff=5, firing_rate_cutoff=1):
    """Compute place fields and decoding error."""
    
    pf_results = find_PFs(
        initial_variables, data['position'], 
        preprocessed['spike_info_original'],
        preprocessed['spike_times_filtered'],
        bin_size=bin_size,
        velocity_cutoff=velocity_cutoff,
        firing_rate_cutoff=firing_rate_cutoff,
        track_type='linear'
    )
    
    decoding_error = find_decoding_error(
        preprocessed['position'],
        preprocessed['spike_info_moving'],
        pf_results,
        data['excitatory'], data['inhibitory'],
        bin_size=bin_size,
        velocity_threshold=10,
        time_bin_size=0.25,
        time_step=0.25
    )
    
    return {
        'pf_results': pf_results,
        'decoding_error': decoding_error
    }