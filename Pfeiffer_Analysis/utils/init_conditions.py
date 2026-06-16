def init_conditions():
    import pynapple as nap
    import numpy as np
    
    """
    These are defined from Brad's codes.  He defines variables such as 
    forward/backward theta lfp window, 
    timepoints to remove,
    bin sizes, etc
    
    For simplicity for now, I will just define these exactly as it is done in Matlab
    
    
    For some experiments, there are brief sections with extremely high noise
    that needed to be removed to properly analyze the data.  Here are the
    sections that were removed (and sometimes, why):
    
    For Janni Open 2, one chip came unplugged in the middle of the experiment
    and those timepoints need to be removed from the analysis
    Open 2: 33756-34007
    
    For Harpy, there was an issue with the ground wire that very rarely
    caused the LFP to be super noisy.  Here are the time points that need to
    be eliminated from analysis to remove those noisy epochs
    Linear 1: 12850-12956, 17880-17929 (many more short bursts of noise -- likely when he was at the well)
    Linear 2: 19307-19322, 19476-19489
    Linear 3: 27025-27035
    Open 1: 27332-27639
    Open 2: 19528-19539, 19582-19592, 19701-19722, 20802-20815, 21607-21621, 21690-21696, 21701-21702, 22141-22180, 22258-22265
    
    For Imp, there were a few noisy sections that need to be removed as well.
    Linear 1: 33920-33962 (many more short bursts of noise -- likely when he was at the well)
    Open 1: 25160-25275 (and maybe 23450-23569 although it's not that bad)
    Open 2: 20122-20126, 20147-20164
    """
    
        # Initial analysis parameters
    initial_variables = {
        # Spike and position integration parameters
        'spike_position_integration_minimum_time_difference': np.inf,  # seconds
        
        # Place field parameters
        'bin_size': 2,  # cm
        'place_field_velocity_cutoff': 5,  # cm/s
        'place_field_firing_rate_cutoff': 1,  # Hz
                
        # Theta filtering parameters
        'limit_analysis_by_theta_length': 1,  # If set to 1, theta oscillations greater or less than a maximum (defined by Theta_Length_Min_Max) will be excluded from analysis; if set to 0, all theta oscillations will be included (although a velocity limit will still be applied)
        'theta_length_min_max': [0.08, 0.16],  # seconds (trough-to-trough)
        
        # Statistical parameters
        'number_of_shuffles': 500,
        'number_of_shuffles_for_phase_position_relationship': 1000,
        
        # Phase analysis parameters
        'phase_bin': 10,  # degrees
        'gaussian_smoothing_sigma': 12,  # degrees
        'rayleigh_test_p_value_cutoff': 0.05,
        
        # Place field definition parameters
        'minimum_place_field_firing_rate_fraction': 0.2,
        'minimum_contiguous_place_field_bins': 20,
        
        # Velocity and spike count thresholds
        'velocity_cutoff': 10,  # cm/s minimum for theta locking
        'minimum_spike_count': 100, #A neuron has to fire at least this many spikes during running to be included in the analysis
        
        # Bayesian decoding parameters
        'decoding_time_window': 0.02,  # seconds Bayesian decoding time window
        'decoding_time_advance': 0.005,  # seconds, size of the advancement between adjacent Bayesian decoding window frames
        
        # Sequence analysis parameters
        'sequence_score_distance': 10,  # Distance (in cm) from best fit line to calculate the Sequence Score
        'use_maximum_posterior_probability': 1,  # If set to 1, use the maximum posterior probability for identifying the single point for each frame (a value of 0 means use the weighted mean)
        'maximum_step_size': 10,  # The maximum distance (cm) the posterior probability can move between consecutive decoding windows and still be considered part of the same spatial trajectory
        'minimum_step_size': 2,  # The minimum distance (cm) the posterior probability can move between consecutive decoding windows and still be considered part of a real spatial trajectory
        'minimum_posterior_probability': 0.05,
        'minimum_step_number': 5,
        'start_to_end_distance': 10,  # cm
        
        # Theta sequence windows (in degrees) - these will change
        'forward_window': [250, 60],   # wraps around 360°
        'reverse_window': [80, 230],   # no wrap
    }
    
    # Using pynapple IntervalSet for easy time-based operations
    # These are all the same as from Brad's code
    timepoints_to_remove = {
        'janni_open2': nap.IntervalSet(start=[33756], end=[34007]),
        'janni_linear2': nap.IntervalSet(start=[18721, 23511], end=[22773, 29423]),
        'janni_linear3': nap.IntervalSet(start=[11650, 16390], end=[15498, 20184]),
        
        'harpy_linear1': nap.IntervalSet(start=[12850, 17880], end=[12956, 17929]),
        'harpy_linear2': nap.IntervalSet(start=[19307, 19476], end=[19322, 19489]),
        'harpy_linear3': nap.IntervalSet(start=[27025], end=[27035]),
        'harpy_open1': nap.IntervalSet(start=[27332], end=[27639]),
        'harpy_open2': nap.IntervalSet(start=[19528, 19582, 19701, 20802, 21607, 21690, 21701, 22141, 22258],
                                        end=[19539, 19592, 19722, 20815, 21621, 21696, 21702, 22180, 22265]),
        
        'imp_linear1': nap.IntervalSet(start=[25880, 30570, 33920], end=[29735, 33885, 33962]),
        'imp_open1': nap.IntervalSet(start=[25160], end=[25275]),
        'imp_open2': nap.IntervalSet(start=[20122, 20147], end=[20126, 20164])
    }
    
    # Theta phase windows for forward/reverse sweeps and peaks
    # Phase windows in degrees
    bimodal_windows = {
        'forward_window': [250, 60],  # Forward sweep phase window
        'reverse_window': [80, 230],  # Reverse sweep phase window
        'major_peak_window': [200, 70],  # Major peak window
        'minor_peak_window': [80, 190]  # Minor peak window
    }
    
    return initial_variables, timepoints_to_remove, bimodal_windows
    
    