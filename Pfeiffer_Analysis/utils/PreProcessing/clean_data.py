import pynapple as nap
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def integrate_spike_position(position_data, spike_data, excitatory_neurons, 
                            inhibitory_neurons,
                            minimum_time_difference=np.inf, plot=False):
    
    '''
    From MATLAB: Minimumm time difference is the minimum time (seconds) that a spike
    can be from the closest frame of the position data to be kept.  Used to eliminate
    spikes that occur when there is no position data avialable (such as if video aquisition
    system suddenly stops working). If inf = all spikes kept
    
    '''
        
    spike_info = {}
    spike_times_filtered = {}
    
    total_spikes = sum(len(spike_data[c]) for c in spike_data.keys())
    #print(f"total_spikes before integration: {total_spikes:,}")
    kept_spikes = 0
    max_time_diff = 0
    
    # Get position time and values as arrays
    pos_times = position_data.index.values
    #print(f"position times restricted to running epoch: {np.shape(pos_times)}")
    #print(f"position start: {pos_times[0]}, with end: {pos_times[-1]}")
    x_vals = position_data['x'].values
    y_vals = position_data['y'].values
    hd_vals = position_data['head_direction'].values
    vel_vals = position_data['velocity'].values
    
    t0 = pos_times[0]
    t1 = pos_times[-1]
    
    for cell_id in spike_data.keys():
        
        spike_times = spike_data[cell_id].index.values
        #print(f"spike times with length: {np.shape(spike_times)} for cell with id: {cell_id}")
        
        # Find nearest position sample for each spike
        # find index where time difference is minimum
        indices_searchsorted = np.searchsorted(pos_times, spike_times, side="left")
        indices_digitize = np.digitize(spike_times, pos_times)

        indices = indices_searchsorted
                
        # Consider both neighbors (before and after insertion point)
        # clip prevents indices from going out of bounds, 
        # otherwise you can just use ind_prev = indices -1 and indx_curr = indices
        idx_prev = np.clip(indices-1, 0, len(pos_times) - 1)
        idx_curr = np.clip(indices, 0, len(pos_times)-1)
        
        #check previous and current index to determine closest neighbor
        #This way is more careful and closest to Brad's I think
        #Calculate time differences to both neighbors
        dt_prev = np.abs(spike_times - pos_times[idx_prev])
        dt_cur = np.abs(spike_times - pos_times[idx_curr])
        #creates boolean array for if dt_prev is less than dt_cur
        prev = dt_prev <= dt_cur
        #np.where(condition, x, y), if true, yield x otherwise y 
        #returns elements chosen from x or y depending on condition
        #Choose the closer one
        nearest_neighbor = np.where(prev, idx_prev, idx_curr)
        time_diff = np.where(prev, dt_prev, dt_cur)
        
        # Filter by time threshold
        valid = time_diff <= minimum_time_difference

        sel_idx = nearest_neighbor[valid]
        sel_times = spike_times[valid]

        x_at_spikes = x_vals[sel_idx]
        y_at_spikes = y_vals[sel_idx]
        vel_at_spikes = vel_vals[sel_idx]
        time_diffs_sel = time_diff[valid]
        
        if cell_id == 1 and plot==True:
            # This will be lopsided for the most part because it is not organized by velocity
            # print(f"spike sorting using searchsorted with length: {np.shape(indices_searchsorted)}")
            #print(f"spike sorting with digitize with length: {np.shape(indices_digitize)}")
            fig, axes = plt.subplots(1,2, figsize=(20, 8))
            
            # Histogram of differences
            diff = indices_digitize - indices_searchsorted
            axes[0].hist(diff, bins=50, edgecolor='black')
            axes[0].set_xlabel('digitize - searchsorted')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'Index Differences (mean={diff.mean():.3f})')
            
            # Temporal distribution of indices
            axes[1].hist(indices_searchsorted, bins=100, alpha=0.5, label='searchsorted', color='blue')
            axes[1].hist(indices_digitize, bins=100, alpha=0.5, label='digitize', color='red')
            axes[1].set_xlabel('Position index')
            axes[1].set_ylabel('Spike count')
            axes[1].set_title('Distribution of matched position indices')
            axes[1].legend()
            
            #print("Digitize and Searchsorted are the same for this case since it is 1D so use searchsorted ")

            fig2, ax2 = plt.subplots(1,3, figsize=(20, 8))
            ax2[0].hist(time_diff, bins=100, edgecolor='black')
            ax2[0].set_xlabel('Time difference (s)')
            ax2[0].set_ylabel('Count')
            ax2[0].set_title(f'Spike-Position Time Differences\n(max={time_diff.max():.4f}s)')
            ax2[0].axvline(x=0.1, color='r', linestyle='--', label='100ms threshold')
            ax2[0].legend()
            
            #Comparison of raw searchsorted vs nearest neighbor
            ax2[1].scatter(indices, nearest_neighbor, alpha=0.5, s=1)
            ax2[1].plot([0, len(pos_times)], [0, len(pos_times)], 'r--', label='y=x')
            ax2[1].set_xlabel('Raw searchsorted index')
            ax2[1].set_ylabel('Nearest neighbor index')
            ax2[1].set_title('Effect of nearest neighbor correction')
            ax2[1].legend()
            
            #Position coverage
            matched_positions = pos_times[nearest_neighbor]
            ax2[2].hist2d(spike_times, matched_positions, bins=100, cmap='Blues')
            ax2[2].plot([pos_times[0], pos_times[-1]], [pos_times[0], pos_times[-1]], 'r--', alpha=0.5)
            ax2[2].set_xlabel('Spike time (s)')
            ax2[2].set_ylabel('Matched position time (s)')
            ax2[2].set_title('Spike-Position Time Matching')

        # Store info
        if cell_id in excitatory_neurons:
            cell_type = 1
        elif cell_id in inhibitory_neurons:
            cell_type = 0
        else:
            cell_type = -1
            
        ## YOu can store everything in a dictionary if you want
        # but pynapple is specifically made for time-based slicing and neural data
        # so to keep it consistent with position, 
        # I used TsdFrame. I previously used dictionary
        
        # spike_info[cell_id] = {
        #     "spike_times": sel_times,
        #     "x": x_at_spikes,
        #     "y": y_at_spikes,
        #     "velocity": vel_at_spikes,
        #     "time_diff": time_diffs_sel,
        #     "cell_type": cell_type}
        
        ##This makes sure cell_type is the same shape as sel_times
        cell_types = np.full(sel_times.shape,cell_type)
        spike_info[cell_id] = nap.TsdFrame(
            t=sel_times,
            d=np.c_[
            x_at_spikes,
            y_at_spikes,
            vel_at_spikes,
            time_diffs_sel,
            cell_types],
        columns=['x', 'y', 'velocity', 'time_diff', 'cell_type'])
        
        spike_times_filtered[cell_id] = nap.Ts(t=sel_times)
        kept_spikes += sel_times.size
            
    spike_times_filtered = nap.TsGroup(spike_times_filtered)
    #spike_info = nap.TsGroup(spike_info)
    
    print(f'{kept_spikes:,} out of {total_spikes:,} spikes kept')
    
    return spike_info, spike_times_filtered

def remove_noisy_epochs(position_data, spike_data, spike_info, lfp_data, 
                        animal, experiment, timepoints_to_remove):
    
    #This comes from sections specified in Brad's initial variables.
    #If the noisy epochs do not overlap with running data, it will remain the same.

    
    # Get session-specific noisy epochs
    session_key = f"{animal.lower()}_{experiment.lower()}"
    
    if session_key in timepoints_to_remove:
        noisy_epochs = timepoints_to_remove[session_key]
        
        # print(f'Removing noisy epochs for {animal} {experiment}')
        # print(f'Number of noisy sections: {len(noisy_epochs)}')
        # print(f'Noisy epochs: {noisy_epochs}')
        
        total_support = position_data.time_support
        clean_epochs = total_support.set_diff(noisy_epochs)

        # print(f"Total duration before: {total_support.tot_length('s'):.1f}s")
        # print(f"Noisy duration:{noisy_epochs.tot_length('s'):.1f}s")
        # print(f"Clean duration:{clean_epochs.tot_length('s'):.1f}s")

        
        # Restrict all data to clean epochs
        position_clean = position_data.restrict(clean_epochs)
        spike_data_clean = spike_data.restrict(clean_epochs)
                
        spike_info_clean = {}
        for cell_id, info in spike_info.items():
            info_clean = info.restrict(clean_epochs)
            if len(info_clean) > 0:
                spike_info_clean[cell_id] = info_clean
                
    
        lfp_data_clean = {
            "filtered_lfp": lfp_data["filtered_lfp"].restrict(clean_epochs),
            "amplitude": lfp_data["amplitude"].restrict(clean_epochs),
            "power": lfp_data["power"].restrict(clean_epochs),
            "phase": lfp_data["phase"].restrict(clean_epochs),
            "raw_lfp": lfp_data["raw_lfp"].restrict(clean_epochs),
            "sampling_rate": lfp_data["sampling_rate"],
            "run_interval": clean_epochs,
            "metadata": lfp_data["metadata"].copy(),
        }
        lfp_data_clean["metadata"]["clean_epochs"] = clean_epochs

        total_duration = clean_epochs.tot_length("s")

        
    else:
        print(f'No noisy epochs to remove for {animal} {experiment}')
        position_clean = position_data
        spike_data_clean = spike_data
        spike_info_clean = spike_info
        lfp_data_clean = lfp_data
        total_duration = position_data.time_support.tot_length('s')
        
    return position_clean, spike_data_clean, spike_info_clean, lfp_data_clean, total_duration


def restrict_to_movement(spike_info, lfp_data, velocity_cutoff=5):
#     """
#     So I kind of kept this function from Brad's pipeline
#     because when I load the data, I already filter it to make sure the times are overlapping 
#       and within the running epoch.
#     but didn't do a velocity cutoff, I will do so now.
#     """

    spike_info_moving = {}
    all_spike_times = []

    for cell_id, info in spike_info.items():
        vel_mask = info['velocity'].values >= velocity_cutoff
        moving_info = info[vel_mask]
        spike_info_moving[cell_id] = moving_info
        all_spike_times.append(moving_info.index.values)

    all_spike_times = np.concatenate(all_spike_times)
    first_spike = all_spike_times.min()
    last_spike = all_spike_times.max()
    spike_interval = nap.IntervalSet(start=first_spike, end=last_spike)
    

    lfp_moving = {
        'filtered_lfp': lfp_data['filtered_lfp'].restrict(spike_interval),
        'amplitude':    lfp_data['amplitude'].restrict(spike_interval),
        'power':        lfp_data['power'].restrict(spike_interval),
        'phase':        lfp_data['phase'].restrict(spike_interval),
        'raw_lfp':      lfp_data['raw_lfp'].restrict(spike_interval),
        'sampling_rate': lfp_data['sampling_rate'],
        'run_interval':  spike_interval,
        'metadata':      lfp_data['metadata'].copy()}
    
    return spike_info_moving, lfp_moving
