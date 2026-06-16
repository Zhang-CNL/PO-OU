from scipy.special import logsumexp
from pathlib import Path
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

def find_decoding_error(position_data, spike_info_moving, field_data,
                                excitatory_neurons, inhibitory_neurons,
                                bin_size=2, velocity_threshold=5,
                                time_bin_size=0.25, time_step=0.25):
    
    # print(f"The excitatory neurons are: {excitatory_neurons}")
    # print(f"The inhibitory neurons are: {inhibitory_neurons}")

    Field_Data = field_data['Field_Data']
    #As a reminder of Field_Data shape:
    #print(f"Field_Data shape is y bins, x bins, cells (not parsed by exc/inh yet): {Field_Data.shape}")
    bin_size = field_data['bin_size']
    cell_ids = field_data['cell_ids']
    x_edges = field_data["x_bin_edges"]         # 1D array, length nx+1
    y_edges = field_data["y_bin_edges"]         # 1D array, length ny+1
    ny,nx, n_cells = Field_Data.shape
    #print(f"Field_Data shape (ny, nx, n_cells): {Field_Data.shape}")
    
    #This should already be restricted to run data from load_data.py but not filtered by speed
    post = position_data.index.values
    xpos = position_data['x'].values
    ypos = position_data['y'].values
    vel = position_data['velocity'].values
    
    #Filter by speed
    moving_mask = vel >= velocity_threshold
    xpos_run = xpos[moving_mask]
    ypos_run = ypos[moving_mask]
    t_run = post[moving_mask]
        
    # Replace zeros with small values
    # Otherwise, any field with zero prevents algorithm from representing that location
    fields_modified = Field_Data.copy()
    for i in range(n_cells):
        field = fields_modified[:,:, i]
        positive = field[field>0]
        if positive.size > 0:
            minimum = positive.min()
            eps = minimum / 10 if minimum / 10 > 0 else minimum
            field[field <= 0] = eps
        else:
            field[:] = 1.0
        fields_modified[:,:, i] = field
        
    
    # Get excitatory cells only
    #Finds excitatory cells in cell_ids from field data
    #You can modify the field_data as well but it's kind of redundant
    #exc_indices = [i for i, cid in enumerate(cell_ids) if cid in excitatory_neurons]    #where excitatory neuron is in field_Data
    #fields_exc = fields_modified[:,:, exc_indices]

    
    # Precompute sum of fields for exp(-T * sum(f_i(pos)))
    #sum_fields = fields_modified.sum(axis=2)  # shape (ny, nx), inh+exc
    
    # Filter spike_info_moving to excitatory only
    #This has already been filtered by speed in restrict_to_movement function in clean_data.py
    exc_cell_ids =[cid for cid in cell_ids if cid in excitatory_neurons]
    #This maps to the field data
    cid_index = {cid: idx for idx, cid in enumerate(cell_ids)}
    exc_indices = [cid_index[cid] for cid in exc_cell_ids if cid in cid_index]
    sum_fields  = fields_modified[:, :, exc_indices].sum(axis=2)
    # neuron id corresponding to fields_exc
    #This is a dictionary way to do it
    # spike_info_exc = {cid: spike_info_moving[cid] for cid in exc_cell_ids
    #                 if cid in spike_info_moving}
    #spike_times_exc = {cid: ts.index.values for cid, ts in spike_info_exc.items()}
    
    #For pynapple: 
    spikes_exc = nap.TsGroup({cid: nap.Ts(t=spike_info_moving[cid].index.values)
                            for cid in exc_cell_ids
                            if cid in spike_info_moving})


    # print("Excitatory cells with place fields:", len(exc_cell_ids))
    # print("Excitatory cells with movement spikes:", len(spike_info_exc))
    # print("Silent excitatory PF cells:",
    #     set(exc_cell_ids) - set(spike_info_exc.keys()))
    
    
    #Find occupancy at each position throughout experiment
    # Use the same bin edges as for the place fields to ensure consistency. 
    #This is a similar procedure from get_place_fields.py
    # np.digitize returns indices in [1, len(edges)] -> subtract 1 to get [0, nbins-1]
    #This should be similar to round(New_Position_Data) in Brad's matlab framework
    # as a reminder, the edges were calculated as 
    # x_bin_edge = np.arange(xmin, xmax + bin_size, bin_size) --> 2cm bins
    # y_bin_edge = np.arange(ymin, ymax + bin_size, bin_size)
    x_bin = np.digitize(xpos_run, x_edges) - 1
    y_bin = np.digitize(ypos_run, y_edges) - 1

    # clamp into valid range in case of numerical edge cases
    x_bin = np.clip(x_bin, 0, nx - 1)
    y_bin = np.clip(y_bin, 0, ny - 1)

    # Approximate dt between position samples in moving periods
    #The matlab code uses mean(diff(t_m))
    dt_pos = np.mean(np.diff(t_run))

    # True occupancy (time spent in each bin while moving)
    True_Cumulative_Position_Occupancy = np.zeros((ny, nx), dtype=float)
    for xb, yb in zip(x_bin, y_bin):
        True_Cumulative_Position_Occupancy[yb, xb] += dt_pos
    
    # Time range
    # start_time = position_data.start_time()
    # end_time =  position_data.end_time()
    times = np.concatenate([np.arange(position_data.time_support.start[i],
                position_data.time_support.end[i],
                time_step) for i in range(len(position_data.time_support))])
    #times = np.arange(start_time, end_time, time_step)
    #print(f"Decoding time range: {start_time:.3f} to {end_time:.3f} (step {time_step})")
    
    #Brad preallocates Position_error but i'm going to append as I go
    cumulative_position_occupancy = np.zeros((ny, nx))
    position_errors = []
    cumulative_posterior = None
    
    for i in times:
        t1 = i + time_bin_size
        #This is to make sure the current position is above velocity threshold
        pos_mask = (post >= i) & (post < t1)
        if not pos_mask.any():
            continue
        if vel[pos_mask].mean() < velocity_threshold:
            continue

        # spike in this current decoding window 
        spike_exc_indices = []
        interval = nap.IntervalSet(start=i, end=t1)
        spikes_bin = spikes_exc.restrict(interval)
        for cid, ts in spikes_bin.items():
            n = len(ts)
            if n > 0:
                k = cid_index[cid]
                #This is how many times that cell fired in this current time window, 
                # takes into account if it spiked multiple times
                spike_exc_indices.extend([k] * n)
        # for cid, t_spikes in spike_times_exc.items():
        #     mask = (t_spikes >= i) & (t_spikes < t1)
        #     if mask.any():
        #         spike_exc_indices.extend([exc_cell_ids[cid]] * int(mask.sum()))

        if len(spike_exc_indices) == 0:
            continue
        
        # true position
        x_true_pos = xpos[pos_mask].mean()
        y_true_pos = ypos[pos_mask].mean()

        x_true = np.clip(np.digitize(x_true_pos, x_edges) - 1, 0, nx - 1)
        y_true = np.clip(np.digitize(y_true_pos, y_edges) - 1, 0, ny - 1)

        # decoding
        # lambda_prod = fields_modified[:, :, spike_exc_indices].prod(axis=2)
        # decoded_matrix = lambda_prod * np.exp(-time_bin_size * sum_fields)
        log_lambda = np.log(fields_modified[:, :, spike_exc_indices]).sum(axis=2)
        log_post   = log_lambda - time_bin_size * sum_fields
        log_post  -= log_post.max()       # stability before exp
        decoded_matrix  = np.exp(log_post)
        decoded_matrix /= decoded_matrix.sum()

        # total = decoded_matrix.sum()
        # if total == 0:
        #     continue

        # decoded_matrix /= total
        y_hat, x_hat = np.unravel_index(np.argmax(decoded_matrix), decoded_matrix.shape)
        
        err = np.sqrt((x_hat - x_true)**2 + (y_hat - y_true)**2) * bin_size
        position_errors.append([i, err])

        # accumulate posterior
        if cumulative_posterior is None:
            cumulative_posterior = decoded_matrix
        else:
            cumulative_posterior += decoded_matrix
        
        #actual x,y values of true position
        # ax = np.clip(round(x_true), 0, nx - 1)
        # ay = np.clip(round(y_true), 0, ny - 1)
        cumulative_position_occupancy[y_true, x_true] += time_bin_size

    position_errors = np.asarray(position_errors)
    
    # Cumulative distribution
    sorted_errors = np.sort(position_errors[:, 1])
    cumulative_prob = np.linspace(0, 1, len(sorted_errors))
    cumulative_error = np.column_stack([sorted_errors, cumulative_prob])
    
    mean_error = np.mean(position_errors[:, 1])
    median_error = np.median(position_errors[:, 1])
    
    results = {
        'position_error': position_errors,
        'cumulative_error': cumulative_error,
        'mean_error': mean_error,
        'median_error': median_error,
        'cumulative_posterior': cumulative_posterior,
        'cumulative_position_occupancy': cumulative_position_occupancy}
    
    
    return results

