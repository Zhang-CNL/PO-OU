from scipy.signal import filtfilt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from math import ceil, sqrt

def find_PFs(initial_conditions, position_data, spike_info, spike_times_filtered, 
                bin_size=2, velocity_cutoff=10, firing_rate_cutoff=1,
                track_type='linear'):
    
    '''
    This function will find both 2D and 1D linear place fields but only bidirectional.
    If track_type == 'linear', bidirectional place fields
    If track_type == 'open', 2D place fields
    '''    
    #This block is for debugging and checking data types
    # print(f"Position data columns: {position_data.columns}")
    # print("Inspect position data head:\n", position_data.as_dataframe().head())

    # print(f"Spike_times_filtered keys (cell IDs): {list(spike_times_filtered.keys())}")
    # print(f"Spike_info keys (cell IDs): {list(spike_info.keys())}")
    # #Picks the first cell to examine
    # example_cell = next(iter(spike_info.keys()))
    # print(f"Example cell ID: {example_cell}")
    # print("Spike_info[example_cell] columns:", spike_info[example_cell].columns)
    # print("Spike_info[example_cell] head:\n", spike_info[example_cell].as_dataframe().head())
    # As a reminder, 1 is excitatory, 0 is inhibitory
    
    # in load data, we find velocity and movement direction 
    # but only filter for running_epoch, not speed
    # The matlab code recalculated the normalized position to start at 0.001,
    #   distance moved, velocity, and time between frames.
    #   I stored this when I calculated velocity in position_data in the load_data function.
    #   I don't think I need to do this again.
    
    vel = position_data['velocity'].values
    dt  = position_data['time_between_frames'].values
    x   = position_data['x'].values
    y   = position_data['y'].values
    hd = position_data['head_direction'].values
    md = position_data['movement_direction'].values
    
    # print(f"Velocity shape: {np.shape(vel)}, with values: {vel}")
    # print(f"dt shape: {np.shape(dt)}, with values: {dt}")
    # print(f"x shape: {np.shape(x)}, with values: {x}")
    # print(f"y shape: {np.shape(y)}, with values: {y}")
    # print(f"hd shape: {np.shape(hd)}, with values: {hd}")
    # print(f"md shape: {np.shape(md)}, with values: {md}")
    
    ## if you want to add directional filtering, add it here
    # for now we remove it (see calculate_linear_place_fields)

    # Filter by speed
    mask = vel >= velocity_cutoff
    x_run = x[mask]
    y_run = y[mask]
    dt_run = dt[mask]
    
    #make bins
    xmin, xmax = np.min(x_run), np.max(x_run)
    ymin, ymax = np.min(y_run), np.max(y_run)
    x_bin_edge = np.arange(xmin, xmax + bin_size, bin_size)
    y_bin_edge = np.arange(ymin, ymax + bin_size, bin_size)
    # print(f"Number of bins: {len(x_bin_edge), len(y_bin_edge)}")
    # print(f"x bin values: {x_bin_edge}")
    # print(f"y bin values: {y_bin_edge}")
    
    #assign position to bins
    #digitize starts at 1 so we subtract 1
    #if you prefer searchsorted, you would use np.searchsorted(x_bins, x_run, side="right")
    x_bin = np.digitize(x_run, x_bin_edge)-1
    y_bin = np.digitize(y_run, y_bin_edge)-1
    
    #check bins 
    # print(f"Number of positions: {len(x_bin), len(y_bin)}")
    # print(f"x position values sorted into bins: {x_bin}")
    # print(f"y position values sorted into bins: {y_bin}")
    
    nbx = x_bin.max() + 1
    nby = y_bin.max() + 1
    #print(f"Number of bins: {nbx, nby}")
    
    #histogram check
    # plt.figure()
    # plt.hist(x_run, bins=x_bin_edge)
    # plt.xlabel("x position")
    # plt.ylabel("Count")
    # plt.title("Position samples along x (running only)")
    # plt.show()
    
    # occ2d, _, _ = np.histogram2d(x_run, y_run, bins=[x_bin_edge, y_bin_edge], weights=dt_run)

    # plt.figure()
    # plt.imshow(
    #     np.flipud(occ2d.T),
    #     extent=[x_bin_edge[0], x_bin_edge[-1], y_bin_edge[0], y_bin_edge[-1]],
    #     aspect="equal"
    # )
    # plt.colorbar(label="Time (s)", shrink=0.4)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Occupancy map (running only)")
    # plt.show()
    
    #Find the time each animal spends in each bin
    #dt is the time elapsed between position samples
    Time_In_Position = np.zeros((nby, nbx))

    for i in range(len(x_bin)):
        xb = x_bin[i]
        yb = y_bin[i]
        Time_In_Position[yb, xb] += dt_run[i]
    
    #time in each bin check
    # print("Total time (s):", np.sum(dt_run))
    # print("Total occupancy time (s):", np.sum(Time_In_Position))
    # plt.figure()
    # plt.imshow(Time_In_Position, origin="lower", aspect="equal")
    # plt.colorbar(label="Time in bin (s)", shrink=0.4)
    # plt.title("Occupancy map (Time_In_Position)")
    # plt.xlabel("x bin")
    # plt.ylabel("y bin")
    

    # Now we need to find the firing rate in each bin 
    # repeat logic from above to assign spikes to bins 
    # but now we loop over each cell
    cell_ids = sorted(spike_info.keys())
    n_cells = len(cell_ids)

    Spikes_In_Position = np.zeros((nby, nbx, n_cells))

    for k, cell_id in enumerate(cell_ids):
        cell_spikes = spike_info[cell_id]
        sx = cell_spikes["x"]
        sy = cell_spikes["y"]
        svel = cell_spikes["velocity"]

        # Apply same velocity cutoff at spike times
        spike_run_mask = svel >= velocity_cutoff
        sx = sx[spike_run_mask]
        sy = sy[spike_run_mask]

        # Bin spikes
        sx_bins = np.digitize(sx, x_bin_edge) - 1
        sy_bins = np.digitize(sy, y_bin_edge) - 1

        for i in range(len(sx_bins)):
            xb = sx_bins[i]
            yb = sy_bins[i]
            Spikes_In_Position[yb, xb, k] += 1

    Firing_Rate_In_Position = np.zeros(Spikes_In_Position.shape)
    # for each cell k, divide by Time_In_Position
    for k in range(n_cells):
        Firing_Rate_In_Position[:, :, k] = (
            Spikes_In_Position[:, :, k] / Time_In_Position)

    # Replace NaN/inf with 0
    Firing_Rate_In_Position[np.isnan(Firing_Rate_In_Position)] = 0
    Firing_Rate_In_Position[np.isinf(Firing_Rate_In_Position)] = 0

    # Gaussian smoothing (2D), 
    # like MATLAB fspecial('gaussian',[20 20],2) i think
    Field_Data = np.zeros(Firing_Rate_In_Position.shape)
    for i in range(n_cells):
        Field_Data[:, :, i] = gaussian_filter(
            Firing_Rate_In_Position[:, :, i],
            sigma=2.0)

    Field_Data[Field_Data < 0] = 0.0
    #This is like np.isnan and np.isinf together
    Field_Data[~np.isfinite(Field_Data)] = 0.0

    # Apply firing rate cutoff and eliminate low-peak fields
    for i in range(n_cells):
        if np.nanmax(Field_Data[:, :, i]) < firing_rate_cutoff:
            Field_Data[:, :, i] = 0.0


    Field_Data_Linear = None

    if track_type == "linear":
        # Decide primary axis 
        # because some tracks are vertical 
        # and others are horizontal
        #I have not written the diagonal track yet
        
        ##This is basically a copy of the code above 
        # but for linear track
        x_span = x_run.max() - x_run.min()
        y_span = y_run.max() - y_run.min()

        if x_span >= y_span:
            # Horizontal track, linear along X
            Linear_Spikes_In_Position = Spikes_In_Position.sum(axis=0)  # shape (nx, n_cells)
            Linear_Time_In_Position = Time_In_Position.sum(axis=0)      # shape (nx,)
        else:
            # Vertical track, linear along Y
            Linear_Spikes_In_Position = Spikes_In_Position.sum(axis=1)  # shape (ny, n_cells)
            Linear_Time_In_Position = Time_In_Position.sum(axis=1)      # shape (ny,)

        # Compute linear rate in position (bidirectional)
        n_lin_bins = Linear_Spikes_In_Position.shape[0]
        Field_Data_Linear = np.zeros(Linear_Spikes_In_Position.shape)

        for k in range(n_cells):
            Field_Data_Linear[:, k] = (
                Linear_Spikes_In_Position[:, k] / Linear_Time_In_Position)

        Field_Data_Linear[~np.isfinite(Field_Data_Linear)] = 0.0

        # Smooth along the linear axis (1D gaussian),
        # analogous to MATLAB's filtfilt(Filter,1,Linear_Rate_In_Position)
        for k in range(n_cells):
            Field_Data_Linear[:, k] = gaussian_filter1d(
                Field_Data_Linear[:, k],
                sigma=2.0)

        Field_Data_Linear[Field_Data_Linear < 0] = 0.0
        Field_Data_Linear[~np.isfinite(Field_Data_Linear)] = 0.0

        # Apply firing rate cutoff on linear fields as well
        for k in range(n_cells):
            if np.nanmax(Field_Data_Linear[:, k]) < firing_rate_cutoff:
                Field_Data_Linear[:, k] = 0.0


    results = {
        "Field_Data": Field_Data,                  
        "Field_Data_Linear": Field_Data_Linear,    # linear fieldsor None
        "Time_In_Position": Time_In_Position,      
        "Spikes_In_Position": Spikes_In_Position,  
        "cell_ids": cell_ids,
        "x_bin_edges": x_bin_edge,
        "y_bin_edges": y_bin_edge,
        "bin_size": bin_size,
        "velocity_cutoff": velocity_cutoff,
        "firing_rate_cutoff": firing_rate_cutoff}
    
    return results
    
    
    
