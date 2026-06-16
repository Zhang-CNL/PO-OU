import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import random

def plot_loaded_data(position_data, spike_data, excitatory_neurons, inhibitory_neurons, 
                    lfp_data,max_cells=None,time_range=None):
    

    # Extract LFP components
    filtered_lfp = lfp_data['filtered_lfp']
    amplitude = lfp_data['amplitude']
    phase = lfp_data['phase']
    raw_lfp = lfp_data['raw_lfp']
    # actual_start = lfp_data['metadata']['actual_start']
    # actual_end = lfp_data['metadata']['actual_end']
    run_intervals_lfp = lfp_data['run_interval']
    actual_start, actual_end = lfp_data['metadata']['lfp_data_range']
    
#Position and Spikes 
    fig1, axes = plt.subplots(4, 1, figsize=(16, 16))
    
    t_min = position_data.time_support.start[0]
    t_max = position_data.time_support.end[-1]
    
    if time_range is None:
        time_range = (t_min, min(t_min + 60, t_max))
    
    raster_start, raster_end = time_range
    
    x_range = position_data['x'].max() - position_data['x'].min()
    y_range = position_data['y'].max() - position_data['y'].min()
    track_axis = 'x' if x_range > y_range else 'y'
    print(f'track axis: {track_axis} '
        f'(x range: {x_range:.1f}, y range: {y_range:.1f})')
    
    # Select track axis
    if track_axis == 'y':
        track_pos = position_data['y'].values
        track_label = 'Y Position (cm)'
    else:# track_axis == 'x':
        track_pos = position_data['x'].values
        track_label = 'X Position (cm)'
        
    
    # 1. Track position vs time
    ax = axes[0]
    time = position_data.index
    velocity = position_data['velocity'].values
    
    scatter = ax.scatter(time, track_pos, c=velocity, cmap='copper', 
                        s=0.5, alpha=1, vmin=0, vmax=np.percentile(velocity, 95))
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(track_label, fontsize=12)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (cm/s)', fontsize=11)
    
    ax.axvline(raster_start, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(raster_end, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # 2. Trajectory
    if track_axis == 'x':
        traj_h, traj_v = position_data['x'].values, position_data['y'].values
        h_label, v_label = 'X Position (cm)', 'Y Position (cm)'
    else:  # track_axis == 'y'
        traj_h, traj_v = position_data['y'].values, position_data['x'].values
        h_label, v_label = 'Y Position (cm)', 'X Position (cm)'
    ax = axes[1]
    scatter = ax.scatter(traj_h, traj_v, 
                        c=velocity, cmap='copper', s=0.5, alpha=1, 
                        vmin=0, vmax=np.percentile(velocity, 95))
    ax.set_xlabel(h_label, fontsize=12)
    ax.set_ylabel(v_label, fontsize=12) 
    ax.set_title('Trajectory colored by speed', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (cm/s)', fontsize=11)
    
    # 3. Spike raster
    ax = axes[2]
    
    position_interval = position_data.time_support
    spike_data = spike_data.restrict(position_interval)
    
    cell_ids = list(spike_data.keys())
    if max_cells is not None and len(cell_ids) > max_cells:
        cell_ids = cell_ids[:max_cells]
    
    exc_cells = [c for c in spike_data.keys() if c in excitatory_neurons]
    n_exc_shown = 0
    n_other_shown = 0
    
    for cell_id in cell_ids:
        spikes = spike_data[cell_id].restrict(nap.IntervalSet(raster_start, raster_end))
        
        if cell_id in excitatory_neurons:
            color = 'blue'
            alpha = 0.6
            n_exc_shown += 1
        elif cell_id in inhibitory_neurons:
            color = 'red'
            alpha = 0.3
            n_other_shown += 1
        
        if len(spikes) > 0:
            ax.scatter(spikes.index, [cell_id] * len(spikes), 
                        color=color, alpha=alpha, s=1)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cell ID', fontsize=12)
    ax.set_title(f'Spike Raster [{raster_start:.1f}-{raster_end:.1f}s]',
                fontsize=14, fontweight='bold')
    ax.set_xlim(raster_start, raster_end)
    ax.set_ylim(0, max(cell_ids) + 1)
    
    ax.scatter([], [], color='blue', alpha=0.6, s=20, label='Excitatory')
    ax.scatter([], [], color='red', alpha=0.3, s=20, label='Inhibitory')
    ax.legend()
    
    
    #movement direction
    axmv = axes[3]
    mvd = position_data['movement_direction'].values
    
    scatter2 = axmv.scatter(traj_h, traj_v,  
                        c=mvd, cmap='twilight', s=2, alpha=0.6,
                        vmin=0,vmax=360)
    axmv.set_xlabel(h_label, fontsize=12)
    axmv.set_ylabel(v_label, fontsize=12) 
    axmv.set_title('Trajectory colored by movement direction', fontsize=14, fontweight='bold')
    
    cbar2 = plt.colorbar(scatter2, ax=axmv)
    cbar2.set_label('Movement Direction (degrees)', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    #fig1.savefig("Position_Spike_Data_Pfieffer.pdf", format="pdf")

    #LFP Overview  
    t_lfp = filtered_lfp.index.values
    n_plot = min(100000, len(t_lfp))

    
    # LFP zoomed
    n_plot2 = min(20000, len(t_lfp))
    
    fig3, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t_lfp[:n_plot2], raw_lfp.values[:n_plot2], 'k-', linewidth=0.5)
    axes[0].set_ylabel('μV')
    axes[0].set_title('Raw LFP')
    
    axes[1].plot(t_lfp[:n_plot2], filtered_lfp.values[:n_plot2], 'b-', linewidth=0.7)
    axes[1].set_ylabel('μV')
    axes[1].set_title('Filtered LFP (Theta)')
    
    axes[2].plot(t_lfp[:n_plot2], amplitude.values[:n_plot2], 'r-', linewidth=1.0)
    axes[2].set_ylabel('μV')
    axes[2].set_title('Amplitude')
    
    axes[3].plot(t_lfp[:n_plot2], phase.values[:n_plot2], 'r-', linewidth=0.7)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Radians')
    axes[3].set_title('Phase')
    axes[3].set_ylim([-np.pi, np.pi])
    
    plt.tight_layout()
    plt.show()

    
    stitching_fig = plot_session_stitching(position_data, spike_data, track_axis)

    out = {
        "Track Position vs Speed": fig1,
        "LFP 20K samples":         fig3,
    }
    if stitching_fig is not None:
        out["Session_Stitching"] = stitching_fig
    return out
    
def plot_session_stitching(position_data, spike_data, track_axis):


    run_intervals = position_data.time_support
    n_sessions = len(run_intervals)
    if n_sessions <= 1:
        return None

    t        = position_data.index.values
    track    = position_data[track_axis].values
    velocity = position_data['velocity'].values

    # Build running-time axis with the inter-session gaps removed
    running_t  = np.zeros_like(t)
    offset     = 0.0
    boundaries = [0.0]
    for i in range(n_sessions):
        s, e = run_intervals.start[i], run_intervals.end[i]
        mask = (t >= s) & (t <= e)
        running_t[mask] = (t[mask] - s) + offset
        offset += (e - s)
        boundaries.append(offset)
        
    

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.2]},
    )

    # Top: track position over running time, colored by speed
    sc = axes[0].scatter(
        running_t, track, c=velocity, cmap='copper',
        s=1, alpha=0.6,
        vmin=0, vmax=np.percentile(velocity, 95),
    )
    plt.colorbar(sc, ax=axes[0], label='Speed (cm/s)')
    axes[0].set_ylabel(f'{track_axis.upper()} Position (cm)')
    axes[0].set_title(
        f'Concatenated view: {n_sessions} sessions, '
        f'total {boundaries[-1]:.1f}s of running data'
    )

    # Bottom: spike raster on running time
    for cell_id in spike_data.keys():
        sp_t = spike_data[cell_id].t
        if len(sp_t) == 0:
            continue
        sp_running = np.full_like(sp_t, np.nan, dtype=float)
        for i in range(n_sessions):
            s, e = run_intervals.start[i], run_intervals.end[i]
            m = (sp_t >= s) & (sp_t <= e)
            sp_running[m] = (sp_t[m] - s) + boundaries[i]
        good = ~np.isnan(sp_running)
        if good.any():
            axes[1].scatter(
                sp_running[good],
                np.full(good.sum(), cell_id),
                s=0.5, color='black', alpha=0.5,
            )
    axes[1].set_xlabel('Running time (s)')
    axes[1].set_ylabel('Cell ID')

    # Mark session boundaries
    for ax in axes:
        for b in boundaries[1:-1]:
            ax.axvline(b, color='red', linestyle='--',
                    alpha=0.5, linewidth=0.5)
    for i, b in enumerate(boundaries[:-1]):
        axes[0].text(
            b + 5, axes[0].get_ylim()[1] * 0.97,
            f'Session {i+1}', fontsize=10, va='top', color='dimgray',
        )

    plt.tight_layout()
    return fig

def plot_spike_position_integration(spike_info, position_data, excitatory_neurons, 
                                    n_cells=4, cell_selection=None, skip_top_n=0):


    # Get excitatory cells using BOTH the external list and the embedded cell_type
    exc_cells = []
    mismatches = []
    for c in spike_info.keys():
        ts = spike_info[c]
        if len(ts) == 0:
            continue
        cell_type = ts.values[0, list(ts.columns).index('cell_type')]
        in_exc_list = c in excitatory_neurons

        if cell_type == 1 and in_exc_list:
            exc_cells.append(c)
        elif cell_type == 1 or in_exc_list:
            mismatches.append((c, cell_type, in_exc_list))

    # if mismatches:
    #     print(f"  WARNING: {len(mismatches)} cells have inconsistent excitatory status")
    #     for c, ct, lst in mismatches[:5]:
    #         print(f"    Cell {c}: cell_type={ct}, in excitatory_neurons={lst}")

    # print(f"  Filtered to {len(exc_cells)} excitatory cells "
    #     f"(out of {len(spike_info)} total)")

    # Now the existing selection logic, with the TsdFrame-safe len()
    if isinstance(cell_selection, list):
        cell_ids = [c for c in cell_selection if c in spike_info.keys()][:n_cells]
    elif cell_selection == 'random':
        cell_ids = random.sample(exc_cells, min(n_cells, len(exc_cells)))
    elif cell_selection == 'most_active':
        sorted_cells = sorted(exc_cells, key=lambda c: len(spike_info[c]), reverse=True)
        cell_ids = sorted_cells[skip_top_n : skip_top_n + n_cells]
    elif cell_selection == 'least_active':
        sorted_cells = sorted(exc_cells, key=lambda c: len(spike_info[c]))
        cell_ids = sorted_cells[skip_top_n : skip_top_n + n_cells]
    else:
        cell_ids = exc_cells[:n_cells]

    # Print the chosen cells with their spike counts for sanity
    # print(f"  Selected cells (mode='{cell_selection}'): "
    #     f"{[(c, len(spike_info[c])) for c in cell_ids]}")
        
        
    x_range = position_data['x'].max() - position_data['x'].min()
    y_range = position_data['y'].max() - position_data['y'].min()
    track_axis = 'x' if x_range > y_range else 'y'
    
    if track_axis == 'x':
        h_col, v_col = 'x', 'y'
        h_label, v_label = 'X Position (cm)', 'Y Position (cm)'
    else:
        h_col, v_col = 'y', 'x'
        h_label, v_label = 'Y Position (cm)', 'X Position (cm)'
        
    fig, ax1 = plt.subplots(figsize=(16, 12))    
    # Spike positions on trajectory
    
    # Plot trajectory
    ax1.plot(position_data[h_col].values, position_data[v_col].values,
            'k-', alpha=0.2, linewidth=0.5, label='Rat trajectory')
    
    # Plot spikes from selected cells
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))
    for i, cell_id in enumerate(cell_ids):
        info = spike_info[cell_id]
        ax1.scatter(info[h_col], info[v_col], s=5, alpha=0.6, 
                    color=colors[i], label=f'Cell {cell_id}')
    
    ax1.set_xlabel(h_label)
    ax1.set_ylabel(v_label)
    ax1.set_title('Spike Positions on Trajectory')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    
    return fig


