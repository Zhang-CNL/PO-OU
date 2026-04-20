import os
import numpy as np
import matplotlib.pyplot as plt
import functools
from typing import Optional, List, Dict 

import hippocampalseq.utils as hseu

__plotting_initialized = False

def __init_plotting():
    global __plotting_initialized
    if __plotting_initialized:
        return
    __plotting_initialized = True
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=2, color='r')
    #plt.rcParams['font.sans-serif'] = ['Helvetica']

def __save_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, file_path=None, file_name=None, **kwargs):
        __init_plotting() 
        res = fn(*args, **kwargs)
        if file_path is None:
            file_path = "./results/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if file_name is not None:
            plt.savefig(os.path.join(file_path, file_name), dpi=300)
        return res
    return wrapper

@__save_wrapper
def plot_placefields(rat_data: hseu.RatData, pfs: Optional[List[int]] = None, show_titles: bool = True):
    place_fields = rat_data.place_field_data.place_fields

    if pfs is None:
        pfs = np.arange(len(place_fields))

    rows = 10
    if rows > len(pfs):
        rows = 1
    cols = len(pfs) // rows 
    if cols == 0 or len(pfs) % cols > 0:
        cols += 1

    fig, ax = plt.subplots(rows, cols, figsize=(2*(len(pfs) // cols),.5*rows), dpi=300)
    ax = ax.flatten()

    max_firing = np.max(place_fields, axis=(1,2))

    for i in range(len(pfs)):
        if show_titles:
            ax[i].set_title(f"Max FR: {max_firing[i]:.2f}", fontsize=4)
        ax[i].imshow(place_fields[pfs[i]], origin='lower')

    for i in range(len(pfs), len(ax)):
        ax[i].axis('off')

    binned_len = len(place_fields[pfs[0]])
        
    ax[0].set_xticks([0,binned_len])
    ax[0].set_xticklabels([0,"2m"])
    ax[0].set_yticks([0,binned_len])
    ax[0].set_yticklabels([0,"2m"])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].tick_params(direction='out', length=0, width=.5, pad=1)

    for i in range(1,len(pfs)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
       
    rect = plt.Rectangle(
        (0, 0), 1, 1, fill=False, color="k", lw=.5, alpha=.2,
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])

@__save_wrapper
def spikemat_raster_plot(spike_mat: np.ndarray, **fig_kwargs):
    fig = plt.figure(**fig_kwargs, dpi=300)
    ax = fig.add_axes([.2, .05, .75, .8])

    T,n_cells = spike_mat.shape
    

    im = ax.imshow(spike_mat, cmap=plt.cm.gray_r)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(
        axis='x',
        direction='in', 
        length=1, 
        width=.5, 
        top=True, 
        labeltop=True, 
        bottom=False, 
        labelbottom=False, 
        labelleft=False
    )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('cell number')


    ax.set_xticks([0, n_cells])
    ax.set_xticklabels([1, n_cells + 1])
    ax.set_xlim([0, n_cells])

    ax.set_yticks([])

    #fig.colorbar(im, ax=ax, orientation='vertical', ticks=np.unique(spike_mat))


def spike_raster_plot(
        spike_ids: np.ndarray, 
        spike_times: np.ndarray, 
        plot_start_time: Optional[float] = None, 
        plot_end_time: Optional[float] = None, 
        **fig_kwargs
    ):
    if plot_start_time is None:
        plot_start_time = spike_times.min()
    if plot_end_time is None:
        plot_end_time = spike_times.max()

    start_idx   = np.searchsorted(spike_times, plot_start_time)
    end_idx     = np.searchsorted(spike_times, plot_end_time)
    spike_ids   = spike_ids[start_idx:end_idx]
    spike_times = spike_times[start_idx:end_idx]

    unique_cells = np.unique(spike_ids).astype(int)
    cell_spikes = []
    for cell in unique_cells:
        spikes = spike_times[spike_ids == cell]
        cell_spikes.append(spikes)

    return cell_spike_raster_plot(
        cell_spikes,
        plot_start_time=plot_start_time,
        plot_end_time=plot_end_time,
        **fig_kwargs
    )

@__save_wrapper
def cell_spike_raster_plot(
        cell_spikes: Dict[int, np.ndarray]|List[np.ndarray],
        plot_start_time: Optional[float] = None, 
        plot_end_time: Optional[float] = None, 
        **fig_kwargs 
    ):
    if plot_start_time is None:
        plot_start_time = min([spikes.min() for spikes in cell_spikes])
    if plot_end_time is None:
        plot_end_time = max([spikes.max() for spikes in cell_spikes])

    fig = plt.figure(**fig_kwargs, dpi=300)
    ax = fig.add_axes([.2, .05, .75, .8])

    for i,spikes in enumerate(cell_spikes):
        startidx = np.searchsorted(spikes, plot_start_time)
        endidx   = np.searchsorted(spikes, plot_end_time)
        ax.eventplot(
            spikes[startidx:endidx],
            lineoffsets=i,
            linelengths=4, 
            linewidths=.1,
            color='black', 
            orientation='horizontal'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)

    ax.set_ylabel('cell number', rotation=90, labelpad=-5)
    ax.tick_params(direction='out', length=1, width=.5)

    ax.set_yticks([0, len(cell_spikes)])
    ax.set_yticklabels([1, len(cell_spikes) + 1])
    ax.set_ylim([0, len(cell_spikes)])

    ax.set_xticks([])
    ax.set_xlim([plot_start_time, plot_end_time])

    return fig

@__save_wrapper
def plot_trajectories(trajectories: List[np.ndarray], ax=None, **kwargs):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    if ax is None:
        ax = plt.gca()
    for trajectory in trajectories:
        assert trajectory.shape[1] == 2, "Trajectory must be 2D"
        ax.plot(trajectory[:,0], trajectory[:,1], 'k-', alpha=.5, linewidth=.5)

    ax.set_yticks([0, 200])
    ax.set_xticks([0, 200])

    ax.set_ylim([0, 200])
    ax.set_xlim([0, 200])

@__save_wrapper
def plot_spikemat_position_aligned(spike_ids, place_cell_ids, x, y, n_cells=4, cell_selection=None):
    if isinstance(cell_selection, list):
        cell_ids = cell_selection
    elif cell_selection == 'random':
        cell_ids = np.random.choice(place_cell_ids, n_cells, replace=False)
    else:
        cell_ids = place_cell_ids[:n_cells]

    fig,ax = plt.subplots(figsize=(16,16), dpi=300)

    ax.plot(x,y, 'k-', alpha=.2, linewidth=.5, label='Rat Trajectory')

    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))
    for i,cell in enumerate(cell_ids):
        idx = np.where(spike_ids == cell)[0]
        ax.scatter(x[idx], y[idx], s=5, c=colors[i], alpha=.5, label=f'Cell {cell}')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])

    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_title("Spike Positions on Trajectory")
    ax.legend()
    return fig
