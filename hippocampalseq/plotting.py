import numpy as np
import matplotlib.pyplot as plt
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

def plot_placefields(place_fields: hseu.PlacefieldData, pfs: List[int]):
    global __plotting_initialized
    if not __plotting_initialized:
        __init_plotting()

    fig, ax = plt.subplots(1,len(pfs), figsize=(2,.5), dpi=300)

    place_fields = place_fields.place_fields#[place_fields.place_cell_ids]

    for i in range(len(pfs)):
        ax[i].imshow(place_fields[pfs[i]], origin='lower')
        #print(rat_data.PlaceFieldData['place_fields'][pfs[i]].max())

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
        
    rect = plt.Rectangle(
        (0, 0), 1, 1, fill=False, color="k", lw=.5, alpha=.2,
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])

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

def cell_spike_raster_plot(
        cell_spikes: Dict[int, np.ndarray]|List[np.ndarray],
        plot_start_time: Optional[float] = None, 
        plot_end_time: Optional[float] = None, 
        **fig_kwargs 
    ):
    global __plotting_initialized
    if not __plotting_initialized:
        __init_plotting()

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