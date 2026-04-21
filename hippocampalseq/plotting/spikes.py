import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from .core import save_wrapper

@save_wrapper
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

@save_wrapper
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