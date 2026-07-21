import numpy as np
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from .core import save_wrapper
from .trajectories import plot_trajectory_with_velocity

@save_wrapper
def spikemat_raster_plot(spike_mat: np.ndarray, **fig_kwargs):
    fig = plt.figure(**fig_kwargs)
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


@save_wrapper
def spike_raster_plot(
        spike_data: nap.TsGroup,
        plot_start_time: float|None = None, 
        plot_end_time: float|None = None, 
        ax=None,
        **fig_kwargs
    ):
    time = spike_data.time_support
    if plot_start_time is None:
        plot_start_time = time.start.min()
    if plot_end_time is None:
        plot_end_time = time.end.max()

    time_subset = nap.IntervalSet(
        start=plot_start_time,
        end=plot_end_time,
    )
    spike_data = spike_data.restrict(time_subset)

    if ax is None:
        fig = plt.figure(**fig_kwargs, dpi=300)
        ax = fig.add_axes([.2, .05, .75, .8])
    else:
        fig = plt.gcf()

    for uid,spikes in spike_data.items():
        ax.eventplot(
            spikes.index.values,
            lineoffsets=uid,
            linelengths=4, 
            linewidths=.1,
            color='black', 
            orientation='horizontal'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)

    ax.set_ylabel("Cell ID", rotation=90, labelpad=-5)
    ax.set_ylabel("Time (s)")
    ax.tick_params(direction='out', length=1, width=.5)

    ax.set_yticks([0, len(spike_data)])
    ax.set_yticklabels([1, len(spike_data) + 1])
    ax.set_ylim([0, len(spike_data)])

    ax.set_xticks([])
    ax.set_xlim([plot_start_time, plot_end_time])

    return fig

@save_wrapper
def plot_lfp_data(
        lfp_data: nap.TsdFrame,
        n_seconds: float = 10.0,
    ):
    times = lfp_data.index.values
    t0 = times[0]
    t1 = min(t0 + n_seconds, times[-1])

    lfp_data = lfp_data.restrict(nap.IntervalSet(start=t0, end=t1))

    fig,ax = plt.subplots(4,1, figsize=(12,8), dpi=300, sharex=True)
    ax[0].plot(lfp_data.t, lfp_data['Raw LFP'], 'k-', linewidth=0.5)
    ax[0].set_ylabel(r"$\mu$V")
    ax[0].set_title("Raw LFP")

    ax[1].plot(lfp_data.t, lfp_data['Filtered LFP'], 'b-', linewidth=0.7)
    ax[1].set_ylabel(r"$\mu$V")
    ax[1].set_title("Theta-filtered LFP")

    ax[2].plot(lfp_data.t, lfp_data['Amplitude'], 'r-', linewidth=1.0)
    ax[2].set_ylabel(r"$\mu$V")
    ax[2].set_title("Amplitude")

    ax[3].plot(lfp_data.t, lfp_data['Phase Rad'], 'r-', linewidth=0.7)
    ax[3].set_ylabel("Phase (rad)")
    ax[3].set_title("Phase")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylim([-np.pi, np.pi])

    plt.tight_layout()

    return fig

@save_wrapper
def plot_session_stitching(run_position_data: nap.TsdFrame, spike_data: nap.TsGroup, track_axis: str|None = None):
    run_intervals = run_position_data.time_support
    n_sessions = len(run_intervals)
    if n_sessions <= 1:
        return None

    if track_axis is None:
        xr = run_position_data['x'].max() - run_position_data['x'].min()
        yr = run_position_data['y'].max() - run_position_data['y'].min()
        track_axis = 'x' if xr >= yr else 'y'

    t = run_position_data.t
    track = run_position_data[track_axis].values
    velocity = run_position_data['Velocity'].values

    fig, axs = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.2]},
    )

    plot_trajectory_with_velocity(
        np.concatenate((t[:,None], track[:,None]), axis=1),
        velocity,
        ax=axs[0],
        colorbar=False
    )

    spike_raster_plot(
        spike_data.restrict(run_intervals), 
        ax=axs[1]
    )

    axs[1].set_xlabel("Running time (s)")

    offset = 0
    boundaries = [0.0]
    for i in range(n_sessions):
        s,e = run_intervals.start[i], run_intervals.end[i]
        offset += (e - s)
        boundaries.append(offset)

    for ax in axs:
        for b in boundaries[1:-1]:
            ax.axvline(
                b,
                color='red',
                linestyle='--',
                alpha=0.5,
                linewidth=0.5
            )

    """
    for i,b in enumerate(boundaries[:-1]):
        axs[0].text(
            b + 5,
            axs[0].get_ylim()[1] * .97,
            f"Session {i+1}",
            fontsize=10,
            va='top',
            color='dimgray'
        )
    """
    
    plt.tight_layout()
    return fig

    