import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

from .core import save_wrapper

@save_wrapper
def plot_open_placefields(place_fields: np.ndarray, pfs: Optional[List[int]] = None, show_titles: bool = True):

    if pfs is None:
        pfs = np.arange(len(place_fields))

    rows = 10
    if rows > len(pfs):
        rows = 1
    cols = len(pfs) // rows 
    if cols == 0 or len(pfs) % cols > 0:
        cols += 1

    fig, ax = plt.subplots(rows, cols, figsize=(2*(len(pfs) // cols),.5*rows), dpi=300)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

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

@save_wrapper
def plot_linear_placefields(spike_info, place_fields: np.ndarray, pfs: Optional[List[int]] = None, **fig_kwargs):
    if pfs is None:
        pfs = np.arange(len(place_fields))

    fig = plt.figure(**fig_kwargs, dpi=300)
    ax = fig.add_axes([.2, .05, .75, .8])

    # Plot sorted colormap
    max_fr = np.squeeze(np.max(place_fields, axis=1))
    sort_idx = np.argsort(max_fr)
    sorted_place_fields = place_fields[sort_idx]

    im = ax.imshow(
        sorted_place_fields[pfs],
        aspect='auto',
        cmap='hot',
        origin='lower',
        interpolation='nearest'
    )
    ax.set_xlabel("Linear position bin")
    ax.set_ylabel("Cell (sorted by peak)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Firing Rate (Hz)")
    #plt.tight_layout()


