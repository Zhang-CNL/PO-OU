import numpy as np
import matplotlib.pyplot as plt

from .core import save_wrapper

# TODO: Rewrite code to use Eryn's placefield formatting

@save_wrapper
def plot_open_placefields(
        place_fields: np.ndarray, 
        pfs: list[int]|None = None, 
        show_titles: bool = True, 
        cmap: str|plt.Colormap = 'hot',
        ax = None,
    ):

    if pfs is None:
        pfs = np.arange(len(place_fields))

    cols = 10
    if cols > len(pfs):
        cols = 1
    rows = len(pfs) // cols 
    if rows == 0 or len(pfs) % rows > 0:
        rows += 1

    if ax is None:
        fig,ax = plt.subplots(rows, cols, figsize=(.5*rows, 2*len(pfs)//cols), dpi=300)
    else:
        fig = plt.gcf()
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    max_firing = np.max(place_fields, axis=(1,2))

    for i in range(len(pfs)):
        if show_titles:
            ax[i].set_title(f"Max FR: {max_firing[i]:.2f}", fontsize=4)
        ax[i].imshow(place_fields[pfs[i]], origin='lower', cmap=cmap)

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
    ax[0].set_xlabel("X Position")
    ax[0].set_ylabel("Y Position")

    for i in range(1,len(pfs)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    #plt.tight_layout()
       
    rect = plt.Rectangle(
        (0, 0), 1, 1, fill=False, color="k", lw=.5, alpha=.2,
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])
    return fig

@save_wrapper
def plot_linear_placefields(
        place_fields: np.ndarray, 
        pfs: list[int]|None = None, 
        cmap: str|plt.Colormap = 'hot', 
        ax = None,
        **fig_kwargs
    ):

    if ax is None:
        fig = plt.figure(**fig_kwargs, dpi=300)
        ax = fig.add_axes([.2, .05, .75, .8])

    # Plot sorted colormap
    if pfs is None:
        max_fr = np.squeeze(np.max(place_fields, axis=1))
        sort_idx = np.argsort(max_fr)
        sorted_place_fields = place_fields[sort_idx]
        aspect='auto'
    else:
        sorted_place_fields = place_fields[pfs]
        aspect=None

    im = ax.imshow(
        sorted_place_fields,
        aspect=aspect,
        cmap=cmap,
        origin='lower',
        interpolation='nearest'
    )
    ax.set_xlabel("Linear position bin")
    ax.set_ylabel("Cell (sorted by peak)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Firing Rate (Hz)")
    #plt.tight_layout()

