import numpy as np
import matplotlib.pyplot as plt
from typing import List

from .core import save_wrapper

@save_wrapper
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

@save_wrapper
def plot_spikemat_position_aligned(spike_info, position_info, place_cell_ids, n_cells=4, cell_selection=None):
    if isinstance(cell_selection, list):
        cell_ids = cell_selection
    elif cell_selection == 'random':
        cell_ids = np.random.choice(place_cell_ids, n_cells, replace=False)
    else:
        cell_ids = place_cell_ids[:n_cells]

    fig,ax = plt.subplots(figsize=(16,16), dpi=300)

    ax.plot(position_info['x'], position_info['y'], c='black',alpha=.4, linewidth=.5, label='Rat Trajectory')

    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))
    for i,cell in enumerate(cell_ids):
        subset = spike_info[cell] 
        ax.scatter(subset['x'], subset['y'], s=5, c=colors[i], alpha=.5, label=f'Cell {cell}')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])

    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_title("Spike Positions on Trajectory")
    ax.legend()
    return fig