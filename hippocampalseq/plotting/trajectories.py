import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from typing import List, Optional, Dict, Tuple

from .core import save_wrapper

@save_wrapper
def plot_trajectories(trajectories: List[np.ndarray], ax=None, **kwargs):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    if ax is None:
        ax = plt.gca()
    for trajectory in trajectories:
        if trajectory.shape[1] == 2:
            ax.plot(trajectory[:,0], trajectory[:,1], 'k-', alpha=.5, linewidth=.5)
        elif trajectory.shape[1] == 1:
            x = np.arange(len(trajectory))
            ax.plot(x, trajectory[:,0], 'k-', alpha=.5, linewidth=.5)

    if trajectories[0].shape[1] == 2:
        ax.set_yticks([0, 200])
        ax.set_xticks([0, 200])

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 200])

@save_wrapper
def plot_spikemat_position_aligned(
        spike_info: Dict[int, nap.TsdFrame], 
        position_info: nap.TsdFrame, 
        place_cell_ids: np.ndarray, 
        environment_size: Optional[Tuple[int]] = (0,0,200,200),
        n_cells: int = 4, 
        cell_selection: Optional[List[int]]|str = None,
        ax = None,
    ):
    if isinstance(cell_selection, list):
        cell_ids = cell_selection
    elif cell_selection == 'random':
        cell_ids = np.random.choice(place_cell_ids, n_cells, replace=False)
    else:
        cell_ids = place_cell_ids[:n_cells]

    if ax is None:
        fig,ax = plt.subplots(figsize=(16,16), dpi=300)
    else:
        fig = plt.gcf()

    ax.plot(position_info['x'], position_info['y'], color='black',alpha=.4, linewidth=.5, label='Rat Trajectory')

    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))
    for i,cell in enumerate(cell_ids):
        subset = spike_info[cell] 
        ax.scatter(subset['x'], subset['y'], s=5, color=colors[i], alpha=.5, label=f'Cell {cell}')
    if environment_size is None:
        environment_size = (
            np.min(position_info['x']),
            np.min(position_info['y']),
            np.max(position_info['x']),
            np.max(position_info['y'])
        )
    ax.set_xlim([environment_size[0], environment_size[2]])
    ax.set_ylim([environment_size[1], environment_size[3]])

    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_title("Spike Positions on Trajectory")
    ax.legend()
    return fig

@save_wrapper
def plot_kalman_2d_trajectories(
        means: np.ndarray, 
        covs: np.ndarray, 
        ax = None
    ):
    pass