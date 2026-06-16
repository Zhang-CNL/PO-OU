import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from typing import List, Optional, Dict, Tuple

from .core import save_wrapper

@save_wrapper
def plot_trajectories(trajectories: List[np.ndarray]|Dict[str,np.ndarray], ax=None, **kwargs):
    if not isinstance(trajectories, list) and not isinstance(trajectories, dict):
        trajectories = [trajectories]
    if ax is None:
        ax = plt.gca()
    for trajectory in trajectories:
        if isinstance(trajectories, dict):
            label = trajectory
            trajectory = trajectories[label]
        else:
            label = None
        if trajectory.shape[1] == 2:
            ax.plot(
                trajectory[:,0],
                trajectory[:,1], 
                '-', 
                alpha=.5, 
                linewidth=.5,
                label=label
            )
        elif trajectory.shape[1] == 1:
            x = np.arange(len(trajectory))
            ax.plot(x, trajectory[:,0], '-', alpha=.5, linewidth=.5, label=label)
        else:
            raise ValueError(f"Trajectory shape {trajectory.shape} not supported")

    if isinstance(trajectories, dict):
        ax.legend()
        shape = list(trajectories.items())[0][1].shape
    else:
        shape = trajectories[0].shape
    if shape[1] == 2:
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

    ax.set_title("Spike Positions on Trajectory")
    ax.plot(
        position_info['x'],
        position_info['y'], 
        color='black',
        alpha=.4, 
        linewidth=.5, 
        label='Rat Trajectory'
    )

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
    ax.set_xlim([environment_size[0], environment_size[2] - 1])
    ax.set_ylim([environment_size[1], environment_size[3] - 1])
    ax.set_xticks([environment_size[0], environment_size[2] - 1])
    ax.set_yticks([environment_size[1], environment_size[3] - 1])
    ax.set_xticklabels([0,f"{int((environment_size[2]-environment_size[0])/100)}m"])
    ax.set_yticklabels([0,f"{int((environment_size[3]-environment_size[1])/100)}m"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', length=0, width=.5, pad=1)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.legend()
    return fig

@save_wrapper
def plot_kalman_2d_trajectories(
        means: np.ndarray, 
        covs: np.ndarray, 
        ax = None
    ):
    pass