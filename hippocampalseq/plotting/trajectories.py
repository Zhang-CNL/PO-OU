import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from typing import Optional

import hippocampalseq.utils as hseu
from .core import save_wrapper, colored_line

def _plot_trajectories1d(
        trajectories: list[np.ndarray]|dict[str,np.ndarray]|np.ndarray,
        ax=None,
        **plot_kwargs
    ):
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

        x = np.arange(len(trajectory))
        ax.plot(
            x, trajectory.squeeze(),
            '-',
            alpha=.5,
            linewidth=.5,
            label=label,
            **plot_kwargs
        )

    if isinstance(trajectories, dict):
        ax.legend()

def _plot_trajectories2d(
        trajectories: list[np.ndarray]|dict[str,np.ndarray]|np.ndarray, 
        environment_size: Optional[tuple[int,...]] = [(0,200),(0,200)],
        ax=None, 
        **plot_kwargs
    ):
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
        ax.plot(
            trajectory[:,0],
            trajectory[:,1], 
            '-', 
            alpha=.5, 
            linewidth=.5,
            label=label,
            **plot_kwargs
        )
        
    if environment_size is not None:
        xb = environment_size[0]
        yb = environment_size[1]
        ax.set_xlim(xb)
        ax.set_ylim(yb)
        ax.set_xticks(xb)
        ax.set_yticks(yb)
        
    if isinstance(trajectories, dict):
        ax.legend()

@save_wrapper
def plot_trajectories(
        trajectories: list[np.ndarray]|dict[str,np.ndarray]|np.ndarray, 
        environment_size: Optional[tuple[int,...]] = None,
        ax=None, 
        **plot_kwargs
    ):
    if isinstance(trajectories, dict):
        dim = next(iter(trajectories.values())).shape[1]
    elif isinstance(trajectories, list):
        dim = trajectories[0].shape[1]
    else:
        dim = trajectories.shape[-1]
    if dim == 1:
        return _plot_trajectories1d(trajectories, ax=ax, **plot_kwargs)
    elif dim == 2:
        return _plot_trajectories2d(trajectories, environment_size=environment_size, ax=ax, **plot_kwargs)

@save_wrapper
def plot_trajectory_with_velocity(
        trajectory: np.ndarray, 
        velocity: np.ndarray,
        environment_size: tuple[int,...]|None = None,
        label: str|None = None,
        ax=None, 
        colorbar: bool = True,
        **plot_kwargs
    ):
    if ax is None:
        ax = plt.gca()
    
    dim = trajectory.shape[1]

    if dim == 1:
        x = np.arange(len(trajectory))
        y = trajectory
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (cm)')
    else:
        x = trajectory[:,0]
        y = trajectory[:,1]
        ax.set_xlabel('X position (cm)')
        ax.set_ylabel('Y position (cm)')

    lines = colored_line(
        x, y, 
        velocity,
        cmap='copper',
        ax=ax,
        alpha=1,
        label=label,
        linewidth=0.7,
        **plot_kwargs
    )
    if colorbar:
        cbar = plt.colorbar(lines, ax=ax)
        cbar.set_label('Velocity (cm/s)')

    if environment_size is not None:
        xb = environment_size[0]
        yb = environment_size[1]
        ax.set_xlim(xb)
        ax.set_ylim(yb)
        ax.set_xticks(xb)
        ax.set_yticks(yb)
        
    if label is not None:
        ax.legend()

@save_wrapper
def plot_spikemat_position_aligned(
        spike_info: dict[int, nap.TsdFrame], 
        position_info: nap.TsdFrame, 
        place_cell_ids: np.ndarray, 
        environment_size: list[tuple[int,...]]|None = [(0,200),(0,200)],
        n_cells: int = 4, 
        cell_selection: Optional[list[int]]|str = None,
        ax = None,
    ):
    if isinstance(cell_selection, list):
        cell_ids = cell_selection
    elif cell_selection == 'random':
        cell_ids = np.random.choice(place_cell_ids, n_cells, replace=False)
    else:
        cell_ids = place_cell_ids[:n_cells]

    if ax is None:
        fig,ax = plt.subplots(figsize=(16,16))
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
    xax,yax = environment_size
    ax.set_xlim(xax)
    ax.set_ylim(yax)
    ax.set_xticks(xax)
    ax.set_yticks(yax)
    ax.set_xticklabels([0,f"{int((xax[1]-xax[0])/100)}m"])
    ax.set_yticklabels([0,f"{int((yax[1]-yax[0])/100)}m"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', length=0, width=.5, pad=1)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.legend()
    return fig
