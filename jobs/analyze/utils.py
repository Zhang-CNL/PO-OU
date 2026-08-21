import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

def get_major_axis(tsd: nap.TsdFrame) -> str:
    return 'x' if (
        np.max(tsd['x'].values) - np.min(tsd['x'].values) 
        >= np.max(tsd['y'].values) - np.min(tsd['y'].values)
    ) else 'y'

def extract_trajectory(tsd: nap.TsdFrame, environment_size: list[tuple[int,...]], track_type: str) -> np.ndarray:
    if track_type == 'Linear' and len(environment_size) == 1:
        return tsd[utils.get_major_axis(tsd)].values
    else:
        return tsd[['x','y']].values

def model_result_plot(
        subplot_row: int,
        subplot_col: int,
        subplot: int,
        trajectories: np.ndarray,
        environment_size: list[tuple[int,...]],
        cumprob: np.ndarray,
        title: str,
        aic: float|None = None,
        bic: float|None = None,
    ):
    plt.subplot(subplot_row,subplot_col,subplot)
    hsepl.plot_trajectories(
        trajectories,
        environment_size = environment_size,
    )
    plt.title(title + " trajectory."
        + f"\nAIC: {aic}, BIC: {bic}" if aic is not None else ""
    )
    plt.subplot(subplot_row,subplot_col,subplot+1)
    if cumprob.ndim < 2:
        cumprob = cumprob[...,None]
    plt.imshow(cumprob.T, origin='lower', cmap='hot')
    plt.title(title + " cumulative probability.")