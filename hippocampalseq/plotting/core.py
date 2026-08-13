import os
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

__plotting_initialized = False

def change_font_sizes(small_size, medium_size, big_size):
    plt.rc('font', size=small_size, family='sans-serif')          # controls default text sizes
    plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)    # legend fontsize
    plt.rc('axes', titlesize=big_size)     # fontsize of the axes title
    plt.rc('figure', titlesize=big_size)  # fontsize of the figure title
    plt.rc('lines', linewidth=2, color='r')
    plt.rcParams['figure.dpi'] = 300
    #plt.rcParams['font.sans-serif'] = ['Helvetica']

def __init_plotting():
    global __plotting_initialized
    if __plotting_initialized:
        return
    __plotting_initialized = True
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    change_font_sizes(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)


def reset_plotting():
    global __plotting_initialized
    __plotting_initialized = False

def save_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, file_path: str=None, file_name: str|list[str]=None, **kwargs):
        __init_plotting() 
        res = func(*args, **kwargs)
        if res is not None and not isinstance(res, (list, tuple)):
            res = [res]
            if file_name is not None and len(res) > 0:
                if file_path is None:
                    file_path = "./results/"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                if ".pdf" in file_name:
                    with PdfPages(os.path.join(file_path, file_name)) as pdf:
                        for fig in res:
                            pdf.savefig(fig)
                else:
                    for i,fig in enumerate(res):
                        fn = str(i) + file_name if i > 0 else file_name
                        fig.savefig(os.path.join(file_path, fn))
        return res
    return wrapper


def colored_line(
        x: np.ndarray, 
        y: np.ndarray, 
        values: np.ndarray, 
        ax=None, 
        cmap='viridis', 
        norm=None, 
        **lc_kwargs
    ):
    """
    Plot a line whose color varies along its length according to `values`.

    Args:
        x, y (np.ndarray): Coordinates of the line (e.g. position over time).
        values (np.ndarray): Same length as x, y. Value used to color each segment (e.g. velocity).
        ax (plt.Axes): Axes to plot on. Uses current axes if not given. Defaults to None.
        cmap (str or Colormap): Colormap to use. Defaults to 'viridis'.
        norm (matplotlib.colors.Normalize): Normalization for the colormap. Defaults to min/max of `values`.
        **lc_kwargs: Passed through to LineCollection (e.g. linewidth).

    Returns:
        (LineCollection) The plotted collection (useful for adding a colorbar).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    if ax is None:
        ax = plt.gca()

    # Build an array of line segments: each segment connects point i to i+1
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if norm is None:
        norm = Normalize(vmin=values.min(), vmax=values.max())

    lc = LineCollection(segments, cmap=cmap, norm=norm, **lc_kwargs)
    lc.set_array(values[:-1])

    line = ax.add_collection(lc)
    ax.autoscale()  # add_collection doesn't autoscale axes automatically

    return line