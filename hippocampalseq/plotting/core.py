import os
import functools
import matplotlib.pyplot as plt
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
        if not isinstance(res, list):
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