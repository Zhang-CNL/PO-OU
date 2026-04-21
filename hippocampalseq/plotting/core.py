import os
import matplotlib.pyplot as plt
import functools

__plotting_initialized = False

def __init_plotting():
    global __plotting_initialized
    if __plotting_initialized:
        return
    __plotting_initialized = True
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=2, color='r')
    #plt.rcParams['font.sans-serif'] = ['Helvetica']


def save_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, file_path=None, file_name=None, **kwargs):
        __init_plotting() 
        res = fn(*args, **kwargs)
        if file_path is None:
            file_path = "./results/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if file_name is not None:
            plt.savefig(os.path.join(file_path, file_name), dpi=300)
        return res
    return wrapper