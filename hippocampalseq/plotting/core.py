import os
import matplotlib.pyplot as plt
import functools

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

def save_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, file_path=None, file_name=None, **kwargs):
        __init_plotting() 
        res = fn(*args, **kwargs)
        if file_name is not None:
            if file_path is None:
                file_path = "./results/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(os.path.join(file_path, file_name), dpi=300)
        return res
    return wrapper