import matplotlib.pyplot as plt
import hippocampalseq.preprocessing as hsep

def init_plotting():
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

def plot_placefields(place_fields, pfs):
    fig, ax = plt.subplots(1,len(pfs), figsize=(2,.5), dpi=300)

    for i in range(len(pfs)):
        ax[i].imshow(place_fields[pfs[i]], origin='lower')
        #print(rat_data.PlaceFieldData['place_fields'][pfs[i]].max())
        
    ax[0].set_xticks([0,49])
    ax[0].set_xticklabels([0,"2m"])
    ax[0].set_yticks([0,49])
    ax[0].set_yticklabels([0,"2m"])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].tick_params(direction='out', length=0, width=.5, pad=1)

    for i in range(1,len(pfs)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    rect = plt.Rectangle(
        (0, 0), 1, 1, fill=False, color="k", lw=.5, alpha=.2,
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])

def spike_raster_plot(spike_data):
    plt.eventplot(spike_data, color='black', linelengths=.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Neurons")