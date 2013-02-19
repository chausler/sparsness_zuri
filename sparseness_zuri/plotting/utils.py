import pylab as plt
import numpy as np

def do_box_plot(data, xval, c):
    box = plt.boxplot(data, positions=xval)
    plt.setp(box['boxes'], color=c, lw=1.5)
    plt.setp(box['medians'], color=c, lw=3)
    plt.setp(box['whiskers'], color='0.6')
    plt.setp(box['caps'], color=c)


def do_spot_scatter_plot(data, xval, c):
    plt.hold(True)
    for d in data:
        plt.scatter(xval + np.random.randn(1) * 0.05, d, c=c, marker='x')
    dd = data[data != 0]
    if len(dd) == 0:
        dd = 0
    else:
        dd = dd.mean()

    plt.plot([xval - 0.1, xval + 0.1], [dd, dd], c, lw=3)
    plt.show()

def adjust_spines(ax, spines, outward=None):
    for loc, spine in ax.spines.iteritems():
        if loc not in spines:
            spine.set_color('none')  # don't draw spine
        elif outward != None:
            spine.set_position(('outward', outward))

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
