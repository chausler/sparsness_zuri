import pylab as plt
import numpy as np

def do_box_plot(data, xval, c, widths=[1]):
    dd = data[data != 0]
    plt.text(xval - 0.05, 0.9, '#%d' % len(dd))
    if len(dd) > 0:
        box = plt.boxplot(dd, positions=xval, widths=widths)
        plt.setp(box['boxes'], color=c, lw=1.5)
        plt.setp(box['medians'], color=c, lw=3)
        plt.setp(box['whiskers'], color='0.6')
        plt.setp(box['caps'], color=c)


def do_spot_scatter_plot(data, xval, c):
    plt.hold(True)
    dd = data[data != 0]
    for d in dd:
        plt.scatter(xval + np.random.randn(1) * 0.02, d, c=c, marker='x')
    if len(dd) > 0:
        # fishers z transform for mean and then transform back
        mn = np.tanh(np.arctanh(dd).mean())
        plt.plot([xval - 0.1, xval + 0.1], [mn, mn], c, lw=3)
    plt.grid(True, axis='y')
    plt.text(xval - 0.05, 0.9, '#%d' % len(dd))


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
