import pylab as plt


def do_box_plot(data, xval, c):
    box = plt.boxplot(data, positions=xval)
    plt.setp(box['boxes'], color=c, lw=1.5)
    plt.setp(box['medians'], color=c, lw=3)
    plt.setp(box['whiskers'], color='0.6')
    plt.setp(box['caps'], color=c)


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
