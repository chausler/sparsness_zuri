import pylab as plt
import numpy as np
from data_utils.utils import average_corrs
import scipy.stats

def do_box_plot(data, xval, c, widths=[1]):
    dd = data#[data != 0]
    #plt.text(xval - 0.05, 0.9, '#%d' % len(dd))
    if len(dd) > 0:
        box = plt.boxplot(dd, positions=xval, widths=widths)
        plt.setp(box['boxes'], color=c, lw=1.5)
        plt.setp(box['medians'], color=c, lw=3)
        plt.setp(box['whiskers'], color='0.6')
        plt.setp(box['caps'], color=c)


def do_spot_scatter_plot(data, xval, c='k', width=0.2, text=True,
                         mean_adjust=True):
    plt.hold(True)
    dd = data[data != 0]
    r = width / 2.
    for d in data:
        plt.scatter(xval + (np.random.rand(1) - 0.5) * r, d, c=c, marker='x')
    if len(data) > 0:
        # fishers z transform for mean and then transform back
        if mean_adjust:
            mn = average_corrs(data)
        else:
            mn = data.mean()
        plt.plot([xval - r, xval + r], [mn, mn], 'k', lw=3)
    plt.grid(True, axis='y')
    if text:
        plt.text(xval - r, 0.9, '#%d' % len(dd))


def do_point_line_plot(data, xvals, c=['r', 'b'], width=0.2,
                         mean_adjust=True, text=True, alpha=0.3):
    plt.hold(True)
    dd = data[data.sum(1) != 0]
    r = width / 2.
    ind = np.argsort(data[:, 0])
    data = data[ind]
    for d in data:
        offset = (np.random.rand(1) - 0.5) * r
        x = xvals + offset
        if len(x) > 1:
            plt.plot(x, d, '-', c='0.7', lw=0.5, alpha=alpha)
        for i in range(len(x)):
            plt.scatter(x[i], d[i], c=c[i], marker='x', alpha=alpha)
    if len(data) > 0:
        # fishers z transform for mean and then transform back
        for i in range(len(xvals)):
            if mean_adjust:
                mn = average_corrs(data[:, i])
            else:
                mn = data[:, i].mean()
            plt.plot([xvals[i] - r, xvals[i] + r], [mn, mn], c[i], lw=4)
            plt.plot([xvals[i] - r, xvals[i] + r], [mn, mn], 'k', lw=1.5)

        for j in range(data.shape[1]):
            if data[:, j].sum() == 0:
                        continue
            p_offset = -0.2
            for ind in range(j + 1, data.shape[1]):
                    if data[:, ind].sum() == 0:
                        continue
                    if mean_adjust:
                        _, p = scipy.stats.ttest_ind(np.arctanh(data[:, j]),
                                                     np.arctanh(data[:, ind]))
                    else:
                        _, p = scipy.stats.ttest_ind(data[:, j], data[:, ind])
                    print 'p', p, j, ind
                    if p < 0.05:
                        plt.scatter(xvals[j] + p_offset,
                                    0.95, c=c[ind],
                                    edgecolor=c[ind],
                                    marker='*')
                        p_offset *= -1
    plt.grid(True, axis='y')
    plt.text(xvals[0] - r, 0.9, '#%d' % len(dd))


def plot_mean_std(xs, mn, std, title, plt_num=[1, 1, 1], legend=True,
                  clr1='0.3',
                  clr2='0.9', label='Trial Mean', line='-'):
    """plot the mean +/- std as timeline """
    if xs  is None:
        xs = range(len(mn))
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.fill_between(xs, mn - std, mn + std, facecolor=clr2)
    plt.plot(xs, mn, line, color=clr1,
             label=label, linewidth=2)
    if legend:
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False,
                   ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)
    return ax


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


if __name__ == "__main__":
    dat = np.random.randn(10, 2)
    dat[:, 1] += 2
    do_point_line_plot(dat, [0,1])
    plt.show()
    