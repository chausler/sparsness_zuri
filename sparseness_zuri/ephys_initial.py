import sys
sys.path.append('..')
from startup import *
import numpy as np
import pylab as plt
from plotting.utils import adjust_spines
from data_utils.load_ephys import load_EphysData_SOM

# Sub directory of the figure path to put the plots in
fig_path = fig_path + 'ephys/som/'
dat = load_EphysData_SOM()

for d in dat.values():

    print 'doing ', d['expdate']
    psth_s = d['psth_s']
    psth_c = d['psth_c']
    psth_w = d['psth_w']

    fig = plt.figure(figsize=(16, 9))
    fig.set_facecolor('white')
    clr1 = np.array([252, 104, 149]) / 255.
    clr2 = np.array([255, 151, 115]) / 255.

    ax = plt.subplot(531)
    plt.hold(True)
    mn = psth_c.mean(0)
    std = np.std(psth_c, 0)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor='0.9')
    plt.plot(range(len(std)), mn, color='0.3',
             label='Center', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])

    ax = plt.subplot(532)
    plt.hold(True)
    mn = psth_w.mean(0)
    std = np.std(psth_w, 0)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor='0.9')
    plt.plot(range(len(std)), mn, color='0.3',
             label='Whole Field', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Mean & STD')

    ax = plt.subplot(533)
    plt.hold(True)
    mn = psth_s.mean(0)
    std = np.std(psth_s, 0)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor='0.9')
    plt.plot(range(len(std)), mn, color='0.3',
             label='Surround', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])

    ax = plt.subplot(523)
    plt.hold(True)
    plt.plot(psth_w.mean(0), label='Whole', color=clr2,)  # 0.7  0.9
    plt.fill_between(range(psth_w.shape[1]), psth_c.mean(0), psth_w.mean(0),
                     facecolor=clr2)
    plt.plot(psth_c.mean(0), color='0.3',
             label='Center', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    ylim = plt.ylim()
    plt.ylim([-0.01, ylim[1]])
    plt.title('Differences')

    ax = plt.subplot(524)
    plt.hold(True)
    plt.plot(psth_s.mean(0), label='Surround', color=clr1,)  # 0.7  0.9
    plt.fill_between(range(psth_s.shape[1]), psth_c.mean(0), psth_s.mean(0),
                     facecolor=clr1)
    plt.plot(psth_c.mean(0), color='0.3',
             label='Center', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.45)
    ylim = plt.ylim()
    plt.ylim([-0.01, ylim[1]])
    plt.title('Differences')

    ax = plt.subplot(525)
    plt.hold(True)
    plt.plot(psth_w.mean(0) - psth_c.mean(0), label='Diff', color='0.3',
             linewidth=1.5)  # 0.7  0.9
    plt.plot([0, psth_c.shape[1]], [0, 0], '--r')

    adjust_spines(ax, ['bottom', 'left'])

    ax = plt.subplot(526)
    plt.hold(True)
    plt.plot(psth_s.mean(0) - psth_c.mean(0), label='Diff',
             color='0.3', linewidth=1.5)  # 0.7  0.9
    plt.plot([0, psth_c.shape[1]], [0, 0], '--r')
    #plt.plot([0,0],[0,psth_c.shape[1]],':')
    adjust_spines(ax, ['bottom', 'left'])

    bins = np.arange(4) - 0.5
    tks = np.arange(4)
    ax = plt.subplot(5, 3, 10)
    plt.hist(psth_c.ravel(), bins)
    plt.xticks(tks)
    adjust_spines(ax, ['bottom', 'left'])
    ax = plt.subplot(5, 3, 11)
    plt.hist(psth_w.ravel(), bins)
    plt.xticks(tks)
    plt.title('Spike Count Histogram')
    adjust_spines(ax, ['bottom', 'left'])
    ax = plt.subplot(5, 3, 12)
    plt.hist(psth_s.ravel(), bins)
    plt.xticks(tks)
    adjust_spines(ax, ['bottom', 'left'])

    ax = plt.subplot(529, aspect='equal')
    plt.scatter(psth_c.mean(0), psth_w.mean(0))
    plt.xlabel('Center')
    plt.ylabel('Whole Field')
    xlm = plt.xlim()
    ylm = plt.ylim()
    plt.plot([-1, 1], [-1, 1], '--')
    plt.xlim(xlm)
    plt.ylim(ylm)
    adjust_spines(ax, ['bottom', 'left'])

    ax = plt.subplot(5, 2, 10, aspect='equal')
    plt.scatter(psth_c.mean(0), psth_s.mean(0))
    plt.xlabel('Center')
    plt.ylabel('Surround')
    xlm = plt.xlim()
    ylm = plt.ylim()
    plt.plot([-1, 1], [-1, 1], '--')
    plt.xlim(xlm)
    plt.ylim(ylm)
    adjust_spines(ax, ['bottom', 'left'])

    fname = '%s%s' % (fig_path, d['expdate'])
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()    
    plt.close(fig)
