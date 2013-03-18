import sys
sys.path.append('..')
from startup import *
import numpy as np
import scipy.stats
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
from data_utils.utils import do_thresh_corr, corr_trial_to_mean


# Sub directory of the figure path to put the plots in

exp_type = 'POP'
groups = ['R^2: Trial to Mean', 'Avg Activation', 'R^2: Trial Types']
colors = ['r', 'b', 'g', 'y', 'c', 'm']
headers = ['CellId', 'XCorr Center', 'XCorr Whole', 'XCorr Center',
                       'Avg Center', 'Avg Whole', 'Avg Surround',
                       'XCorr C v W', 'XCorr C v S', 'XCorr S v W']

norm = True
for randomise in [None]:
    if randomise is None:
        f_path = fig_path + 'Sparseness/%s/initial/norm/' % (exp_type)
    elif randomise == 'random':
        f_path = fig_path + 'Sparseness/%s/initial/random/' % (exp_type)

    fname = '%s%s' % (f_path, 'initial_summary.png')
    if os.path.exists(fname):
        print fname + ' exist, SKIPPING'
        continue
    exps = list_PopExps()

    csv_vals = []
    cellids = []
    for exp_id in exps:
        dat = load_PopData(exp_id)
        vals = []
        print 'doing ', exp_id
        if randomise is None:
            dat_c = dat['dat_c']
            dat_w = dat['dat_w']
        elif randomise == 'random':
            assert(False)
            dat_c = dat['dat_c']
            dat_w = dat['dat_w']
        print dat_c.shape, dat_c.mean(2).shape
        if not os.path.exists(f_path):
            os.makedirs(f_path)

        mn_c = dat_c.mean(2).mean(0)
        std_c = np.std(dat_c, 2).mean(0)
        xcorr_c = 0# corr_trial_to_mean(dat_c, mn_c)
        avg_c = dat_c.ravel().mean()
        vals.append(xcorr_c)

        mn_w = dat_w.mean(2).mean(0)
        std_w = np.std(dat_w, 2).mean(0)
        xcorr_w = 0# corr_trial_to_mean(dat_w, mn_w)
        avg_w = dat_w.mean()
        vals.append(xcorr_w)
        mx = np.maximum(mn_c.max(), mn_w.max())

        vals.append(avg_c)
        vals.append(avg_w)

        if norm:
            mn_c = mn_c / mx
            std_c = std_c / mx
            mn_w = mn_w / mx
            std_w = std_w / mx

        fig = plt.figure(figsize=(18, 12))
        fig.set_facecolor('white')
        clr_c = '0.3'
        clr_w = np.array([79, 79, 217]) / 255.

        ax = plt.subplot(421)
        plt.hold(True)
        plt.fill_between(range(len(std_c)), mn_c - std_c, mn_c + std_c,
                         facecolor='0.9')
        plt.plot(range(len(std_c)), mn_c, color=clr_c,
                 label='Center', linewidth=2)
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                   frameon=False, ncol=2)
        plt.text(1, plt.ylim()[1], 'corr trl v avg: %.2f' % (xcorr_c))
        adjust_spines(ax, ['bottom', 'left'])

        ax = plt.subplot(422)
        plt.hold(True)
        plt.fill_between(range(len(std_w)), mn_w - std_w, mn_w + std_w,
                         facecolor='0.9')
        plt.plot(range(len(std_w)), mn_w, color=clr_w,
                 label='Whole Field', linewidth=2)
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                   frameon=False, ncol=2)
        plt.text(1, plt.ylim()[1], 'corr trl v avg: %.2f' % (xcorr_w))
        adjust_spines(ax, ['bottom', 'left'])
        plt.title('Mean & STD')

        hist_mx = 0
        bins = np.arange(0, 1.01, 0.05)
        tks = bins[0::2]

        hst_ax1 = plt.subplot(423)
        cnts, _, _ = plt.hist(mn_c.ravel(), bins)
        y = plt.ylim()[1]
        plt.text(0.7, y * 0.9, 'Avg Act: %.2f' % avg_c)
        hist_mx = np.maximum(cnts.max(), hist_mx)
        plt.xticks(tks)
        adjust_spines(hst_ax1, ['bottom', 'left'])

        hst_ax2 = plt.subplot(424)
        cnts, _, _ = plt.hist(mn_w.ravel(), bins)
        y = plt.ylim()[1]
        plt.text(0.7, y * 0.9, 'Avg Act: %.2f' % avg_w)
        hist_mx = np.maximum(cnts.max(), hist_mx)
        plt.xticks(tks)
        plt.title('Mean Activation Histogram')
        adjust_spines(hst_ax2, ['bottom', 'left'])

        hst_ax1.set_ylim(0, hist_mx)
        hst_ax2.set_ylim(0, hist_mx)

        ax = plt.subplot(425)
        plt.hold(True)
        cr = do_thresh_corr(mn_w, mn_c)
        vals.append(cr)
        plt.plot(mn_w, label='Whole', color=clr_w, linewidth=1.5)
        #plt.fill_between(range(psth_w.shape[1]), mn_c, mn_w,
        #                facecolor=clr2)
        plt.plot(mn_c, color=clr_c,
                 label='Center', linewidth=2)
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
        plt.text(10, 0.9, 'corr: %.2f' % (cr))
        adjust_spines(ax, ['bottom', 'left'])
        ylim = plt.ylim(0, 1)
        plt.ylim([-0.01, ylim[1]])
        plt.title('Mean Comparison')

        ax = plt.subplot(426)
        plt.hold(True)
        plt.plot(mn_w - mn_c, label='Diff', color='0.3',
                 linewidth=1.5)  # 0.7  0.9
        plt.plot([0, dat_c.shape[1]], [0, 0], '--r')
        adjust_spines(ax, ['bottom', 'left'])
        plt.title('Differences')

        ax = plt.subplot(414, aspect='equal')
        plt.scatter(mn_c, mn_w)
        plt.xlabel('Center')
        plt.ylabel('Whole Field')
        xlm = plt.xlim()
        ylm = plt.ylim()
        plt.plot([-1, 1], [-1, 1], '--')
        plt.xlim(xlm)
        plt.ylim(ylm)
        adjust_spines(ax, ['bottom', 'left'])

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                                wspace=0.2, hspace=0.45)
        fname = '%s%s' % (f_path, exp_id)
        fig.savefig(fname + '.eps')
        fig.savefig(fname + '.png')
        #plt.show()
        plt.close(fig)
        csv_vals.append(vals)
        cellids.append(exp_id)

    csv_vals = np.array(csv_vals)
    fig = plt.figure()
    ax = plt.subplot(111)
    adjust_spines(ax, ['bottom', 'left'])
    xvals = []
    xlbls = []
    offset = -1
    divider = 3
    for i in range(csv_vals.shape[1]):
        col = colors[i % divider]
        if (i % divider == 0):
            offset += 2
            plt.text(offset, 1, groups[i / divider])

#                    do_box_plot(csv_vals[:, i], np.array([offset]), col,
#                                widths=[0.7])
        mean_adjust = False if i / divider == 1 else True
        do_spot_scatter_plot(csv_vals[:, i], offset, col,
                    width=0.7, mean_adjust=mean_adjust)
        inds = np.arange(divider)
        inds = inds[inds != i % divider]
        base_ind = (i / divider) * divider
        p_offset = -0.2
        if csv_vals[:, i].sum() > 0:
            for ind in inds:
                if csv_vals[:, base_ind + ind].sum() == 0:
                    continue
                stat, p = scipy.stats.ttest_ind(csv_vals[:, i],
                                        csv_vals[:, base_ind + ind])
                if p < 0.05:
                    plt.scatter(offset + p_offset, 0.95, c=colors[ind],
                                edgecolor=colors[ind],
                                marker='*')
                    p_offset *= -1
        xvals.append(offset)
        xlbls.append(headers[i + 1])
        offset += 1
    plt.xticks(xvals, xlbls, rotation='vertical')
    plt.xlim(0, offset)
    plt.ylim(0, 1.15)
    plt.subplots_adjust(left=0.05, bottom=0.21, right=0.98, top=0.98,
               wspace=0.3, hspace=0.34)

    fname = '%s%s' % (f_path, 'initial_summary')
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)

    f = open(f_path + 'corrs.csv', 'w')
    f.write("\t".join(headers) + '\n')
    for c, v in zip(cellids, csv_vals):
        v = ['%.2f' % vv for vv in v]
        f.write(c + "\t")
        f.write("\t".join(v))
        f.write('\n')
    f.close()

