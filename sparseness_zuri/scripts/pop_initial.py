import sys
sys.path.append('..')
from startup import *
import numpy as np
import scipy.stats
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, do_point_line_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
from data_utils.utils import do_thresh_corr, corr_trial_to_mean, corr_trial_to_mean_multi,\
    average_corrs


# Sub directory of the figure path to put the plots in

exp_type = 'POP'
groups = ['Corr: Trial to Mean', 'Avg Activation', 'Corr: Trial Types']
colors = ['r', 'b', 'g', 'y', 'c', 'm']
headers = ['CellId', 'XCorr Center', 'XCorr Whole',
                       'Avg Center', 'Avg Whole',
                       'XCorr C v W']

norm = True
for randomise in [None]:
    if randomise is None:
        f_path = fig_path + 'Sparseness/%s/initial/norm/' % (exp_type)
    elif randomise == 'random':
        f_path = fig_path + 'Sparseness/%s/initial/random/' % (exp_type)

    fname = '%s%s' % (f_path, 'initial_summary.png')
    if os.path.exists(fname):
        print fname + ' exist, SKIPPING'
#        continue
    exps = list_PopExps()
    csv_vals = [[], [], [], [], []]
    sum_vals = []
    cellids = []
    #exps = ['121127']
    for exp_id in exps:
        dat = load_PopData(exp_id, True)
        vals = []
        print 'doing ', exp_id
        if randomise is None:
            dat_c = dat['dat_raw_c']
            dat_w = dat['dat_raw_w']
        elif randomise == 'random':
            assert(False)
            dat_c = dat['dat_raw_c']
            dat_w = dat['dat_raw_w']

        if not os.path.exists(f_path):
            os.makedirs(f_path)

        mn_c = dat_c.mean(2).mean(0)
        std_c = np.std(dat_c, 2).mean(0)
        xcorr_c, cell_xcorr_c = corr_trial_to_mean_multi(dat_c)
        vals.append(xcorr_c)
        csv_vals[0] += cell_xcorr_c.tolist()

        mn_w = dat_w.mean(2).mean(0)
        std_w = np.std(dat_w, 2).mean(0)
        xcorr_w, cell_xcorr_w = corr_trial_to_mean_multi(dat_w)
        vals.append(xcorr_w)
        csv_vals[1] += cell_xcorr_w.tolist()
        mx = np.maximum(mn_c.max(), mn_w.max())

        mn_c = mn_c / mx
        std_c = std_c / mx
        mn_w = mn_w / mx
        std_w = std_w / mx
        avg_c = mn_c.mean()
        avg_w = mn_w.mean()
        csv_vals[2] += dat_c.mean(2).mean(1).tolist()
        csv_vals[3] += dat_w.mean(2).mean(1).tolist()
        vals.append(avg_c)
        vals.append(avg_w)

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
        plt.text(5, plt.ylim()[1] * 0.9,
                 'corr trl-trl: %.2f,  #cells: : %d,  #trials: %d' %
                 (xcorr_c, dat_c.shape[0], dat_c.shape[2]))
        adjust_spines(ax, ['bottom', 'left'])

        ax = plt.subplot(422)
        plt.hold(True)
        plt.fill_between(range(len(std_w)), mn_w - std_w, mn_w + std_w,
                         facecolor='0.9')
        plt.plot(range(len(std_w)), mn_w, color=clr_w,
                 label='Whole Field', linewidth=2)
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                   frameon=False, ncol=2)
        plt.text(5, plt.ylim()[1] * 0.9,
                 'corr trl-trl: %.2f,  #cells: : %d,  #trials: %d' %
                 (xcorr_w, dat_w.shape[0], dat_w.shape[2]))
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
        crrs = []
        for c, w in zip(dat_c.mean(2), dat_w.mean(2)):
            crrs.append(do_thresh_corr(c, w))
        cr = average_corrs(crrs)
        vals.append(cr)
        csv_vals[4] += crrs

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
        cellids.append(exp_id)
        sum_vals.append(vals)

    fig = plt.figure()
    ax = plt.subplot(111)
    adjust_spines(ax, ['bottom', 'left'])
    xvals = []
    xlbls = []
    offset = 0
    divider = 2
    for i in range(len(groups)):
        base_ind = i * divider
        print base_ind
        dt = np.array(csv_vals[base_ind: base_ind + divider]).T
        offsets = np.arange(dt.shape[1]) + offset
        mean_adjust = (i != 1)
        do_point_line_plot(dt, offsets, mean_adjust=mean_adjust)
#        for j in range(dt.shape[1]):
#            p_offset = -0.2
#            if dt[j].sum() > 0:
#                for ind in range(j + 1, dt.shape[1]):
#                    if dt[ind].sum() == 0:
#                        continue
#                    stat, p = scipy.stats.ttest_ind(dt[j], dt[ind])
#                    print 'p', p, j, ind
#                    if p < 0.05:
#                        plt.scatter(offset + j + p_offset,
#                                    0.95, c=colors[ind],
#                                    edgecolor=colors[ind],
#                                    marker='*')
#                        p_offset *= -1

#        if base_ind == 1:
#            do_spot_scatter_plot(dt, offset, col,
#                    width=0.7, mean_adjust=False)
#        inds = np.arange(divider)
#        inds = inds[inds != i % divider]
#        p_offset = -0.2
#        if dt.sum() > 0:
#            for ind in inds:
#                if (base_ind + ind) >= len(csv_vals):
#                        continue
#                dt2 = np.array(csv_vals[base_ind + ind])
#                if dt2.sum() == 0:
#                    continue
#                stat, p = scipy.stats.ttest_ind(dt,
#                                        dt2)
#                if p < 0.05:
#                    plt.scatter(offset + p_offset, 0.95, c=colors[ind],
#                                edgecolor=colors[ind],
#                                marker='*')
#                    p_offset *= -1
        xvals += offsets.tolist()
        xlbls += headers[1 + base_ind: base_ind + divider + 1]
        plt.text(offset - 0.5, 1, groups[i])
        offset += 2

    plt.xticks(xvals, xlbls, rotation='vertical')
    #plt.xlim(0, offset + 0.5)
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
    for c, v in zip(cellids, sum_vals):
        v = ['%.2f' % vv for vv in v]
        f.write(c + "\t")
        f.write("\t".join(v))
        f.write('\n')
    f.close()

