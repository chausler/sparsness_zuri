import sys
sys.path.append('..')
from startup import *
import numpy as np
import scipy.stats
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot
from data_utils.load_ephys import load_EphysData
import os
from data_utils.utils import do_thresh_corr, corr_trial_to_mean


# Sub directory of the figure path to put the plots in

exp_types = ['FS', 'PYR', 'SOM']
groups = ['R^2: Trial to Mean', 'Avg Activation', 'R^2: Trial Types']
colors = ['r', 'b', 'g', 'y', 'c', 'm']
headers = ['CellId', 'XCorr Center', 'XCorr Whole', 'XCorr Center',
                       'Avg Center', 'Avg Whole', 'Avg Surround',
                       'XCorr C v W', 'XCorr C v S', 'XCorr S v W']
filters = [0.1]
norm = True
for exp_type in exp_types:
    for filt in filters:
        for randomise in [None, 'generated', 'random']:
            dat = load_EphysData(exp_type, filt=filt)
            csv_vals = []
            cellids = []
            for k in sorted(dat.keys()):
                vals = []
                d = dat[k]
                print 'doing ', d['cellid']
                if randomise is None:
                    f_path = fig_path + 'ephys/%s/initial/%.2f/' % (exp_type, filt)
                    psth_s = d['psth_s']
                    psth_c = d['psth_c']
                    psth_w = d['psth_w']
                elif randomise == 'random':
                    f_path = fig_path + 'ephys/%s/initial/%.2f_random/' % (exp_type, filt)
                    psth_c = d['psth_c_rand']
                    psth_s = d['psth_s_rand']
                    psth_w = d['psth_w_rand']
                elif randomise == 'generated':
                    f_path = fig_path + 'ephys/%s/initial/%.2f_generated/' % (exp_type, filt)
                    psth_c = d['psth_c_gen']
                    psth_s = d['psth_s_gen']
                    psth_w = d['psth_w_gen']
                if not os.path.exists(f_path):
                    os.makedirs(f_path)
                edge = d['edge']
                mn_c = psth_c.mean(0)
                std_c = np.std(psth_c, 0)
                xcorr_c = corr_trial_to_mean(psth_c, mn_c)
                vals.append(xcorr_c)
                mn_w = psth_w.mean(0)
                std_w = np.std(psth_w, 0)
                xcorr_w = corr_trial_to_mean(psth_w, mn_w)
                vals.append(xcorr_w)
                mx = np.maximum(mn_c.max(), mn_w.max())
                if psth_s is not None:
                    mn_s = psth_s.mean(0)
                    std_s = np.std(psth_s, 0)
                    xcorr_s = corr_trial_to_mean(psth_s, mn_s)
                    vals.append(xcorr_s)
                    mx = np.maximum(mx, mn_s.max())
                else:
                    vals.append(0)
                if norm:
                    mn_c = mn_c / mx
                    std_c = std_c / mx
                    mn_w = mn_w / mx
                    std_w = std_w / mx
                    if psth_s is not None:
                        mn_s = mn_s / mx
                        std_s = std_s / mx

                fig = plt.figure(figsize=(18, 12))
                fig.set_facecolor('white')
                clr_c = '0.3'
                clr_w = np.array([79, 79, 217]) / 255.
                clr_s = np.array([255, 151, 115]) / 300.

                ax = plt.subplot(531)
                plt.hold(True)
                plt.fill_between(range(len(std_c)), mn_c - std_c, mn_c + std_c,
                                 facecolor='0.9')
                plt.plot(range(len(std_c)), mn_c, color=clr_c,
                         label='Center', linewidth=2)
                plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                           frameon=False, ncol=2)
                plt.text(1, plt.ylim()[1], 'corr trl v avg: %.2f' % (xcorr_c))
                adjust_spines(ax, ['bottom', 'left'])
                ax = plt.subplot(532)
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

                if psth_s is not None:
                    ax = plt.subplot(533)
                    plt.hold(True)
                    plt.fill_between(range(len(std_s)), mn_s - std_s,
                                     mn_s + std_s, facecolor='0.9')
                    plt.plot(range(len(std_s)), mn_s, color=clr_s,
                             label='Surround', linewidth=2)
                    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                               frameon=False, ncol=2)
                    plt.text(1, plt.ylim()[1], 'corr trl v avg: %.2f'
                             % (xcorr_s))
                    adjust_spines(ax, ['bottom', 'left'])

                hist_mx = 0
                bins = np.arange(0, 1.01, 0.05)
                tks = bins[0::2]
                hst_ax1 = plt.subplot(534)
                cnts, _, _ = plt.hist(mn_c.ravel(), bins)
                y = plt.ylim()[1]
                avg = mn_c.ravel().mean()
                vals.append(avg)
                plt.text(0.7, y * 0.9, 'Avg Act: %.2f' % avg)
                hist_mx = np.maximum(cnts.max(), hist_mx)
                plt.xticks(tks)
                adjust_spines(hst_ax1, ['bottom', 'left'])

                hst_ax2 = plt.subplot(535)
                cnts, _, _ = plt.hist(mn_w.ravel(), bins)
                y = plt.ylim()[1]
                avg = mn_w.ravel().mean()
                vals.append(avg)
                plt.text(0.7, y * 0.9, 'Avg Act: %.2f' % avg)
                hist_mx = np.maximum(cnts.max(), hist_mx)
                plt.xticks(tks)
                plt.title('Mean Activation Histogram')
                adjust_spines(hst_ax2, ['bottom', 'left'])

                if psth_s is not None:
                    ax = plt.subplot(536)
                    cnts, _, _ = plt.hist(mn_s.ravel(), bins)
                    y = plt.ylim()[1]
                    avg = mn_s.ravel().mean()
                    vals.append(avg)
                    plt.text(0.7, y * 0.9, 'Avg Act: %.2f' % avg)
                    hist_mx = np.maximum(cnts.max(), hist_mx)
                    plt.xticks(tks)
                    adjust_spines(ax, ['bottom', 'left'])
                    ax.set_ylim(0, hist_mx)
                else:
                    vals.append(0)

                hst_ax1.set_ylim(0, hist_mx)
                hst_ax2.set_ylim(0, hist_mx)

                ax = plt.subplot(537)
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

                ax = plt.subplot(538)
                plt.title('Mean Comparison')
                if psth_s is not None:
                    plt.hold(True)
                    cr = do_thresh_corr(mn_s, mn_c)
                    vals.append(cr)
                    plt.plot(mn_s, label='Surround', color=clr_s,
                             linewidth=1.5)
                    #plt.fill_between(range(psth_s.shape[1]), mn_c,
                    #                   psth_s.mean(0), facecolor=clr1)
                    plt.plot(mn_c, color=clr_c,
                             label='Center', linewidth=2)
                    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                               frameon=False, ncol=2)
                    plt.text(10, 0.9, 'corr: %.2f' % (cr))
                    adjust_spines(ax, ['bottom', 'left'])
                    ylim = plt.ylim(0, 1)
                    plt.ylim([-0.01, ylim[1]])

                    ax = plt.subplot(539)
                    plt.hold(True)
                    cr = do_thresh_corr(mn_s, mn_w)
                    vals.append(cr)
                    plt.plot(mn_s, label='Surround', color=clr_s,
                             linewidth=1.5)
                    #plt.fill_between(range(psth_s.shape[1]), mn_c,
                    #                   psth_s.mean(0), facecolor=clr1)
                    plt.plot(mn_w, color=clr_w,
                             label='Whole', linewidth=2)
                    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1),
                               frameon=False, ncol=2)
                    plt.text(10, 0.9, 'corr: %.2f' % (cr))
                    adjust_spines(ax, ['bottom', 'left'])
                    ylim = plt.ylim(0, 1)
                    plt.ylim([-0.01, ylim[1]])
                else:
                    vals.append(0)
                    vals.append(0)

                ax = plt.subplot(5, 3 ,10)
                plt.hold(True)
                plt.plot(mn_w - mn_c, label='Diff', color='0.3',
                         linewidth=1.5)  # 0.7  0.9
                plt.plot([0, psth_c.shape[1]], [0, 0], '--r')
                adjust_spines(ax, ['bottom', 'left'])

                ax = plt.subplot(5, 3, 11)
                plt.title('Differences')
                if psth_s is not None:
                    plt.hold(True)
                    plt.plot(mn_s - mn_c, label='Diff',
                             color='0.3', linewidth=1.5)  # 0.7  0.9
                    plt.plot([0, psth_c.shape[1]], [0, 0], '--r')
                    #plt.plot([0,0],[0,psth_c.shape[1]],':')
                    adjust_spines(ax, ['bottom', 'left'])

                    ax = plt.subplot(5, 3, 12)
                    plt.hold(True)
                    plt.plot(mn_s - mn_w, label='Diff',
                             color='0.3', linewidth=1.5)  # 0.7  0.9
                    plt.plot([0, psth_c.shape[1]], [0, 0], '--r')
                    #plt.plot([0,0],[0,psth_c.shape[1]],':')
                    adjust_spines(ax, ['bottom', 'left'])

                ax = plt.subplot(5, 3, 13, aspect='equal')
                plt.scatter(mn_c, mn_w)
                plt.xlabel('Center')
                plt.ylabel('Whole Field')
                xlm = plt.xlim()
                ylm = plt.ylim()
                plt.plot([-1, 1], [-1, 1], '--')
                plt.xlim(xlm)
                plt.ylim(ylm)
                adjust_spines(ax, ['bottom', 'left'])

                ax = plt.subplot(5, 3, 14, aspect='equal')
                plt.title('Activation Comparison')
                if psth_s is not None:
                    plt.scatter(mn_c, mn_s)
                    plt.xlabel('Center')
                    plt.ylabel('Surround')
                    xlm = plt.xlim()
                    ylm = plt.ylim()
                    plt.plot([-1, 1], [-1, 1], '--')
                    plt.xlim(xlm)
                    plt.ylim(ylm)
                    adjust_spines(ax, ['bottom', 'left'])
                    ax = plt.subplot(5, 3, 15, aspect='equal')
                    plt.scatter(mn_w, mn_s)
                    plt.xlabel('Whole Field')
                    plt.ylabel('Surround')
                    xlm = plt.xlim()
                    ylm = plt.ylim()
                    plt.plot([-1, 1], [-1, 1], '--')
                    plt.xlim(xlm)
                    plt.ylim(ylm)
                    adjust_spines(ax, ['bottom', 'left'])

                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                                        wspace=0.2, hspace=0.45)
                fname = '%s%s' % (f_path, d['cellid'])
                fig.savefig(fname + '.eps')
                fig.savefig(fname + '.png')
                #plt.show()
                plt.close(fig)
                csv_vals.append(vals)
                cellids.append(d['cellid'])


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
                if csv_vals[:, i].sum() > 0:
                    do_box_plot(csv_vals[:, i], np.array([offset]), col,
                                widths=[0.7])

                    inds = np.arange(divider)
                    inds = inds[inds != i % divider]
                    base_ind = i / divider
                    p_offset = -0.2
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
            plt.ylim(0, 1.05)
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

