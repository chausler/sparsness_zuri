import pickle
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, plot_mean_std
import numpy.fft as fft
import pylab as plt
import numpy as np
import os

# The bin size in degrees for flow directions
bin_ang = 45
div = 360. / bin_ang
bins = np.arange(-180 - 180 / div, 181 + 180 / div, 360 / div)

# Some Colours for plotting
clr1 = '0.3'
clr2 = '0.9'
clr3 = np.array([255, 151, 115]) / 255.
clr4 = np.array([200, 105, 75]) / 255.



def reverse_fft(four_weights):
        A = fft.ifft2(four_weights)
        return np.sqrt(A.real ** 2 + A.imag ** 2)

def plot_four(four_weights, ylims, plt_num=[1, 1, 1], title=None,
              ax_off=False, interpolation='None'):
    """plot the weights of the spatial fourier """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    ext = four_weights.shape[0] / 2.
    _ = plt.imshow(four_weights.T, cmap=plt.cm.gray,
                    interpolation=interpolation,
               extent=(-ext, ext, -ext, ext))
    plt.clim(ylims)
    if ax_off:
        adjust_spines(ax, [])
    else:
        adjust_spines(ax, ['bottom', 'left'])        
    if title is None:
        title = 'Filter'
    plt.title(title)
#    cax = plt.subplot(244)
#    plt.colorbar(im, cax=cax, cmap=plt.cm.gray)
#    plt.title('Colour Scale')


def plot_summary(cmb_corrs, fig_path, extras=''):
    """ plot a summary showing the mean and std for all attempted
    combinations"""
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(111)
    adjust_spines(ax, ['bottom', 'left'])
    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    _ = plt.subplot(111)
    xaxis = []
    boxes = []
    lbls = []

    #targets.append('Overall')
    for i, cmb in enumerate(sorted(cmb_corrs.keys())):
        xaxis.append(cmb)
        offset = 0
        for j, k in enumerate(sorted(cmb_corrs[cmb].keys())):
            dat = cmb_corrs[cmb][k]
            if len(dat) == 0:
                continue
            col = colors[j]
            do_box_plot(np.array(dat), np.array([i + offset]),
                        col, widths=[0.2])
            #do_spot_scatter_plot(np.array(dat), np.array([i + offset]), col)
            offset += 0.3
            if i == 0:
                boxes.append(plt.Rectangle((0, 0), 1, 1, fc=col))
                lbls.append(k)
                if j == 0:
                    plt.title('N=%d' % len(dat))

    plt.xticks(range(len(xaxis)), xaxis, rotation='vertical')
    plt.xlim(-1, len(xaxis) + 1)
    plt.plot([-1, len(xaxis) + 1], [0, 0], '--')
    plt.ylim(0, 1)
    plt.subplots_adjust(left=0.05, bottom=0.5, right=0.97, top=0.95,
                       wspace=0.3, hspace=0.34)
    plt.legend(boxes, lbls, frameon=False, loc=4)
    plt.ylabel('Correlation between prediction and Experimental Mean')
    fname = '%s%s_pred_%s' % (fig_path, 'summary', extras)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)


def plot_cell_summary(res, fig_path, extras=''):
    colors = ['0.5', '0.05', 'k', 'r', 'g', 'y', 'c', 'm']
    summary = []
    for cnum, cell in enumerate(sorted(res.keys())):
        vals = []
        maxes = {}
        x = 0
        lbls = []
        xtcks = []
        fig = plt.figure(figsize=(14, 12))
        ax = plt.subplot(111)
        plt.hold(True)
        for cmb in sorted(res[cell].keys()):
            x += 1
            plt.text(x, 0.95, cmb, rotation=60, verticalalignment='bottom')
            for i, k in enumerate(sorted(res[cell][cmb])):
                val = res[cell][cmb][k]['crr_pred']
                lbls.append(k)
                if k in maxes:
                    if val > maxes[k][1]:
                        maxes[k] = [cmb, val]
                else:
                    maxes[k] = [cmb, val]
                xtcks.append(x)
                plt.plot([x, x], [0, val], ':', color='0.3')
                plt.scatter(x, val, c=colors[i], s=200)
                x += 1
                if val > 0:
                    vals.append([cmb, '%s' % (k), '%.2f' % val])
            x += 2
        plt.plot([0, x], [0, 0], '--')
        plt.xlim(0, x)
        plt.ylim(0, 1)
        plt.xticks(xtcks, lbls, rotation='vertical')
        adjust_spines(ax, ['bottom', 'left'])
        plt.ylabel('Prediction Correlation to Experimental Mean')
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.97, top=0.92,
                       wspace=0.3, hspace=0.34)
        fig.savefig(fig_path + 'summary_' + cell + '.eps')
        fig.savefig(fig_path + 'summary_' + cell + '.png')
        plt.close(fig)

        f = open(fig_path + 'summary_' + cell + '.csv', 'w')
        if len(vals) == 0:
            vals.append(['', '', ''])
        for v in vals:
            f.write("\t".join(v))
            f.write('\n')
        f.close()
        line2 = '%s\t' % cell
        line1 = '\t'
        for mm in sorted(maxes.keys()):
            line1 += '%s\t\t' % mm
            if maxes[mm][1] > 0:
                line2 += '%.2f\t%s\t' % (maxes[mm][1], maxes[mm][0])
            else:
                line2 += '\t\t'
        if cnum == 0:
            summary.append(line1 + '\n')
        summary.append(line2 + '\n')

    f = open(fig_path + 'summary_all.csv', 'w')
    for l in summary:
        f.write(l)
    f.close()


def plot_preds_fourier(cellid, results, fname):
    fig = plt.figure(figsize=(18, 9))
    fig.set_facecolor('white')
    plt.suptitle('Experiment: %s  Fourier Source' % (cellid))
    cols = len(results)
    legend = True
    for i, [t, dat] in enumerate(results.iteritems()):
        #dat = results[t]
        pred = dat['pred']
        mn = dat['mn']
        std = dat['mn']
        crr_pred = dat['crr_pred']
        crr_exp = dat['crr_exp']
        weights = dat['coefs'].mean(0)
        plot_params = dat['plot_params']

        plot_mean_std(None, mn, std, '%s - Corr Trials to Mean: %.2f' %
                            (t, crr_exp), [3, cols, 1 + i], legend=legend)
        plot_prediction(pred, mn, 'Prediction Corr to Mean: %.2f' % (crr_pred),
                        [3, cols, cols + 1 + i], legend=legend)
        legend = False
        all_weights = None
        inds = [0, 0]
        for f in plot_params['idx_four']:
            four_weights = weights[f].reshape(plot_params['four_shape'][0])
            ff = reverse_fft(four_weights)
            if all_weights is None:
                nx = int(np.sqrt(len(plot_params['idx_four'])))
                x = plot_params['four_shape'][0][0]
                xx = x * nx
                all_weights = np.zeros([xx, xx])
            all_weights[inds[0]: inds[0] + x,
                        inds[1]: inds[1] + x] = ff
            inds[1] += x
            if inds[1] == xx:
                inds[0] += x
                inds[1] = 0
        plot_four(all_weights, [0, all_weights.max()],
                  [3, cols, 2 * cols + 1 + i],
                      interpolation='bilinear')
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                        top=0.87, wspace=0.3, hspace=0.6)

    #plt.show()
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)




def plot_preds_freq(cellid, results, targets, fname):
    return None

def plot_preds_orient(cellid, results, targets, fname):
    return None

def plot_preds(cellid, results, source, fname):
    fig = plt.figure(figsize=(18, 9))
    fig.set_facecolor('white')
    plt.suptitle('Experiment: %s: %s Source' % (cellid, source))
    cols = len(results)
    legend = True
    for i, [t, dat] in enumerate(results.iteritems()):
        #dat = results[t]
        pred = dat['pred']
        mn = dat['mn']
        std = dat['std']
        crr_pred = dat['crr_pred']
        crr_exp = dat['crr_exp']

        plot_mean_std(None, mn, std, '%s - Corr Trials to Mean: %.2f' %
                            (t, crr_exp), [2, cols, 1 + i], legend=legend)
        plot_prediction(pred, mn, 'Prediction Corr to Mean: %.2f' % (crr_pred),
                        [2, cols, cols + 1 + i], legend=legend)
        legend = False
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                        top=0.87, wspace=0.3, hspace=0.6)

    #plt.show()
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def plot_prediction(pred, actual, title, plt_num=[1, 1, 1], legend=True):
    """ plot the prediction of the model vs the mean of the actual
     experiment """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    #plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
    pred = (pred - pred.mean()) / np.std(pred)
    actual = (actual - actual.mean()) / np.std(actual)
    plt.plot(range(len(pred)), pred, color=clr4, linewidth=2,
             label='Predicted')
    plt.plot(range(len(pred)), actual, color=clr1,
             label='Mean Experimental', linewidth=2)
    if legend:
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)


filt = 0.1
randomisers = [None, 'generated', 'random']
for randomise in randomisers:
    for exp_type in ['FS', 'PYR', 'SOM']:
        fig_path = startup.fig_path + 'ephys/%s/pred/' % (exp_type)
        dat_path = startup.data_path + 'ephys/%s/pred/' % (exp_type)
        if randomise is not None:
            dat_path = dat_path + randomise + '_' + str(filt)
            fig_path = fig_path + randomise + '_' + str(filt)
        else:
            dat_path = dat_path + str(filt)
            fig_path = fig_path + str(filt)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        dat_file = dat_path + '/preds.pkl'
        with open(dat_file, 'rb') as infile:
            cell_results = pickle.load(infile)
        print cell_results.keys()

        dat = {}
        shifts = np.array(cell_results.values()
                          [0].values()[0].values()[0].keys(), dtype=np.int)
        shifts.sort()
        for s in shifts:
            fname = '%s/' % (fig_path)
            if os.path.exists('%ssummary_pred_%s.png' % (fname, s)):
                print '%ssummary_%s.png exists, SKIPPING' % (fname, s)
                continue

            if s not in dat:
                dat[str(s)] = {}

            for cell in cell_results:
                if cell not in dat[str(s)]:
                        dat[str(s)][cell] = {}

                for k in cell_results[cell]:

                    for cmb in cell_results[cell][k]:
                        if cmb not in dat[str(s)][cell]:
                            dat[str(s)][cell][cmb] = {}
                        if k not in dat[str(s)][cell][cmb]:
                            dat[str(s)][cell][cmb][k] = {}
                        val = cell_results[cell][k][cmb][str(s)]
                        dat[str(s)][cell][cmb][k] = val

        for s in dat:
            cmb_dat = {}
            fname = '%s/%s/' % (fig_path, s)
            if not os.path.exists(fname):
                os.makedirs(fname)
            plot_cell_summary(dat[s], fname)
            for cell in dat[s]:
                fname = '%s/%s/%s/' % (fig_path, s, cell)
                if not os.path.exists(fname):
                    os.makedirs(fname)
                for cmb in dat[s][cell]:
                    if cmb not in cmb_dat:
                        cmb_dat[cmb] = {}
                    print fname, cmb
                    if 'Fourier' in cmb:
                        plot_preds_fourier(cell, dat[s][cell][cmb],
                                         fname + cmb)
                    else:
                        plot_preds(cell, dat[s][cell][cmb],
                                   cmb, fname + cmb)

                    for k in dat[s][cell][cmb]:
                        val = [dat[s][cell][cmb][k]['crr_pred']]
                        if k in cmb_dat[cmb]:
                            cmb_dat[cmb][k] += val
                        else:
                            cmb_dat[cmb][k] = val
            fname = '%s/' % (fig_path)
            plot_summary(cmb_dat, fname, s)
