"""
investigate similarities between results and if compare correlation of
those in the receptive field and those outside of it


"""

import sys
sys.path.append('..')
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import pairwise_corr, do_thresh_corr, average_corrs
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os


def plot_corrs(c_vals, w_vals, n_cells, header, fname):
    fig = plt.figure(figsize=(14, 8))
    fig.set_facecolor('white')
    c_vals = average_corrs(c_vals)
    w_vals = average_corrs(w_vals)
    c_vals_mn = average_corrs(c_vals)
    w_vals_mn = average_corrs(w_vals)

    ax = plt.subplot(311)
    plt.hold(True)
    plt.plot(c_vals.T, '0.8')
    plt.plot(c_vals_mn, '0.4', linewidth=2)
    plt.xlim(0, c_vals.shape[1])
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Masked')
    plt.ylabel('Mean R^2')

    ax = plt.subplot(312)
    plt.hold(True)
    plt.plot(w_vals.T, '0.8')
    plt.plot(w_vals_mn, '0.4', linewidth=2)
    plt.xlim(0, w_vals.shape[1])
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Whole Field')
    plt.ylabel('Mean R^2')

    ax = plt.subplot(313)
    plt.hold(True)
    plt.plot(w_vals_mn, 'g', linewidth=2, label='Whole')
    plt.plot(c_vals_mn, 'k', linewidth=2, label='Centre')
    crr = do_thresh_corr(w_vals_mn, c_vals_mn)
    leg = plt.legend(ncol=2)
    leg.draw_frame(False)
    plt.xlim(0, c_vals.shape[1])
    adjust_spines(ax, ['bottom', 'left'])
    plt.ylabel('Mean R^2')
    plt.xlabel('Sample')
    plt.title('Whole vs Masked: Crr: {0:.2f}'.format(crr))

    plt.suptitle('%s - Intercell R^2 over Trials - #Cells: %d' %
                 (header, n_cells))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9,
                                wspace=0.2, hspace=0.25)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)

    print fname
    print n_cells
    print '\tMean\tMax'
    print 'Centre:\t%.3f\t%.3f' % (c_vals_mn.mean(), c_vals.max())
    print 'Whole:\t%.3f\t%.3f' % (w_vals.mean(), w_vals.max())
    print


d_path = data_path + 'Sparseness/POP/time_corr/'
f_path = fig_path + 'Sparseness/POP/time_corr/'
if not os.path.exists(f_path):
    os.makedirs(f_path)
for f in os.listdir(d_path):
    print f, os.path.join(d_path, f)
files = [f for f in os.listdir(d_path) if
         os.path.isfile(os.path.join(d_path, f))]
for fname in files:
    exp = fname.split('.')[0]
    exp_dat = load_PopData(exp)
    active = np.where(exp_dat['active'][:, 1])[0]
    rf_cells = exp_dat['rf_cells']
    dat = np.load(d_path + fname)
    corr_c = dat["corr_c"]
    corr_w = dat["corr_w"]

    # do all cells
    typ = 'All_Cells'
    print typ
    c_crr = []
    w_crr = []
    print 'try to predict correlation from movie features, try to predict one from another, try to predict movie features from population'
    for i in range(corr_c.shape[1]):
        for j in range(corr_c.shape[1]):
            if i < j:
                c_crr.append(corr_c[:, i, j, :])
                w_crr.append(corr_w[:, i, j, :])
    c_crr = np.array(c_crr)
    w_crr = np.array(w_crr)
    n_cells = corr_c.shape[1]
    fname = '%s%s_%s' % (f_path, exp, typ)
    plot_corrs(c_crr, w_crr, n_cells, "%s %s" % (exp, typ), fname)

    # do active cells
    typ = 'Active_Cells'
    print typ
    c_crr = []
    w_crr = []

    for i in active:
        for j in active:
            if i < j:
                c_crr.append(corr_c[:, i, j, :])
                w_crr.append(corr_w[:, i, j, :])
    c_crr = np.array(c_crr)
    w_crr = np.array(w_crr)
    n_cells = len(active)
    fname = '%s%s_%s' % (f_path, exp, typ)
    plot_corrs(c_crr, w_crr, n_cells, "%s %s" % (exp, typ), fname)

    # do rf cells
    if len(rf_cells) > 0:
        typ = 'RF_Cells'
        print typ
        c_crr = []
        w_crr = []

        for i in rf_cells:
            for j in rf_cells:
                if i < j:
                    c_crr.append(corr_c[:, i, j, :])
                    w_crr.append(corr_w[:, i, j, :])
        c_crr = np.array(c_crr)
        w_crr = np.array(w_crr)
        n_cells = len(rf_cells)
        fname = '%s%s_%s' % (f_path, exp, typ)
        plot_corrs(c_crr, w_crr, n_cells, "%s %s" % (exp, typ), fname)



