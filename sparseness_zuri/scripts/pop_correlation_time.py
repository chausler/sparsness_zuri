"""
calculate the correlation between population neurons over time
for both the centre and whole field stimulus paridigms

"""

import sys
sys.path.append('..')
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import pairwise_corr, average_corrs
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os


def cell_corr(dat, win, trial=None):
    corrs = np.zeros([dat.shape[0], dat.shape[0], dat.shape[1]])
    for t in xrange(win, dat.shape[1]):
        if len(dat.shape) > 2:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1,
                                           trial])
        else:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1])
    return corrs


def do_corrs(dat, win):
    corrs = []
    trials = range(dat.shape[2])
    for t in trials:
        print 'trial %d' % t
        val = cell_corr(dat, win, t)
        corrs.append(val)
    mean_corr = cell_corr(dat.mean(2), win)
    return np.array(corrs), mean_corr

d_path = data_path + 'Sparseness/POP/time_corr/'
if not os.path.exists(d_path):
    os.makedirs(d_path)

corr_win = 20
exps = list_PopExps()
for exp in exps:
    fname = '%s%s' % (d_path, exp)
    if os.path.exists(fname + '.npz'):
        print '%s Already exists. Skipping' % (fname)
        continue
    print 'Doing ', exp
    dat = load_PopData(exp)
    active = dat['active']
    d = np.where(active[:, 1])[0]
    dat_c = dat['dat_raw_c']
    dat_w = dat['dat_raw_w']
    trial_corr_c, mean_corr_c = do_corrs(dat_c, corr_win)
    trial_corr_w, mean_corr_w = do_corrs(dat_w, corr_win)
    np.savez_compressed(fname, trial_corr_c=trial_corr_c,
                        mean_corr_c=mean_corr_c, trial_corr_w=trial_corr_w,
                        mean_corr_w=mean_corr_w)

