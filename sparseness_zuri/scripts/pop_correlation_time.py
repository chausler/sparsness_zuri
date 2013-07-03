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


def trial_corr(trial, dat, win):
    corrs = np.zeros([dat.shape[0], dat.shape[0], dat.shape[1] - win])
    for t in xrange(dat.shape[1] - win):
        corrs[:, :, t] = pairwise_corr(dat[:, t: t + win, trial])
    return corrs


def trial_corr_parallel(trial):
    corrs = np.zeros([dat.shape[0], dat.shape[0], dat.shape[1] - win])
    for t in xrange(dat.shape[1] - win):
        corrs[:, :, t] = pairwise_corr(dat[:, t: t + win])
    return corrs


def do_corrs(dat, win):
    corrs = []
    trials = range(dat.shape[2])
    for t in trials:
        print 'trial %d' % t
        val = trial_corr(t, dat, win)
        corrs.append(val)
    return np.array(corrs)

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
    dat_c = dat['dat_c']
    dat_w = dat['dat_w']
    corr_c = do_corrs(dat_c, corr_win)
    corr_w = do_corrs(dat_w, corr_win)
    np.savez_compressed(fname, corr_c=corr_c, corr_w=corr_w)

