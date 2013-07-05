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

try:
    from IPython.parallel import Client
    from IPython.parallel.error import RemoteError
except:
    Client = None
    print 'Failed to find IPython.parallel - No parallel processing available'


print 'hooraz' if Client else 'blag'
rc = Client()
dview = rc[:]
dview.execute('import numpy as np')
dview.execute('from data_utils.utils import pairwise_corr')
print '%d engines found' % len(rc.ids)


def cell_corr_parallel(trial):
    corrs = np.zeros([dat.shape[0], dat.shape[0], dat.shape[1]])
    for t in xrange(win / 2, dat.shape[1] - (win / 2)):
        if len(dat.shape) > 2:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1,
                                           trial], do_abs=False)
        else:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1],
                                           do_abs=False)
    return corrs


def cell_corr(dat, win, trial=None):
    corrs = np.zeros([dat.shape[0], dat.shape[0], dat.shape[1]])
    for t in xrange(win / 2, dat.shape[1] - (win / 2)):
        if len(dat.shape) > 2:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1,
                                           trial], do_abs=False)
        else:
            corrs[:, :, t] = pairwise_corr(dat[:, t - win / 2: t + win / 2 + 1],
                                           do_abs=False)
    return corrs


def do_corrs(dat, win):
    dview.push({'dat': dat, 'win': win})
    try:
        crrs = dview.map(cell_corr_parallel, range(dat.shape[2]))
    except RemoteError as e:
        print e
        if e.engine_info:
            print "e-info: " + str(e.engine_info)
        if e.ename:
            print "e-name:" + str(e.ename)

    corrs = []
    for c in crrs:
        corrs.append(c)

    dview.results.clear()
    rc.purge_results('all')
    rc.results.clear()

#    corrs = []
#    trials = range(dat.shape[2])
#    for t in trials:
#        print 'trial %d' % t
#        val = cell_corr(dat, win, t)
#        corrs.append(val)
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
                        mean_corr_w=mean_corr_w, win=corr_win)

