import sys
sys.path.append('..')
from startup import *
import numpy as np
import scipy.stats
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
from data_utils.utils import do_thresh_corr, corr_trial_to_trial



# Sub directory of the figure path to put the plots in
thresh = 3
rounds = 100
d_path = data_path + 'Sparseness/POP/'

exps = list_PopExps()

all_res = []
for exp_id in exps:
    fname = '%s%s%s' % (d_path, 'xcorr_active_', exp_id)
    if os.path.exists(fname):
        print fname + ' exist, SKIPPING'
        #continue
    try:
        dat = load_PopData(exp_id)
    except:
        continue
    dat_c = dat['dat_c']
    dat_w = dat['dat_w']

    print dat_c.shape, dat_w.shape

    res = []
    for cell in xrange(dat_c.shape[0]):
        c_mn = dat_c[cell].mean(1)
        w_mn = dat_w[cell].mean(1)
        samples = len(c_mn)
        xcorr_c = corr_trial_to_trial(dat_c[cell].T, 0)
        xcorr_w = corr_trial_to_trial(dat_w[cell].T, 0)
        shuffles = []
        for i in xrange(rounds):
            shift = np.random.randint(-samples, samples, 1)
            shift_xcorr_c = corr_trial_to_trial(dat_c[cell].T, shift)
            shift_xcorr_w = corr_trial_to_trial(dat_w[cell].T, shift)
            shuffles.append([shift_xcorr_c, shift_xcorr_w])
        shuffles = np.array(shuffles)
        [shift_xcorr_c, shift_xcorr_w] = shuffles.mean(0)
        active = (xcorr_c > (shift_xcorr_c * thresh)
                  and xcorr_w > (shift_xcorr_w * thresh))
        vals = [cell, active, xcorr_c, xcorr_w, shift_xcorr_c, shift_xcorr_w]
        res.append(vals)
        print vals
    res = np.array(res)
    total = np.sum(res[:, 1])
    all_res.append([exp_id, total, len(dat_c)])
    print 'Total Active: %d' % total
    np.save(fname, res)

for r in all_res:
    print r
#    fig = plt.figure()
#    plt.subplot(121, aspect='equal')
#    plt.scatter(res[:, 1], res[:, 3])
#    plt.xlabel('Trial to Mean Xcorr')
#    plt.ylabel('Shuffled Xcorr')
#    plt.subplot(122, aspect='equal')
#    plt.scatter(res[:, 2], res[:, 4])
#    plt.xlabel('Trial to Mean Xcorr')
#    plt.show()
