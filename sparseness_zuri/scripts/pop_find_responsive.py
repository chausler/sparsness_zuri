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


def responsive_shuffle_xcorr(cell, centre, whole):
    thresh = 2
    rounds = 50
    samples = len(centre)
    xcorr_c = corr_trial_to_trial(centre.T, 0)
    xcorr_w = corr_trial_to_trial(whole.T, 0)
    shuffles = []
    for i in xrange(rounds):
        shift = np.random.randint(-samples, samples, 1)
        shift_xcorr_c = corr_trial_to_trial(centre.T, shift)
        shift_xcorr_w = corr_trial_to_trial(whole.T, shift)
        shuffles.append([shift_xcorr_c, shift_xcorr_w])
    shuffles = np.array(shuffles)
    [shift_xcorr_c, shift_xcorr_w] = shuffles.mean(0)
    active = (xcorr_c > (shift_xcorr_c * thresh)
              and xcorr_w > (shift_xcorr_w * thresh))
    vals = [cell, active, xcorr_c, xcorr_w, shift_xcorr_c, shift_xcorr_w]
    return vals


def run_over_thresh(dat, thresh, length):
    df = (dat > thresh) * 1.
    cnt = 0
    for n in df:
        if n > 0:
            cnt += 1
            if  cnt >= length:
                return True
        else:
            cnt = 0
    return False


def responsive_baseline_thresh(cell, centre, whole, bs_centre, bs_whole):
    thresh = 3.
    thresh_len = 2
    c_found = []
    for trial in range(centre.shape[1]):
        thrsh = bs_centre[trial, 0] + bs_centre[trial, 1] * thresh
        c_found.append(run_over_thresh(centre[:, trial], thrsh, thresh_len))
    c_found = np.mean(np.array(c_found) * 1.)

    w_found = []
    for trial in range(whole.shape[1]):
        thrsh = bs_whole[trial, 0] + bs_whole[trial, 1] * thresh
        w_found.append(run_over_thresh(whole[:, trial], thrsh, thresh_len))
    w_found = np.mean(np.array(c_found) * 1.)
    active = (c_found >= 0.5 or w_found >= 0.5)
    vals = [cell, active, c_found, w_found]
    #print vals
    return vals



# Sub directory of the figure path to put the plots in
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
        print fname
    except:
        continue
    rf_cells = dat['rf_cells']
    dat_c = dat['dat_c']
    dat_w = dat['dat_w']
    bs_c = dat['bs_c']
    bs_w = dat['bs_w']


    res = []
    for cell in xrange(dat_c.shape[0]):
        res.append(responsive_shuffle_xcorr(cell, dat_c[cell], dat_w[cell]))
#        res.append(responsive_baseline_thresh(cell, dat_c[cell], dat_w[cell],
#                                   bs_c[cell], bs_w[cell]))
    res = np.array(res)
    total = np.sum(res[:, 1])
    act = np.where(res[:, 1])[0]
    cnt = 0
    for r in rf_cells:
        if r in act:
            cnt += 1

    all_res.append([exp_id, total, len(dat_c)])
    print 'Total Active: %d, RF Cells: %d,  Active RF Cells: %d' % (total,
                                                            len(rf_cells), cnt)
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
