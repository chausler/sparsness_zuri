
"""
to do, 2 separate plots. compare correlation of prediction to inter trial correlation

"""


import sys
sys.path.append('..')
sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
from sklearn.decomposition import PCA
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import corr_trial_to_trial, do_thresh_corr
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
import cPickle
from data_utils.tsne import calc_tsne
from sklearn.linear_model import LinearRegression as clf
clf_args = {}

target = data_path + 'Sparseness/POP/predict/'
if not os.path.exists(target):
    os.makedirs(target)
pca_dims = 30
out_dims = 2


exps = list_PopExps()
for exp in exps:
    out_fname = exp + '.pkl'
    dat = load_PopData(exp)
    active = dat['active']
    d = np.where(active[:, 1])[0]
    print d
    res = []
    cell_idx = np.arange(dat['dat_c'].shape[0])
    crr_c = None
    crr_w = None
    for cell in cell_idx:
        if cell in d:
            print 'active'
        else:
            continue
        print cell
        cell_res = []
        for i, src in enumerate(['dat_c', 'dat_w']):
            dt = dat[src]
            if i == 0 and crr_c is None:
                try:
                    crr_c = np.load('%s%s_%s.npy' % (target, exp, src))
                except:
                    crr_c = []
                    for c in cell_idx:
                        crr_c.append(corr_trial_to_trial(dt[c].T))
                    crr_c = np.array(crr_c)
                    np.save('%s%s_%s' % (target, exp, src), crr_c)
            elif i == 1 and crr_w is None:
                try:
                    crr_w = np.load('%s%s_%s.npy' % (target, exp, src))
                except:
                    crr_w = []
                    for c in cell_idx:
                        crr_w.append(corr_trial_to_trial(dt[c].T))
                    crr_w = np.array(crr_w)
                    np.save('%s%s_%s' % (target, exp, src), crr_w)
            src_res = []
            trials = np.arange(dt.shape[2])
            ys = []
            preds = []
            for t in trials:
                X = dt[cell_idx != cell]
                X = X[:, :, trials != t].reshape(X.shape[0], -1).T
                y = dt[cell, :, trials != t].ravel()
                XX = dt[cell_idx != cell, :, t].reshape(X.shape[1], -1).T
                yy = dt[cell, :, t].ravel()
                c = clf(**clf_args)
                c.fit(X, y)
                ys.append(yy)
                preds.append(c.predict(XX))
            crr = do_thresh_corr(np.array(ys).ravel(), np.array(preds).ravel())
            cell_res.append(crr)
        res.append(cell_res)
    res = np.array(res)
    print res.shape
    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.scatter(res[:, 0], res[:, 1])
    plt.xlabel('Pred Corr Centre')
    plt.ylabel('Pred Corr Whole')
    plt.subplot(132)
    plt.scatter(res[:, 0], crr_c[d])
    plt.xlabel('Pred Corr Centre')
    plt.ylabel('Trial Corr Centre')
    plt.subplot(133)
    plt.scatter(res[:, 1], crr_w[d])
    plt.xlabel('Pred Corr Whole')
    plt.ylabel('Trial Corr Whole')
    plt.subplots_adjust(left=0.07, bottom=0.05, right=0.95, top=0.9,
                                wspace=0.2, hspace=0.35)
    plt.show()
        