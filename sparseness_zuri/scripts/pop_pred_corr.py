
import sys
sys.path.append('..')
sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import pairwise_corr , average_corrs
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os

from data_utils.movie import load_parsed_movie_dat
#from sklearn.linear_model import LinearRegression as clf
clf_args={}
#from sklearn.linear_model import Ridge as clf
from sklearn.linear_model import Lasso as clf
clf_args={'alpha': 0.001}
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
from data_utils.utils import do_thresh_corr, normalise_cell
from sklearn.preprocessing import normalize

import itertools
import os
import sklearn
print sklearn.__version__
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
print '%d engines found' % len(rc.ids)


def classify_time(cv):
    train = cv[0]
    # dont use the edge bits for training
    train = train[train > (edges[0] - 1)]
    train = train[train < (len(train) - edges[1])]
    test = cv[1] 
    regr = clf(**clf_args)
    XX = X[:, train].T
    yy = y[:, train].ravel()
    Xt = X[:, test].T

    regr.fit(XX, yy)
    coef = None
    if return_coefs:
        try:
            coef = regr.coef_
        except:
            pass
    pred = np.nan_to_num(regr.predict(Xt))
    return (pred, coef)


def classify_trial(cv):
    train = cv[0]
    test = cv[1]
    regr = clf(**clf_args)
    XX = X[train].reshape(-1, X.shape[2])
    yy = y[train].ravel()
    Xt = X[test].reshape(-1, X.shape[2])

    regr.fit(XX, yy)
    coef = None
    if return_coefs:
        try:
            coef = regr.coef_
        except:
            pass
    pred = np.nan_to_num(regr.predict(Xt))
    return (pred, coef)
    #return (pred, regr.alpha_)
        #return (train, test)


def CV_time(clf, X, y, folds=20, clf_args={}, clf_fit_args={},
       clf_pred_args={}, return_coefs=True, unique_targs=[1], edges=[0, 0]):

    cv = KFold(X.shape[1], folds, indices=True, shuffle=False)
    dview.push({'X': X, 'y': y, 'clf': clf, 'clf_args': clf_args,
                     'fit_args': clf_fit_args, 'pred_args': clf_pred_args,
                     'return_coefs': return_coefs, 'unique_targs': unique_targs,
                     'edges': edges})
    pred = []
    try:
        pred = dview.map(classify_time, cv)
    except RemoteError as e:
        print e
        if e.engine_info:
            print "e-info: " + str(e.engine_info)
        if e.ename:
            print "e-name:" + str(e.ename)

    preds = None
    coefs = []
    for (p, c) in pred:
        if preds is None:
            preds = p
        else:
            preds = np.append(preds, p, 1)
        coefs += [c]
    dview.results.clear()
    rc.purge_results('all')
    rc.results.clear()
    return np.array(preds), np.array(coefs)


def CV_trial(clf, X, y, folds=20, clf_args={}, clf_fit_args={},
       clf_pred_args={}, return_coefs=True, unique_targs=[1], edges=[0, 0]):

    cv = KFold(X.shape[0], folds, indices=True, shuffle=False)
    dview.push({'X': X, 'y': y, 'clf': clf, 'clf_args': clf_args,
                     'fit_args': clf_fit_args, 'pred_args': clf_pred_args,
                     'return_coefs': return_coefs, 'unique_targs': unique_targs,
                     'edges': edges})
    pred = []
    try:
        pred = dview.map(classify_trial, cv)
    except RemoteError as e:
        print e
        if e.engine_info:
            print "e-info: " + str(e.engine_info)
        if e.ename:
            print "e-name:" + str(e.ename)

    preds = []
    coefs = []
    for (p, c) in pred:
        preds.append(p)
        coefs += [c]
    dview.results.clear()
    rc.purge_results('all')
    rc.results.clear()
    return np.array(preds), np.array(coefs)


filter = [] #'120425']

folds = 5
exps = list_PopExps()
d_path = data_path + 'Sparseness/POP/time_corr/'
for alpha in [0.001, 0.002, 0.005, 0.01, 0.15, 0.5, 1.]:
    clf_args={'alpha': alpha}
    crrs = []
    for exp in exps:
        if len(filter) > 0 and exp not in filter:
            print exp, ' not in filter, skipping'
            continue
#        print 'Doing ', exp

        exp_dat = load_PopData(exp)
        active = np.where(exp_dat['active'][:, 1])[0]
        rf_cells = exp_dat['rf_cells']
        try:
            dat = np.load(d_path + exp + '.npz')
        except:
            print 'file not found ', exp
            continue
        mean_corr_c = dat["mean_corr_c"]
        mean_corr_w = dat["mean_corr_w"]
        win = dat['win']

        c_crr = []
        w_crr = []
        mn_c_crr = []
        mn_w_crr = []
        print 'try to predict correlation from movie features, try to predict one from another, try to predict movie features from population'
        for i in range(mean_corr_c.shape[1]):
            for j in range(mean_corr_c.shape[1]):
                if i < j:
                    mn_c_crr.append(mean_corr_c[i, j, win / 2: -(win / 2)])
                    mn_w_crr.append(mean_corr_w[i, j, win / 2: -(win / 2)])
        dat_c = average_corrs(np.array(mn_c_crr))
        dat_w = average_corrs(np.array(mn_w_crr))

    
        lum_mask, con_mask, flow_mask, four_mask, four_mask_shape,\
                freq_mask, orient_mask,\
                lum_surr, con_surr, flow_surr, four_surr, four_surr_shape,\
                freq_surr, orient_surr,\
                lum_whole, con_whole, flow_whole, four_whole, four_whole_shape,\
                freq_whole, orient_whole = load_parsed_movie_dat(exp, 'POP', None)
        all_dat = {'Data': {'Centre': dat_c, 'Whole': dat_w , 'Diff': dat_c - dat_w}, 'Movie': {}}
        all_dat['Movie']['Contrast'] = con_mask
        all_dat['Movie']['Luminence'] = lum_mask
        all_dat['Movie']['Fourier'] = np.append(four_mask.real, four_mask.imag,
                                                axis=2).astype(np.float)
        all_dat['Movie']['Frequency'] = freq_mask
        all_dat['Movie']['Orientation'] = orient_mask
    
        for d in all_dat['Data']:
#            print d
            y = all_dat['Data'][d]
            print d
            for m in all_dat['Movie']:
#                print m
                xs = all_dat['Movie'][m]
                if len(xs.shape) < 3:
                    xs = xs[:, :, np.newaxis]
                #print ys.shape
                xs = (xs - xs.mean()) / np.std(xs)
                #print ys.shape
                for dim in xrange(1, xs.shape[2]):
                    X = xs[:, :, dim]
                    samples = np.minimum(y.shape[0], X.shape[1])
                    yy = y[win / 2: samples]
                    X = X[:, win / 2: samples]

                    if np.diff(yy).sum() == 0:
                        print 'no activity, skipping'
                        continue
                    #y = np.tile(y, [X.shape[0], 1])
                    trls = np.arange(X.shape[0])
                    pred_time, coefs = CV_time(clf, X, yy, folds=folds,
                                               clf_args=clf_args,
                                     edges=[0, 0])

                    pred_time = pred_time.ravel()
                    yy = yy.ravel()
                    crr = do_thresh_corr(pred_time, yy ,corr_type=None)#, corr_type='pearsonr')
                    crrs.append(crr)
                    res= 'Predict: %s, Using: %s, Dimension: %d, Crr: %.2f' %(
                                                       d, m, dim, crr)
                    pred_time = (pred_time - pred_time.mean()) / np.std(pred_time)
                    yy = (yy - yy.mean()) / np.std(yy)
                    print res
                    if crr > 0.01:
                        plt.figure(figsize=(14, 8))
                        plt.hold(True)
                        plt.plot(pred_time, label='pred')
                        plt.plot(yy, label='actual corr')
                        plt.legend()
                        plt.title(res)
                        plt.show()
    crrs = np.array(crrs)
    av_crrs = average_corrs(crrs)
    print alpha, av_crrs
    plt.hist(crrs, bins=30)
    plt.title('%.3f %.3f' % (alpha, av_crrs))
    plt.show()
    
#    print pred.shape, con_mask.shape

