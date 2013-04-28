
import sys
sys.path.append('..')
sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import pairwise_corr 
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
from data_utils.movie import load_parsed_movie_dat
from sklearn.linear_model import LinearRegression as clf
clf_args={}
#from sklearn.linear_model import Ridge as clf
#from sklearn.linear_model import Lasso as clf
#clf_args={'alpha': 0.001}
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
from data_utils.utils import do_thresh_corr, normalise_cell

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
    XX = X[:, train].reshape(-1, X.shape[2])
    yy = y[:, train].ravel()
    Xt = X[:, test].reshape(-1, X.shape[2])

    regr.fit(XX, yy)
    coef = None
    if return_coefs:
        try:
            coef = regr.coef_
        except:
            pass
    pred = np.nan_to_num(regr.predict(Xt)).reshape([y.shape[0], len(test)])
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


filter = ['120425']

folds = 10
exps = list_PopExps()
for exp in exps:
    if len(filter) > 0 and exp not in filter:
        print exp, ' not in filter, skipping'
        continue
    print 'Doing ', exp
    dat = load_PopData(exp)
    dat_c = dat['dat_c']
    #dat_c = normalise_cell(dat_c)
    dat_c = np.swapaxes(dat_c, 0, 2)
    dat_w = dat['dat_w']
    #dat_w = normalise_cell(dat_w)
    dat_w = np.swapaxes(dat_w, 0, 2)
    active = dat['active']
    d = np.where(active[:, 1])[0]
    lum_mask, con_mask, flow_mask, four_mask, four_mask_shape,\
            freq_mask, orient_mask,\
            lum_surr, con_surr, flow_surr, four_surr, four_surr_shape,\
            freq_surr, orient_surr,\
            lum_whole, con_whole, flow_whole, four_whole, four_whole_shape,\
            freq_whole, orient_whole = load_parsed_movie_dat(exp, 'POP', 7)
    all_dat = {'Data': {'Centre': dat_c, 'Whole': dat_w}, 'Movie': {}}
    #all_dat['Movie']['Contrast'] = con_mask
    #all_dat['Movie']['Luminence'] = lum_mask
    #all_dat['Movie']['Fourier'] = four_mask
    all_dat['Movie']['Frequency'] = freq_mask
    #all_dat['Movie']['Orientation'] = orient_mask

    for d in all_dat['Data']:
        print d
        X = all_dat['Data'][d]
#        for cell in xrange(XX.shape[2]):
#            print 'Cell %d' % cell
        #X = XX#[:, :, cell: cell + 1]
        for m in all_dat['Movie']:
            print m
            ys = all_dat['Movie'][m]
            if len(ys.shape) < 3:
                ys = ys[:, :, np.newaxis]
            #print ys.shape
            ys = (ys - ys.mean()) / np.std(ys)
            #print ys.shape
            for dim in xrange(1, ys.shape[2]):
                y = ys[:, :, dim]
                samples = np.minimum(y.shape[1], X.shape[1])
                y = y[:, :samples]
                X = X[:, :samples]
                y = np.tile(y, [X.shape[0], 1])
                pred_trial = np.zeros([X.shape[0], y.shape[1]])
                trls = np.ar ange(X.shape[0])
                for trl in trls:
                    regr = clf(**clf_args)
                    XX = X[trls != trl].reshape(-1, X.shape[2])
                    yy = y[:-1].ravel()
                    Xt = X[trl].reshape(-1, X.shape[2])
                    regr.fit(XX, yy)
                    pred = np.nan_to_num(regr.predict(Xt))
#                    print do_thresh_corr(pred, y[0].ravel())
                    pred_trial[trl, :] = pred

##                pred_time, coefs = CV_time(clf, X, y, folds=folds, clf_args=clf_args,
##                                 edges=[0, 0])
#                pred_trial, coefs = CV_trial(clf, X, y, folds=X.shape[0], clf_args=clf_args,
#                                 edges=[0, 0])
##                for p in pred_time:
##                    crr = do_thresh_corr(p, con_mask[0])
##                    if crr > 0:
##                        res= 'Predict TIME: %s, Using: %s, Dimension: %d, Crr: %.2f' %(
##                                                       d, m, dim, crr)
##                        print res

                for i, p in enumerate(pred_trial):
                    crr = do_thresh_corr(p, y[0])
                    res= '%d: %s, Using: %s, Dimension: %d, Crr: %.2f' %(
                                                       i, d, m, dim, crr)
#                    if crr > 0:
                    print res
                crr = do_thresh_corr(pred_trial.ravel(), y.ravel())#, corr_type='pearsonr')
                res= 'Predict TRIAL: %s, Using: %s, Dimension: %d, Crr: %.2f' %(
                                                   d, m, dim, crr)
                
                if crr > 0:
                    print res
                    plt.figure(figsize=(14, 8))
                    plt.hold(True)
                    plt.plot(pred_trial.ravel(), label='pred')
                    plt.plot(y.ravel(), label='movie')
                    plt.legend()
                    plt.title(res)
                    plt.show()
#    print pred.shape, con_mask.shape

