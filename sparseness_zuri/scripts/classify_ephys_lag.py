import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')
import pickle
sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
import numpy.fft as fft
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.utils import filter
from data_utils.load_ephys import load_EphysData
from data_utils.movie import load_parsed_movie_dat
#from sklearn.linear_model import LinearRegression as clf
#from sklearn.linear_model import Ridge as clf
from sklearn.linear_model import Lasso as clf
clf_args={'alpha': 0.001}
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from data_utils.utils import do_thresh_corr, corr_trial_to_mean
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
#from sklearn.tree import DecisionTreeRegressor as clr
#from sklearn.linear_model import SGDRegressor as clf
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


def classify(cv):
    train = cv[0]
    # dont use the edge bits for training
    train = train[train > (edges[0] - 1)]
    train = train[train < (len(train) - edges[1])]
    test = cv[1] 
    regr = clf(**clf_args)
    XX = X[:, train].reshape(-1, X.shape[2])
    yy = y[:, train].ravel().copy()
    Xt = X[0, test]

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


def CV(clf, X, y, folds=20, clf_args={}, clf_fit_args={},
       clf_pred_args={}, return_coefs=True, unique_targs=[1], edges=[0, 0]):

    cv = KFold(X.shape[1], folds, indices=True, shuffle=False)
    dview.push({'X': X, 'y': y, 'clf': clf, 'clf_args': clf_args,
                     'fit_args': clf_fit_args, 'pred_args': clf_pred_args,
                     'return_coefs': return_coefs, 'unique_targs': unique_targs,
                     'edges': edges})
    pred = []
    try:
        pred = dview.map(classify, cv)
    except RemoteError as e:
        print e
        if e.engine_info:
            print "e-info: " + str(e.engine_info)
        if e.ename:
            print "e-name:" + str(e.ename)

    preds = []
    coefs = []
    for (p, c) in pred:
        preds += p.tolist()
        coefs += [c]
    dview.results.clear()
    rc.purge_results('all')
    rc.results.clear()
    return np.array(preds), np.array(coefs)


# The bin size in degrees for flow directions
bin_ang = 45
div = 360. / bin_ang
bins = np.arange(-180 - 180 / div, 181 + 180 / div, 360 / div)

# Some Colours for plotting
clr1 = '0.3'
clr2 = '0.9'
clr3 = np.array([255, 151, 115]) / 255.
clr4 = np.array([200, 105, 75]) / 255.


# Bin the flow directions into a number of general directions based on
# size of bin_ang
def bin_flow(flow):
    all_flows = []
    for dim in flow:
        new_flow = []
        for f in dim:
            n, _ = np.histogram(np.degrees(f[0]), bins=bins)
            n[0] += n[-1]
            n = n[:-1]
            new_flow.append(n.tolist() + [f[1]])
        all_flows.append(new_flow)
    new_flow = np.array(new_flow, dtype=np.float)
    if len(new_flow.shape) < 3:
        new_flow = new_flow[np.newaxis, :, :]
    return new_flow


def get_mov_data(comb, targ_type, src_type, e, cellid, exp_type,
                 four_downsample=None, shift=0, randomise=None):

    lum_mask, con_mask, flow_mask, four_mask, four_mask_shape,\
            freq_mask, orient_mask,\
            lum_surr, con_surr, flow_surr, four_surr, four_surr_shape,\
            freq_surr, orient_surr,\
            lum_whole, con_whole, flow_whole, four_whole, four_whole_shape,\
            freq_whole, orient_whole\
                    = load_parsed_movie_dat(cellid, exp_type, four_downsample)

    flow_mask = bin_flow(flow_mask)
    flow_surr = bin_flow(flow_surr)
    flow_whole = bin_flow(flow_whole)

    idx_bar = []
    idx_flow = []
    idx_four = []
    idx_freq = []
    idx_orient = []
    lbl_bar = []
    four_shape = []
    all_dat = None
    if randomise is None:
        if targ_type == 'Center':
            source = e['psth_c_shift'][shift]
        elif targ_type == 'Surround':
            source = e['psth_s_shift'][shift]
        elif targ_type == 'Whole':
            source = e['psth_w_shift'][shift]
    elif randomise == 'random':
        if targ_type == 'Center':
            source = e['psth_c_rand']
        elif targ_type == 'Surround':
            source = e['psth_s_rand']
        elif targ_type == 'Whole':
            source = e['psth_w_rand']
    elif randomise == 'generated':
        if targ_type == 'Center':
            source = e['psth_c_gen']
        elif targ_type == 'Surround':
            source = e['psth_s_gen']
        elif targ_type == 'Whole':
            source = e['psth_w_gen']

    edge = e['edge']

    if src_type == 'Center':
        if 'Luminance' in comb:
            for l in lum_mask:
                all_dat = append_Nones(all_dat,
                                   l[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Luminance']
        if 'Contrast' in comb:
            for c in con_mask:
                all_dat = append_Nones(all_dat,
                                       c[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Contrast']
        if 'Flow' in comb:
            for f in flow_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat,
                                       f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
            for f in flow_mask:
                all_dat = append_Nones(all_dat,
                                       f[:, -1:], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Flow Vel']
        if 'Fourier' in comb:
            for f in four_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                f = np.append(f.real, f.imag, 1)
                all_dat = append_Nones(all_dat, f, 1)
                idx_four += [range(pre_len, all_dat.shape[1])]
                four_shape.append(four_mask_shape)
        if 'Frequency' in comb:
            for f in freq_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_freq += [range(pre_len, all_dat.shape[1])]
        if 'Orientation' in comb:
            for f in orient_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_orient += [range(pre_len, all_dat.shape[1])]
    elif src_type == 'Surround':
        if 'Luminance' in comb:
            for l in lum_surr:
                all_dat = append_Nones(all_dat,
                                       l[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Luminance']
        if 'Contrast' in comb:
            for c in con_surr:
                all_dat = append_Nones(all_dat,
                                       c[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Contrast']
        if 'Flow' in comb:
            for f in flow_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]

            for f in flow_surr:
                all_dat = append_Nones(all_dat, f[:, -1:], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Flow Vel']
        if 'Fourier' in comb:
            for f in four_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_four += [range(pre_len, all_dat.shape[1])]
                four_shape.append(four_surr_shape)
        if 'Frequency' in comb:
            for f in freq_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_freq += [range(pre_len, all_dat.shape[1])]
        if 'Orientation' in comb:
            for f in orient_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_orient += [range(pre_len, all_dat.shape[1])]

    elif src_type == 'Whole':
        if 'Luminance' in comb:
            for l in lum_whole:
                all_dat = append_Nones(all_dat,
                                   l[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Luminance']
        if 'Contrast' in comb:
            for c in con_whole:
                all_dat = append_Nones(all_dat,
                                       c[:, np.newaxis], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Contrast']
        if 'Flow' in comb:
            for f in flow_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat,
                                       f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]

            for f in flow_whole:
                all_dat = append_Nones(all_dat,
                                       f[:, -1:], 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Flow Vel']
        if 'Fourier' in comb:
            for f in four_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_four += [range(pre_len, all_dat.shape[1])]
                four_shape.append(four_whole_shape)
        if 'Frequency' in comb:
            for f in freq_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_freq += [range(pre_len, all_dat.shape[1])]
        if 'Orientation' in comb:
            for f in orient_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f, 1)
                idx_orient += [range(pre_len, all_dat.shape[1])]
    all_dat = np.tile(all_dat, [source.shape[0], 1, 1])
    plot_params = {}
    plot_params['idx_bar'] = idx_bar
    plot_params['idx_flow'] = idx_flow
    plot_params['idx_four'] = idx_four
    plot_params['lbl_bar'] = lbl_bar
    plot_params['four_shape'] = four_shape
    return all_dat, source, plot_params


def append_Nones(target, addition, axis=1):
    """Append to an array if it exists, otherwise make the array equal to
    addition """
    if target == None or len(target) == 0:
        target = addition
    else:
        target = np.append(target, addition, axis)

    return target


def do_lag_classification(exp_type='SOM', combs=['Fourier', 'Frequency', 'Luminance', 'Contrast',
                        'Orientation',  'Flow'],
                      targets=[['Center', 'Center'], ['Whole', 'Whole']],
                      max_comb=None, min_comb=None,
                       four_downsample=None, max_exp=None, sig_thresh=0.,
                        folds=5, filt=0.2,
                       alpha=0.001, randomise=None):
    # Sub directory of the figure path to put the plots in
    dat_path = startup.data_path + 'Sparseness/%s/pred/' % (exp_type)
    mov_path = startup.data_path + 'Sparseness/%s/' % (exp_type)
    if randomise is not None:
        dat_path = dat_path + randomise + '_' + str(filt)
    else:
        dat_path = dat_path + str(filt)
    dat_file = dat_path + '/preds.pkl'
    cell_results = {}
    if os.path.exists(dat_file):
        print ' already exists. loading %s' % dat_file
        with open(dat_file, 'rb') as infile:
            cell_results = pickle.load(infile)
    else:
        print '%s doesnt exist, starting New' % dat_file
        if not os.path.exists(dat_path):
            os.makedirs(dat_path)

    full_targets = []
    for [targ_type, src_type] in targets:
        full_targets.append('%s_%s' % (targ_type, src_type))

    dat = load_EphysData(exp_type, filt=filt)

    if max_comb is None:
        max_comb = len(combs)
    if min_comb is None:
        min_comb = 0

    for num_combs in [1]:
        for comb in itertools.combinations(combs, num_combs):
            full_comb = str(num_combs) + '_' + "_".join(comb)
            for e in dat.values():
                cellid = str(e['cellid'])
                if cellid not in cell_results:
                    cell_results[cellid] = {}
                if randomise is None:
                    shifts = e['shifts']
                else:
                    shifts = [0]
                edge = e['edge']
                if not os.path.exists(mov_path + cellid + '_processed.npz'):
                    print '\nNo movie found ', cellid
                    continue
                else:
                    print '\ndoing ', e['cellid']
                changed = False
                for shift in shifts:
                    for [targ_type, src_type] in targets:
                        k = '%s_%s' % (targ_type, src_type)                        
                        if k not in cell_results[cellid]:
                            cell_results[cellid][k] = {}
                        if full_comb not in cell_results[cellid][k]:
                            cell_results[cellid][k][full_comb] = {}
                        if shift not in cell_results[cellid][k][full_comb]:
                            cell_results[cellid][k][full_comb][shift] = None
                        else:
                            print 'done - continue'
                            continue
                        changed = True
                        edges = [edge, np.abs(np.minimum(-edge, shift))]
                        X, y, plot_params = get_mov_data(comb, targ_type,
                                                         src_type,
                                        e, cellid, exp_type, four_downsample,
                                        shift=shift, randomise=randomise)
                        # ignore edge effects
                        pred, coefs = CV(clf,
                                X, y, folds=folds, clf_args=clf_args,
                                edges=edges)
                        pred, _ = filter(pred, e['bin_freq'])
                        pred = pred[edges[0]: -edges[1]]
                        mn = y[:, edges[0]: -edges[1]].mean(0)
                        std = np.std(y[:, edges[0]: -edges[1]])
                        crr_pred = np.maximum(r2_score(mn, pred), 0)
                        crr_exp = corr_trial_to_mean(y[:, edges[0]: -edges[1]],
                                                     mn)
                        print exp_type, comb, k, shift, crr_pred
                        res = {}
                        res['pred'] = pred
                        res['mn'] = mn
                        res['std'] = std
                        res['crr_pred'] = crr_pred
                        res['crr_exp'] = crr_exp
                        res['coefs'] = coefs
                        res['plot_params'] = plot_params
                        cell_results[cellid][k][full_comb][shift] = res
                if changed:
                    with open(dat_file, 'wb') as outfile:
                        pickle.dump(cell_results, outfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    corrs = []
    exp_type = 'FS'
    # now its mask movie values for all predictions
    # try also whole
    # and make box plots!
    downsample = None
    exp_types = ['FS', 'PYR', 'SOM']
    randomisers = [None, 'generated', 'random']
    for r in randomisers:
        for exp_type in exp_types:
            for filt in [0.1]:
                print 'DOWNSAMPLE %s' % (str(downsample))
                corrs.append(do_lag_classification(exp_type=exp_type, min_comb=None,
                                            max_comb=None,
                                            targets=[['Center', 'Center'],
                                                     ['Whole', 'Center']
                                                     #['Surround', 'Whole']
                                                     ],
                                               folds=10,
                                            #combs=['Fourier'],
                                            #combs=['Luminance', 'Flow'],
                                            max_exp=None,
                                           #targets=['Center', 'CenterWhole', 'Whole', 'WholeWhole'],
                                           four_downsample=downsample,
                                           #randomise='generated',
                                           alpha=0.001,
                                           randomise=r,
                                           filt=filt))
    for c in corrs:
        print c[0]