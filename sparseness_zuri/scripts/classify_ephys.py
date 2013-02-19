import matplotlib
# force plots to file. no display. comment out to use plt.show()
matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')

sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
import numpy.fft as fft
import startup
from collections import deque
from scipy.stats import pearsonr, spearmanr, kendalltau
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.utils import filter
from data_utils.load_ephys import load_EphysData, load_parsed_movie_dat
#from sklearn.linear_model import LinearRegression as clf
#from sklearn.linear_model import Ridge as clf
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from data_utils.utils import do_thresh_corr, corr_trial_to_mean
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
    pred = regr.predict(Xt)
    return (pred, coef)
        #return (train, test)


def CV(clf, X, y, folds=20, clf_args={}, clf_fit_args={},
       clf_pred_args={}, return_coefs=True, unique_targs=[1]):

    cv = KFold(X.shape[1], folds, indices=True, shuffle=False)
    dview.push({'X': X, 'y': y, 'clf': clf, 'clf_args': clf_args,
                     'fit_args': clf_fit_args, 'pred_args': clf_pred_args,
                     'return_coefs': return_coefs, 'unique_targs': unique_targs})
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


def arctan_norm(data):
    mx = np.abs(data).max()
    data /= mx
    data = np.arctan(data)
    return data


def reverse_fft(four_weights):
        A = fft.ifft2(four_weights)
        return np.sqrt(A.real ** 2 + A.imag ** 2)


def plot_mean_std(mn, std, title, plt_num=[1, 1, 1], legend=True):
    """plot the mean +/- std as timeline """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor=clr2)
    plt.plot(range(len(std)), mn, color=clr1,
             label='Trial Mean', linewidth=2)
    if legend:
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False,
                   ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)


def plot_weight_dist(weights, plt_num=[1, 1, 1]):
    """ plot a histogram of the learnt weight distribution """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.hist(weights, 40)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Weight Distribution')


def plot_bar_weights(weights, labels, ylims, plt_num=[1, 1, 1]):
    """ plot some weights (luminance, contrast, flow strength)
    on a bar chart """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.bar(range(len(weights)), weights)
    ax.set_xticks(np.arange(len(weights)) + 0.5)
    ax.set_xticklabels(labels)
    plt.ylim(ylims)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Other Weights')


def plot_prediction(pred, actual, title, plt_num=[1, 1, 1], legend=True):
    """ plot the prediction of the model vs the mean of the actual
     experiment """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    #plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
    pred = (pred - pred.mean()) / np.std(pred)
    actual = (actual - actual.mean()) / np.std(actual)
    plt.plot(range(len(pred)), pred, color=clr4, linewidth=2,
             label='Predicted')
    plt.plot(range(len(pred)), actual, color=clr1,
             label='Mean Experimental', linewidth=2)
    if legend:
        plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)


def plot_flow(weights, ylims, plt_num=[1, 1, 1]):
    """plot the weights for flow direction as a polar bar graph """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2], polar=True)
    plt.hold(True)
    plt_bins = bins[:-2]
    theta = plt_bins / 360. * 2 * np.pi
    width = np.ones(len(theta)) * bin_ang / 360. * 2 * np.pi
    for x, y, w in zip(theta, weights, width):
        if y < 0.:
            col = 'r'
        else:
            col = 'b'
        plt.bar(x, y, width=w, color=col)
    plt.ylim(ylims)
    plt.title('Flow Direction Weights')


def plot_four(four_weights, ylims, plt_num=[1, 1, 1], title=None,
              ax_off=False, interpolation='None'):
    """plot the weights of the spatial fourier """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    ext = four_weights.shape[0] / 2.
    _ = plt.imshow(four_weights.T, cmap=plt.cm.gray,
                    interpolation=interpolation,
               extent=(-ext, ext, -ext, ext))
    plt.clim(ylims)
    if ax_off:
        adjust_spines(ax, [])
    else:
        adjust_spines(ax, ['bottom', 'left'])        
    if title is None:
        title = 'Filter'
    plt.title(title)
#    cax = plt.subplot(244)
#    plt.colorbar(im, cax=cax, cmap=plt.cm.gray)
#    plt.title('Colour Scale')


def plot_cell_summary(res, fig_path, extras=''):
    colors = ['0.5', '0.05', 'k', 'r', 'g', 'y', 'c', 'm']
    summary = []
    for cnum, cell in enumerate(sorted(res.keys())):
        vals = []
        maxes = {}
        x = 0
        lbls = []
        xtcks = []
        fig = plt.figure(figsize=(14, 12))
        ax = plt.subplot(111)
        plt.hold(True)
        for comb in sorted(res[cell].keys()):
            x += 1
            plt.text(x, 0.95, comb, rotation=60, verticalalignment='bottom')
            for i, targ in enumerate(sorted(res[cell][comb].keys())):
                for src in sorted(res[cell][comb][targ].keys()):
                    val = res[cell][comb][targ][src]
                    targ_src = '%s_%s' % (targ, src)
                    lbls.append(targ_src)
                    if targ_src in maxes:
                        if val > maxes[targ_src][1]:
                            maxes[targ_src] = [comb, val]
                    else:
                        maxes[targ_src] = [comb, val]
                    xtcks.append(x)
                    plt.plot([x, x], [0, val], ':', color='0.3')
                    plt.scatter(x, val, c=colors[i], s=200)
                    x += 1
                    vals.append([comb, '%s_%s' % (targ, src), '%.2f' % val])
            x += 2
        plt.plot([0, x], [0, 0], '--')
        plt.xlim(0, x)
        plt.ylim(-1, 1)
        plt.xticks(xtcks, lbls, rotation='vertical')
        adjust_spines(ax, ['bottom', 'left'])
        plt.ylabel('Prediction Correlation to Experimental Mean')
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.7,
                       wspace=0.3, hspace=0.34)
        fig.savefig(fig_path + 'summary_' + cell + '.eps')
        fig.savefig(fig_path + 'summary_' + cell + '.png')
        plt.close(fig)

        f = open(fig_path + 'summary_' + cell + '.csv', 'w')
        for v in vals:
            f.write("\t".join(v))
            f.write('\n')
        f.close()
        line2 = '%s\t' % cell
        line1 = '\t'
        for mm in sorted(maxes.keys()):
            line1 += '%s\t\t' % mm
            line2 += '%.2f\t%s\t' % (maxes[mm][1], maxes[mm][0])
        if cnum == 0:
            summary.append(line1 + '\n')
        summary.append(line2 + '\n')

    f = open(fig_path + 'summary_all.csv', 'w')
    for l in summary:
        f.write(l)
    f.close()

 
 #       plt.show()
 



def plot_summary(comb_corrs, targets, fig_path, extras=''):
    """ plot a summary showing the mean and std for all attempted
    combinations"""
    fig = plt.figure(figsize=(14, 12))
    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    _ = plt.subplot(111)
    xaxis = []
    boxes = []
    lbls = []

    #targets.append('Overall')
    for i, [comb, vals] in enumerate(comb_corrs):
        xaxis.append(comb)
        offset = 0
        for j, targ in enumerate(targets):
            dat = vals[targ]
            if len(dat) == 0:
                continue
            if targ == 'Overall':
                col = '0.3'
            else:
                col = colors[j]
            #do_box_plot(np.array(dat), np.array([i + offset]), col)
            do_spot_scatter_plot(np.array(dat), np.array([i + offset]), col)
            offset += 0.3
            if i == 0:
                boxes.append(plt.Rectangle((0, 0), 1, 1, fc=col))
                lbls.append(targ)
                if j == 0:
                    plt.title('N=%d' % len(dat))

    plt.xticks(range(len(xaxis)), xaxis, rotation='vertical')
    plt.xlim(-1, len(xaxis) + 1)
    plt.ylim(-1, 1)
    plt.subplots_adjust(left=0.05, bottom=0.5, right=0.97, top=0.95,
                       wspace=0.3, hspace=0.34)
    plt.legend(boxes, lbls, frameon=False, loc=4)
    plt.ylabel('Correlation between prediction and Experimental Mean')
    fname = '%s%s_pred_%s' % (fig_path, 'summary', extras)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)


def plot_preds_fourier(cellid, results, targets, fname):
    fig = plt.figure(figsize=(18, 9))
    fig.set_facecolor('white')
    plt.suptitle('Experiment: %s  Fourier Source' % (cellid))
    cols = len(results)
    legend = True
    for i, t in enumerate(targets):
        dat = results[t]
        pred = dat[0]
        mn = dat[1]
        std = dat[2]
        crr_pred = dat[3]
        crr_y = dat[4]
        weights = dat[5]
        plot_params = dat[6]

        plot_mean_std(mn, std, '%s - Corr Trials to Mean: %.2f' %
                            (t, crr_y), [3, cols, 1 + i], legend=legend)
        plot_prediction(pred, mn, 'Prediction Corr to Mean: %.2f' % (crr_pred),
                        [3, cols, cols + 1 + i], legend=legend)
        legend = False
        all_weights = None
        inds = [0, 0]
        for f in plot_params['idx_four']:
            four_weights = weights[f].reshape(plot_params['four_shape'][0])
            ff = reverse_fft(four_weights)
            if all_weights is None:
                nx = int(np.sqrt(len(plot_params['idx_four'])))
                x = plot_params['four_shape'][0][0]
                xx = x * nx
                all_weights = np.zeros([xx, xx])
            all_weights[inds[0]: inds[0] + x,
                        inds[1]: inds[1] + x] = ff
            inds[1] += x
            if inds[1] == xx:
                inds[0] += x
                inds[1] = 0
        plot_four(all_weights, [0, all_weights.max()],
                  [3, cols, 2 * cols + 1 + i],
                      interpolation='bilinear')
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                        top=0.87, wspace=0.3, hspace=0.6)

    #plt.show()
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def plot_preds(cellid, results, targets, source, fname):
    fig = plt.figure(figsize=(18, 9))
    fig.set_facecolor('white')
    plt.suptitle('Experiment: %s: %s Source' % (cellid, source))
    cols = len(results)
    legend = True
    for i, t in enumerate(targets):
        dat = results[t]
        pred = dat[0]
        mn = dat[1]
        std = dat[2]
        crr_pred = dat[3]
        crr_y = dat[4]

        plot_mean_std(mn, std, '%s - Corr Trials to Mean: %.2f' %
                            (t, crr_y), [2, cols, 1 + i], legend=legend)
        plot_prediction(pred, mn, 'Prediction Corr to Mean: %.2f' % (crr_pred),
                        [2, cols, cols + 1 + i], legend=legend)
        legend = False
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                        top=0.87, wspace=0.3, hspace=0.6)

    #plt.show()
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def plot_clf_compare(clf_vals, fname):
    fig = plt.figure(figsize=(14, 9))
    fig.set_facecolor('white')
    rows = len(clf_vals)
    cols = 10

    for i, c in enumerate(clf_vals):
        targ_type = c[0]
        weights = c[1]
        #weights = arctan_norm(weights)
        lm = np.maximum(np.abs(weights.min()),
                        weights.max())
        ylims = [-lm, lm]
        plot_params = c[2]
        offset = 1
        if targ_type == 'Surround':
            offset = 2
        for jj, f in enumerate(plot_params['idx_four']):
            title = ''
            if len(plot_params['idx_four']) > 0:
                if jj == 0:
                    title = targ_type #'Mask Fourier Weights'
                    ax_off=False
                else:
                    ax_off = True
#                else:
#                    title = 'Surround Fourier Weights'
                four_weights = weights[f].reshape(
                                    plot_params['four_shape'][0])
                four_weights = reverse_fft(four_weights)
                ylims = [four_weights.min(), four_weights.max()]
                plot_four(four_weights, ylims,
                          [rows, cols, (i * cols) + jj + offset], title,
                          ax_off, interpolation='bilinear')
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98,
            top=0.98, wspace=0.1, hspace=0.05)
    fname = '%s%s_pred' % (fname, 'compare_four')
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)


def gen_data(dat, freq):
    idx = np.arange(dat.shape[0])
    np.random.shuffle(idx)
    for d in range(dat.shape[1]):
        dat[:, d], _ = filter(dat[idx, d], freq)
    return dat


def lassoCV(X, y):
    import time
    t1 = time.time()
    model = LassoCV(cv=5).fit(X, y)
    t_lasso_cv = time.time() - t1

    # Display results
    m_log_alphas = model.alphas_

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
               label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(lambda)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
             '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.show()


def get_mov_data(comb, targ_type, src_type, e, expdate, exp_type,
                 four_downsample=None, randomise=None, shift=0):

    lum_mask, con_mask, flow_mask,\
                    four_mask, four_mask_shape,\
                    lum_surr, con_surr, flow_surr,\
                    four_surr, four_surr_shape,\
                    lum_whole, con_whole, flow_whole,\
                    four_whole, four_whole_shape \
                    = load_parsed_movie_dat(expdate, exp_type, four_downsample)

    flow_mask = bin_flow(flow_mask)
    flow_surr = bin_flow(flow_surr)
    flow_whole = bin_flow(flow_whole)

    idx_bar = []
    idx_flow = []
    idx_four = []
    lbl_bar = []
    four_shape = []
    all_dat = None
    if randomise is None:
        if targ_type == 'Center':
            source = e['psth_c']
        elif targ_type == 'Surround':
            source = e['psth_s']
        elif targ_type == 'Whole':
            source = e['psth_w']
    elif randomise == 'random':
        if targ_type == 'Center':
            source = e['psth_c_rand']
        elif targ_type == 'Surround':
            source = e['psth_s_rand']
        elif targ_type == 'Whole':
            source = e['psth_w_rand']
    elif randomise == 'generate':
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
                all_dat = append_Nones(all_dat, f, 1)
                idx_four += [range(pre_len, all_dat.shape[1])]
                four_shape.append(four_mask_shape)
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
    elif src_type == 'Generated':
        if 'Luminance' in comb:
            for l in lum_whole:
                r = gen_data(l[:, np.newaxis], e['bin_freq'])
                all_dat = append_Nones(all_dat, r, 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Luminance']
        if 'Contrast' in comb:
            for c in con_whole:
                r = gen_data(c[:, np.newaxis], e['bin_freq'])
                all_dat = append_Nones(all_dat, r, 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Contrast']
        if 'Flow' in comb:
            for f in flow_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                r = gen_data(f[:, :-1], e['bin_freq'])
                all_dat = append_Nones(all_dat, r, 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
            for f in flow_whole:
                r = gen_data(f[:, -1:], e['bin_freq'])
                all_dat = append_Nones(all_dat, r, 1)
                idx_bar += [all_dat.shape[1] - 1]
                lbl_bar += ['Flow Vel']
        if 'Fourier' in comb:
            for f in four_whole:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                r = gen_data(f, e['bin_freq'])
                all_dat = append_Nones(all_dat, r, 1)
                idx_four += [range(pre_len, all_dat.shape[1])]
                four_shape.append(four_mask_shape)

    all_dat = np.tile(all_dat, [source.shape[0], 1, 1])
    plot_params = {}
    plot_params['idx_bar'] = idx_bar
    plot_params['idx_flow'] = idx_flow
    plot_params['idx_four'] = idx_four
    plot_params['lbl_bar'] = lbl_bar
    plot_params['four_shape'] = four_shape
    print edge
    return all_dat, source, plot_params


def append_Nones(target, addition, axis=1):
    """Append to an array if it exists, otherwise make the array equal to
    addition """
    if target == None or len(target) == 0:
        target = addition
    else:
        target = np.append(target, addition, axis)

    return target


def do_classification(exp_type='SOM', combs=['Luminance', 'Contrast',
                        'Fourier'],
                      targets=[['Center', 'Center'], ['Center', 'Whole'],
                               ['Whole', 'Center'], ['Whole', 'Whole'],
                               ['Surround', 'Whole']],
                      max_comb=None, min_comb=None,
                       four_downsample=None, max_exp=None, sig_thresh=0.,
                       randomise=None, folds=5, filt=0.2):
    # Sub directory of the figure path to put the plots in
    fig_path = startup.fig_path + 'ephys/%s/pred/' % (exp_type)
    mov_path = startup.data_path + 'ephys/%s/' % (exp_type)
    if randomise is not None:
        pth = fig_path + randomise + '_' + str(filt)
    else:
        pth = fig_path + str(filt)

    if os.path.exists(pth):
        if os.path.exists(pth + '/summary_all.csv'):
            print pth, ' already exists. Skipping'
            return ['skipped']
    else:
        os.makedirs(pth)

    full_targets = []
    for [targ_type, src_type] in targets:
        full_targets.append('%s_%s' % (targ_type, src_type))

    dat = load_EphysData(exp_type, filt=filt)

    comb_corrs = []
    if max_comb is None:
        max_comb = len(combs)
    if min_comb is None:
        min_comb = 0

    cell_results = {}
    for num_combs in [1]:#, len(combs)]:
        for comb in itertools.combinations(combs, num_combs):
            print comb
            full_comb = str(num_combs) + '_' + "_".join(comb)
            comb_vals = {'Overall': []}
            for i, e in enumerate(dat.values()):
                if max_exp is not None and i >= max_exp:
                    break
                expdate = e['expdate']
                cellid = str(e['cellid'])
                if cellid not in cell_results:
                    cell_results[cellid] = {}
                if  full_comb not in cell_results[cellid]:
                    cell_results[cellid][full_comb] = {}
                if not os.path.exists(mov_path + cellid + '_processed.npz'):
                    print '\nNo movie found ', cellid
                    continue
                else:
                    print '\ndoing ', e['cellid']

                clf_vals = []
                sig_found = False
                results = {}
                for [targ_type, src_type] in targets:
                    k = '%s_%s' % (targ_type, src_type)
                    if targ_type not in cell_results[cellid][full_comb]:
                        cell_results[cellid][full_comb][targ_type] = {}
                    if src_type not in cell_results[cellid][full_comb][targ_type]:
                        cell_results[cellid][full_comb][targ_type][src_type] = {}

                    X, y, plot_params = get_mov_data(comb, targ_type, src_type,
                                        e, cellid, exp_type, four_downsample,
                                        randomise)

                    if randomise is not None:
                        fname = '%s%s_%s/%s/' % (fig_path,
                                                  randomise, str(filt),
                                                  cellid)
                    else:
                        fname = '%s%s/%s/' % (fig_path,
                                                  str(filt),
                                                  cellid)
                    if not os.path.exists(fname):
                        os.makedirs(fname)
                    print fname
                    # ignore edge effects
                    cv = LeaveOneOut(X.shape[1], indices=True)
                    cv = KFold(X.shape[1], folds, indices=True, shuffle=False)
                    pred = np.zeros(y[0].shape)
                    X_dims = X.shape[2]

                    pred, coefs = CV(Lasso,
                            X, y, folds=folds, clf_args={'alpha': 0.01})
#                    coefs = []
##                    lassoCV(X.reshape(-1, X_dims),
##                                 y.ravel())
#                    for train, test in cv:
#
#                        XX = X[:, train].reshape(-1, X_dims)
#                        yy = y[:, train].ravel().copy()
#                        Xt = X[0, test]
##                        yy[yy == 0] = 0.001
##                        yy = np.log(yy)
#                        scaler = StandardScaler()
#                        scaler.fit(XX)
##                        XX = scaler.transform(XX)
##                        Xt = scaler.transform(Xt)
#                        regr = clf(alpha=0.01)
#                        regr.fit(XX, yy)
#                        coefs.append(regr.coef_)
#                        p = regr.predict(Xt)
##                        p = np.exp(p)
#                        pred[test] = p
                    pred, _ = filter(pred, e['bin_freq'])
                    coefs = np.array(coefs).mean(0)
                    clf_vals.append([targ_type, ])
                    edge = e['edge']
                    mn = y.mean(0)
                    std = np.std(y, 0)
                    crr_pred = do_thresh_corr(mn[edge: -edge],
                                              pred[edge: -edge])
                    if crr_pred > sig_thresh:
                        sig_found = True

                    cell_results[cellid][full_comb][targ_type][src_type] = crr_pred
                    comb_vals['Overall'] += [crr_pred]
                    if k in comb_vals:
                        comb_vals[k] += [crr_pred]
                    else:
                        comb_vals[k] = [crr_pred]

                    xcorr = []
                    for i in range(y.shape[0]):
                        xcorr.append(do_thresh_corr(y[i], mn))
                    crr_y = np.array(xcorr).mean()
                    print crr_pred
                    results[k] = [pred, mn, std, crr_pred, crr_y,
                                          coefs, plot_params]
                    # only do plots for the fourier trained classifier
                if sig_found:
                    cmb = "_".join(comb)
                    if cmb == 'Fourier' and len(comb) == 1:
                        plot_preds_fourier(cellid, results, full_targets,
                                       fname + cmb)
                    else:
                        plot_preds(cellid, results, full_targets, cmb,
                                       fname + cmb)

            comb_corrs.append([comb, comb_vals])
    # make this a boxplot
    if randomise is not None:
        fp = fig_path + randomise + '_'
    else:
        fp = fig_path

    plot_summary(comb_corrs, full_targets, fp, str(filt))
    plot_cell_summary(cell_results, fp + str(filt) + '/')

    return comb_corrs


if __name__ == "__main__":
    corrs = []
    exp_type = 'FS'
    # now its mask movie values for all predictions
    # try also whole
    # and make box plots! 
    downsample = 11
    exp_types = ['FS', 'PYR', 'SOM']
    for exp_type in exp_types:
        for filt in [0.1]:#np.arange(0.1, 1.1, 0.1):
            print 'DOWNSAMPLE %s' % (str(downsample))
            corrs.append(do_classification(exp_type=exp_type, min_comb=None,
                                        max_comb=None,
                                        targets=[['Center', 'Center'],
                                                 ['Whole', 'Whole']
                                                 #['Surround', 'Whole']
                                                 ],
                                           folds=20,
                                        #combs=['Fourier'],
                                        #combs=['Luminance', 'Flow'],
                                        max_exp=None,
                                       #targets=['Center', 'CenterWhole', 'Whole', 'WholeWhole'],
                                       four_downsample=downsample, randomise=None,
                                       filt=filt))
    for c in corrs:
        print c[0]