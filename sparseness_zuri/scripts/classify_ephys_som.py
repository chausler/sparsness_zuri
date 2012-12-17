import matplotlib
# force plots to file. no display. comment out to use plt.show()
matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')

#sys.path = ['/home/chris/programs/aa_scikits/scikit-learn'] + sys.path
import startup
from scipy.stats import pearsonr, spearmanr
from plotting.utils import adjust_spines
from data_utils.load_ephys import load_EphysData_SOM, load_parsed_movie_dat
#from sklearn.linear_model import LinearRegression as clf
#from sklearn.linear_model import Ridge as clf
from sklearn.linear_model import Lasso as clf
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
#from sklearn.linear_model import SGDRegressor as clf
import itertools
import os
# Sub directory of the figure path to put the plots in
fig_path = startup.fig_path + 'ephys/som/pred/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# what type of correlation to use. spearmanr or pearsonr
corr = spearmanr

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


def plot_mean_std(mn, std, title, plt_num=[1, 1, 1]):
    """plot the mean +/- std as timeline """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor=clr2)
    plt.plot(range(len(std)), mn, color=clr1,
             label='Mean Center', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
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


def plot_prediction(pred, actual, title, plt_num=[1, 1, 1]):
    """ plot the prediction of the model vs the mean of the actual
     experiment """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
    plt.plot(range(len(pred)), pred, color=clr4, linewidth=2,
             label='Predicted')
    plt.plot(range(len(pred)), actual, color=clr1,
             label='Mean Experimental', linewidth=2)
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
              ax_off=False):
    """plot the weights of the spatial fourier """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    ext = four_weights.shape[0] / 2.
    im = plt.imshow(four_weights.T, cmap=plt.cm.gray, interpolation='None',
               extent=(-ext, ext, -ext, ext))
    plt.clim(ylims)
    if ax_off:
        adjust_spines(ax, [])
    else:
        adjust_spines(ax, ['bottom', 'left'])
        plt.xlabel('Cycles/Patch', fontsize='large')
        plt.ylabel('Cycles/Patch', fontsize='large')
    if title is None:
        title = 'Spatial Fourier Weights'
    plt.title(title)
#    cax = plt.subplot(244)
#    plt.colorbar(im, cax=cax, cmap=plt.cm.gray)
#    plt.title('Colour Scale')


def plot_summary(comb_corrs, extras=''):
    """ plot a summary showing the mean and std for all attempted
    combinations"""
    fig = plt.figure(figsize=(14, 12))
    _ = plt.subplot(111)
    xaxis = []
    for i, [comb, vals] in enumerate(comb_corrs):
        xaxis.append(comb)
        for [name, mn, std] in vals:
            if name == 'mask':
                col = 'r'
                offset = 0
            elif name == 'surround':
                col = 'b'
                offset = 0.15
            elif name == 'whole':
                col = 'g'
                offset = 0.3
            elif name == 'whole center':
                col = 'y'
                offset = 0.45
            elif name == 'surround center':
                col = 'k'
                offset = 0.6
            elif name == 'overall':
                col = '0.3'
                offset = 0.75
            plt.errorbar([i + offset], [mn], [std], color=col)
            if i == 0:
                plt.plot([i + offset], [mn], 'o', color=col, label=name)
            else:
                plt.plot([i + offset], [mn], 'o', color=col)

    plt.xticks(range(len(xaxis)), xaxis, rotation='vertical')
    plt.xlim(-1, len(xaxis) + 1)
    plt.subplots_adjust(left=0.05, bottom=0.5, right=0.97, top=0.95,
                        wspace=0.3, hspace=0.34)
    plt.legend(loc=4)
    plt.ylabel('Correlation between prediction and Experimental Mean')
    fname = '%s%s_pred_%s' % (fig_path, 'summary', extras)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)


def plot_weights_fouriers(expdate, targ_type, coefs, mn, std, pred,
                          crr_y, p_y, crr_pred, p_pred, plot_params,
                          fname):
    fig = plt.figure(figsize=(14, 9))
    fig.set_facecolor('white')
    if targ_type == 'Center':
        rows = 2
    else:
        rows = 4
    plt.suptitle('Experiment: %s Target: %s' %
                 (expdate, targ_type))
    weights = coefs
    #weights = arctan_norm(weights)
    lm = np.maximum(np.abs(weights.min()),
                    weights.max())
    ylims = [-lm, lm]

    plot_mean_std(mn, std,
        'Mean Correlation. Trials to Mean: %.2f  P: %.2f' %
        (crr_y, p_y), [rows, 2, 1])
    plot_weight_dist(weights, [rows, 4, 3])
    #plot_bar_weights(regr.coef_[idx_bar], lbl_bar, ylims, 244)
    plot_prediction(pred, mn,
        'Prediction Corr to Mean: %.2f  P: %.2f' %
        (crr_pred, p_pred), [rows, 2, 3])
    #plot_flow(regr.coef_[idx_flow], ylims, 247)
    offset = 0
    if targ_type == 'Surround':
        offset = 1
    for jj, f in enumerate(plot_params['idx_four']):
        title = None
        if len(plot_params['idx_four']) > 1:
            if jj == 0:
                title = 'Mask Fourier Weights'
            else:
                title = 'Surround Fourier Weights'
        four_weights = weights[f].reshape(
                            plot_params['four_shape'][0])
        plot_four(four_weights, ylims, [rows, 4, 7 + jj + offset], title=title)
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                        top=0.87, wspace=0.3, hspace=0.6)

    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
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
                plot_four(four_weights, ylims, 
                          [rows, cols, (i * cols) + jj + offset], title, ax_off)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98,
            top=0.98, wspace=0.1, hspace=0.05)
    fname = '%s%s_pred' % (fig_path, 'compare_four')
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    #plt.show()
    plt.close(fig)



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
    plt.axvline(-np.log10(model.alpha), linestyle='--', color='k',
               label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(lambda)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
             '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.show()


def get_mov_data(comb, targ_type, e, expdate, exp_type, four_downsample=None):

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
    if targ_type == 'Center':
        source = e['psth_c']
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
        if 'Flow Directions' in comb:
            for f in flow_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat,
                                       f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
        if 'Flow Strength' in comb:
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
    elif targ_type == 'Surround':
        source = e['psth_s']
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
        if 'Flow Directions' in comb:
            for f in flow_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
        if 'Flow Strength' in comb:
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

    elif targ_type == 'Whole' or targ_type == 'Whole_Center'\
                    or targ_type == 'Surround_Center':
        if targ_type == 'Whole':
            source = e['psth_w']
        elif targ_type == 'Whole_Center':
            source = (e['psth_w'] - e['psth_c'])# / (e['psth_w'] + e['psth_c'])
            #h = 1./0.
        elif targ_type == 'Surround_Center':
            source = (e['psth_s'] - e['psth_c'])# / (e['psth_s'] + e['psth_c'])

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
        if 'Flow Directions' in comb:
            for f in flow_mask:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat,
                                       f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
        if 'Flow Strength' in comb:
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
        if 'Flow Directions' in comb:
            for f in flow_surr:
                if all_dat != None:
                    pre_len = all_dat.shape[1]
                else:
                    pre_len = 0
                all_dat = append_Nones(all_dat, f[:, :-1], 1)
                idx_flow += [range(pre_len, all_dat.shape[1])]
        if 'Flow Strength' in comb:
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


def do_classification(exp_type='SOM', combs=['Luminance', 'Contrast',
                        'Flow Directions', 'Flow Strength', 'Fourier'],
                      targets=['Center', 'Surround', 'Whole',
                                       'Whole_Center', 'Surround_Center'],
                      max_comb=None, min_comb=None,
                       four_downsample=None, max_exp=None):
    if exp_type == 'SOM':
        dat = load_EphysData_SOM()

    comb_corrs = []
    if max_comb is None:
        max_comb = len(combs)
    if min_comb is None:
        min_comb = 0
    for num_combs in range(min_comb, max_comb):
        for comb in itertools.combinations(combs, num_combs + 1):
            print comb
            all_corrs = []
            surr_corrs = []
            mask_corrs = []
            whole_corrs = []
            whole_cent_corrs = []
            surr_cent_corrs = []

            for i, e in enumerate(dat.values()):
                if max_exp is not None and i >= max_exp:
                    break
                expdate = e['expdate']
                print '\ndoing ', e['expdate']
                clf_vals = []
                for targ_type in targets:
                    fname = '%s%s_%s_pred' % (fig_path, expdate, targ_type)
                    print fname
                    X, y, plot_params = get_mov_data(comb, targ_type, e,
                                            expdate, exp_type, four_downsample)

                    cv = LeaveOneOut(X.shape[1], indices=True)
                    cv = KFold(X.shape[1], 30, indices=True, shuffle=True)
                    pred = np.zeros(y[0].shape)
                    X_dims = X.shape[2]
                    coefs = []
                    for train, test in cv:
#                        lassoCV(X[:, train].reshape(-1, X_dims),
#                                 y[:, train].ravel())
                        regr = clf(alpha=0.01)
                        regr.fit(X[:, train].reshape(-1, X_dims),
                                 y[:, train].ravel())
                        coefs.append(regr.coef_)
                        pred[test] = regr.predict(X[0, test])
                    coefs = np.array(coefs).mean(0)
                    clf_vals.append([targ_type, coefs, plot_params])
                    mn = y.mean(0)
                    std = np.std(y, 0)
                    [crr_pred, p_pred] = corr(mn, pred)
                    if p_pred < 0.05:
                        crr_pred = np.nan_to_num(crr_pred)
                    else:
                        crr_pred = 0.
                        p_pred = 0.
                    all_corrs += [crr_pred]

                    if targ_type == 'Center':
                        mask_corrs += [crr_pred]
                    elif targ_type == 'Surround':
                        surr_corrs += [crr_pred]
                    elif targ_type == 'Whole':
                        whole_corrs += [crr_pred]
                    elif targ_type == 'Whole_Center':
                        whole_cent_corrs += [crr_pred]
                    elif targ_type == 'Surround_Center':
                        surr_cent_corrs += [crr_pred]

                    xcorr = []
                    for i in range(y.shape[0]):
                        xcorr.append(corr(y[i], mn))
                    xcorr = np.nan_to_num(np.array(xcorr))
                    xcorr[xcorr[:, 1] > 0.05, 0] = 0
                    [crr_y, p_y] = xcorr.mean(0)
                    print crr_pred, p_pred
                    # only do plots for the fourier trained classifier
                    if True and len(comb) == 1 and comb[0] == 'Fourier':
                        plot_weights_fouriers(expdate, targ_type, coefs, mn,
                            std, pred, crr_y, p_y, crr_pred, p_pred,
                            plot_params, fname)
                if len(comb) == 1 and comb[0] == 'Fourier':
                    plot_clf_compare(clf_vals)

            comb_vals = []
            comb_vals.append(['mask', np.array(mask_corrs).mean(),
                              np.std(np.array(mask_corrs))])
            comb_vals.append(['surround', np.array(surr_corrs).mean(),
                              np.std(np.array(surr_corrs))])
            comb_vals.append(['whole', np.array(whole_corrs).mean(),
                              np.std(np.array(whole_corrs))])
            comb_vals.append(['whole center', np.array(whole_cent_corrs).mean(),
                              np.std(np.array(whole_corrs))])
            comb_vals.append(['surround center', np.array(surr_cent_corrs).mean(),
                              np.std(np.array(whole_corrs))])
            comb_vals.append(['overall', np.array(all_corrs).mean(),
                              np.std(np.array(all_corrs))])
            print comb
            print comb_vals
            comb_corrs.append([comb, comb_vals])

    plot_summary(comb_corrs, str(downsample))
    return comb_corrs


if __name__ == "__main__":
    corrs = []
    for downsample in [3, 5, 7, 9, 11, 13, None]:
        corrs.append(do_classification(min_comb=None, max_comb=None,
                                   combs=['Fourier'], max_exp=None,
                                   four_downsample=downsample))
    for c in corrs:
        print c[0]