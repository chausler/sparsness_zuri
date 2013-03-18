import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')

sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
import numpy.fft as fft
import startup
from scipy.stats import pearsonr, spearmanr
from plotting.utils import adjust_spines, do_box_plot
from data_utils.load_ephys import load_EphysData
from data_utils.movie import load_parsed_movie_dat
#from sklearn.linear_model import LinearRegression as clf
#from sklearn.linear_model import Ridge as clf
#from sklearn.linear_model import Lasso as clf
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.linear_model import SGDRegressor as clf
import itertools
import os



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


def reverse_fft(four_weights):
        A = fft.ifft2(four_weights)
        return np.sqrt(A.real ** 2 + A.imag ** 2)


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
    #plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
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
              ax_off=False, interpolation='None'):
    """plot the weights of the spatial fourier """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    ext = four_weights.shape[0] / 2.
    im = plt.imshow(four_weights.T, cmap=plt.cm.gray,
                    interpolation=interpolation,
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


def plot_summary(comb_corrs, fig_path, extras=''):
    """ plot a summary showing the mean and std for all attempted
    combinations"""
    fig = plt.figure(figsize=(14, 12))
    _ = plt.subplot(111)
    xaxis = []
    boxes = []
    lbls = []
    for i, [comb, vals] in enumerate(comb_corrs):
        xaxis.append(comb)
        offset = 0
        for [name, dat] in vals:
            if len(dat) == 0:
                continue
            if name == 'mask':
                col = 'r'
            elif name == 'surround':
                col = 'b'
            elif name == 'whole':
                col = 'g'
            elif name == 'overall':
                col = '0.3'
            do_box_plot(np.array(dat), np.array([i + offset]), col)
            offset += 0.15
            if i == 0:
                boxes.append(plt.Rectangle((0, 0), 1, 1, fc=col))
                lbls.append(name)

    plt.xticks(range(len(xaxis)), xaxis, rotation='vertical')
    plt.xlim(-1, len(xaxis) + 1)
    plt.subplots_adjust(left=0.05, bottom=0.5, right=0.97, top=0.95,
                       wspace=0.3, hspace=0.34)
    plt.legend(boxes, lbls, frameon=False, loc=4)
    plt.ylabel('Correlation between prediction and Experimental Mean')
    fname = '%s%s_pred_%s' % (fig_path, 'summary', extras)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.show()
    plt.close(fig)


def plot_weights_fouriers(cellid, targ_type, coefs, mn, std, pred,
                          crr_y, p_y, crr_pred, p_pred, plot_params,
                          fname):
    if p_pred > 0.05 or crr_pred == 0:
            return
    fig = plt.figure(figsize=(14, 9))
    fig.set_facecolor('white')
    if targ_type == 'Center':
        rows = 2
    else:
        rows = 4
    plt.suptitle('Experiment: %s Target: %s' %
                 (cellid, targ_type))
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
        ff = reverse_fft(four_weights)
        plot_four(ff, [ff.min(), ff.max()], [rows, 4, 7 + jj + offset], title=title, interpolation='bilinear')
        #plot_four(four_weights, ylims, [rows, 4, 7 + jj + offset + 1], title=title)
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
                          [rows, cols, (i * cols) + jj + offset], title, ax_off, interpolation='bilinear')
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98,
            top=0.98, wspace=0.1, hspace=0.05)
    fname = '%s%s_pred' % (fname, 'compare_four')
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


def get_mov_data(targ_type, e, expdate, exp_type, four_downsample=None):

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
        for f in four_surr:
            if all_dat != None:
                pre_len = all_dat.shape[1]
            else:
                pre_len = 0
            all_dat = append_Nones(all_dat, f, 1)
            idx_four += [range(pre_len, all_dat.shape[1])]
            four_shape.append(four_surr_shape)

    elif targ_type == 'Whole':
        source = e['psth_w']
        for f in four_mask:
            if all_dat != None:
                pre_len = all_dat.shape[1]
            else:
                pre_len = 0
            all_dat = append_Nones(all_dat, f, 1)
            idx_four += [range(pre_len, all_dat.shape[1])]
            four_shape.append(four_mask_shape)
#
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


def do_classification(exp_type='SOM', 
                      targets=['Center', 'Surround', 'Whole'],                      
                       four_downsample=None, max_exp=None):
    # Sub directory of the figure path to put the plots in
    fig_path = startup.fig_path + 'Sparseness/%s/pred/' % (exp_type)
    mov_path = startup.data_path + 'Sparseness/%s/' % (exp_type)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(fig_path + str(downsample)):
        os.makedirs(fig_path + str(downsample))

    dat = load_EphysData(exp_type)

    all_corrs = []
    surr_corrs = []
    mask_corrs = []
    whole_corrs = []

    for i, e in enumerate(dat.values()):
        if max_exp is not None and i >= max_exp:
            break
        expdate = e['expdate']
        cellid = e['cellid']
        if not os.path.exists(mov_path + cellid + '_processed.npz'):
            print '\nNo movie found ', cellid
            continue
        else:
            print '\ndoing ', e['cellid']
        clf_vals = []
        sig_found = False
        for targ_type in targets:
            fname = '%s%s/%s_%s_pred' % (fig_path,
                                          str(four_downsample),
                                          cellid, targ_type)
            print fname
            X, y, plot_params = get_mov_data(targ_type, e,
                                    cellid, exp_type, four_downsample)
            
            for xi in range(X.shape[2]):
                cv = LeaveOneOut(X.shape[1], indices=True)
                cv = KFold(X.shape[1], 20, indices=True, shuffle=True)
                pred = np.zeros(y[0].shape)
                X_dims = X.shape[2]
                coefs = []
                for train, test in cv:
                    regr = clf(alpha=0.01)
                    regr.fit(X[:, train, xi].reshape([-1, 1]),
                             y[:, train].ravel())
                    coefs.append(regr.coef_)
                    pred[test] = regr.predict(X[0, test, xi].reshape([-1, 1]))
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
                if crr_pred > 0:
                    sig_found = True
                if crr_pred > 0.3:
                    plt.figure()
                    plt.hold(True)
                    plt.plot(pred, 'r')
                    plt.plot(mn, 'k')
                    plt.title('Corr: %.2f' % crr_pred)
                    plt.show()
                all_corrs += [crr_pred]
                if targ_type == 'Center':
                    mask_corrs += [crr_pred]
                elif targ_type == 'Surround':
                    surr_corrs += [crr_pred]
                elif targ_type == 'Whole':
                    whole_corrs += [crr_pred]
    plt.subplot(211)
    plt.scatter(range(len(mask_corrs)), mask_corrs)
    plt.subplot(212)
    plt.scatter(range(len(whole_corrs)), whole_corrs)
    plt.show()

#            xcorr = []
#            for i in range(y.shape[0]):
#                xcorr.append(corr(y[i], mn))
#            xcorr = np.nan_to_num(np.array(xcorr))
#            xcorr[xcorr[:, 1] > 0.05, 0] = 0
#            [crr_y, p_y] = xcorr.mean(0)
#            print crr_pred, p_pred
            # only do plots for the fourier trained classifier
            
        

    comb_vals = []
    comb_vals.append(['mask', mask_corrs])
    comb_vals.append(['surround', surr_corrs])
    comb_vals.append(['whole', whole_corrs])
    comb_vals.append(['overall', all_corrs])
    comb_corrs.append([comb, comb_vals])
    # make this a boxplot
    plot_summary(comb_corrs, fig_path, str(downsample))
    return comb_corrs


if __name__ == "__main__":
    corrs = []
    exp_type = 'PYR'
    # now its mask movie values for all predictions
    # try also whole
    # and make box plots!
    for downsample in [11]: #[3, 5, 7, 9, 11, 13, None]:
        print 'DOWNSAMPLE %s' % (str(downsample))
        corrs.append(do_classification(exp_type=exp_type, 
                                    max_exp=None,
                                   targets=['Center', 'Whole'],
                                   four_downsample=downsample))
    for c in corrs:
        print c[0]