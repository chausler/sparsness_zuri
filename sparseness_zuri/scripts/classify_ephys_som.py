import numpy as np
import sys
import pylab as plt
sys.path.append('..')
#sys.path = ['/home/chris/programs/aa_scikits/scikit-learn'] + sys.path
import startup
from scipy.stats import pearsonr, spearmanr
from plotting.utils import adjust_spines
from data_utils.load_ephys import load_EphysData_SOM, load_movie_data
from sklearn.linear_model import LinearRegression as clf
#from sklearn.linear_model import SGDRegressor as clf
import itertools

# Sub directory of the figure path to put the plots in
fig_path = startup.fig_path + 'ephys/som/pred/'

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

    new_flow = []
    for f in flow:
        n, _ = np.histogram(np.degrees(f[0]), bins=bins)
        n[0] += n[-1]
        n = n[:-1]
        new_flow.append(n.tolist() + [f[1]])

    new_flow = np.array(new_flow, dtype=np.float)
    return new_flow


def plot_mean_std(mn, std, plt_num=111):
    ax = plt.subplot(plt_num)
    plt.hold(True)
    plt.fill_between(range(len(std)), mn - std, mn + std, facecolor=clr2)
    plt.plot(range(len(std)), mn, color=clr1,
             label='Mean Center', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False, ncol=2)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Cross Trial Correlation: %.2f  P: %.2f' % (crr_targ, p_targ))


def plot_weight_dist(weights, plt_num=111):
    ax = plt.subplot(plt_num)
    plt.hold(True)
    plt.hist(weights)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Weight Distribution')


def plot_bar_weights(weights, labels, ylims, plt_num=111):
    ax = plt.subplot(plt_num)
    plt.hold(True)
    plt.bar(range(len(weights)), weights)
    ax.set_xticks(np.arange(len(weights)) + 0.5)
    ax.set_xticklabels(labels)
    plt.ylim(ylims)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Other Weights')


def plot_prediction(pred, actual, title, plt_num=111):
    ax = plt.subplot(plt_num)
    plt.hold(True)
    plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
    plt.plot(range(len(pred)), pred, color=clr4, linewidth=2,
             label='Predicted')
    plt.plot(range(len(pred)), actual, color=clr1,
             label='Mean Experimental', linewidth=2)
    plt.legend(bbox_to_anchor=(0.55, 0.95, 0.4, 0.1), frameon=False)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)


def plot_flow(weights, ylims, plt_num=111):
    plt.subplot(plt_num, polar=True)
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


def plot_four(four_weights, ylims, plt_num=111, title=None):
    ax = plt.subplot(plt_num)
    ext = four_shape[0][0] / 2.
    im = plt.imshow(four_weights.T, cmap=plt.cm.gray, interpolation='None',
               extent=(-ext, ext, -ext, ext))
    plt.clim(ylims)
    plt.xlabel('Cycles/Patch', fontsize='large')
    plt.ylabel('Cycles/Patch', fontsize='large')
    adjust_spines(ax, ['bottom', 'left'])
    if title is None:
        title = 'Spatial Fourier Weights'
    plt.title(title)
    cax = plt.subplot(244)
    plt.colorbar(im, cax=cax, cmap=plt.cm.gray)
    plt.title('Colour Scale')


def append_Nones(target, addition, axis=1):
    if target == None or len(target) == 0:
        target = addition
    else:
        target = np.append(target, addition, axis)

    return target


ephys = load_EphysData_SOM()

# which
combs = ['Luminance', 'Contrast', 'Flow Directions',
         'Flow Strength', 'Fourier']

#combs = ['Fourier']

comb_corrs = []

for num_combs in range(0, len(combs)):
    for comb in itertools.combinations(combs, num_combs + 1):
        print comb
        all_corrs = []
        surr_corrs = []
        mask_corrs = []
        whole_corrs = []
        for i, e in enumerate(ephys.values()):
            expdate = e['expdate']
            print 'doing ', e['expdate']
            mov = load_movie_data(expdate)
            lum_mask = mov['lum_mask']
            con_mask = mov['con_mask']
            flow_mask = mov['flow_mask']
            four_mask = mov['four_mask']
            four_mask_shape = four_mask.shape[1:]
            four_mask = four_mask.reshape(four_mask.shape[0], -1)
            lum_surr = mov['lum_surr']
            con_surr = mov['con_surr']
            flow_surr = mov['flow_surr']
            four_surr = mov['four_surr']
            four_surr_shape = four_surr.shape[1:]
            four_surr = four_surr.reshape(four_surr.shape[0], -1)
            lum_whole = mov['lum_whole']
            con_whole = mov['con_whole']
            flow_whole = mov['flow_whole']
            four_whole = mov['four_whole']
            four_whole_shape = four_whole.shape[1:]
            four_whole = four_whole.reshape(four_whole.shape[0], -1)
            flow_mask = bin_flow(flow_mask)
            flow_surr = bin_flow(flow_surr)
            flow_whole = bin_flow(flow_whole)

            for targ_type in ['Center', 'Surround', 'Whole']:
                fname = '%s%s_%s_pred' % (fig_path, expdate, targ_type)
                print fname

                idx_bar = []
                idx_flow = []
                idx_four = []
                lbl_bar = []
                four_shape = []
                all_dat = None
                if targ_type == 'Center':
                    source = e['psth_c']
                    if 'Luminance' in comb:
                        all_dat = append_Nones(all_dat,
                                               lum_mask[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Luminance']
                    if 'Contrast' in comb:
                        all_dat = append_Nones(all_dat,
                                               con_mask[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Contrast']
                    if 'Flow Directions' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat,
                                               flow_mask[:, :-1], 1)
                        idx_flow += [range(pre_len, all_dat.shape[1])]
                    if 'Flow Strength' in comb:
                        all_dat = append_Nones(all_dat,
                                               flow_mask[:, -1:], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Flow Vel']
                    if 'Fourier' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, four_mask, 1)
                        idx_four += [range(pre_len, all_dat.shape[1])]
                        four_shape.append(four_mask_shape)

                elif targ_type == 'Surround':
                    source = e['psth_s']
                    if 'Luminance' in comb:
                        all_dat = append_Nones(all_dat,
                                               lum_surr[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Luminance']
                    if 'Contrast' in comb:
                        all_dat = append_Nones(all_dat,
                                               con_surr[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Contrast']
                    if 'Flow Directions' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, flow_surr[:, :-1], 1)
                        idx_flow += [range(pre_len, all_dat.shape[1])]
                    if 'Flow Strength' in comb:
                        all_dat = append_Nones(all_dat, flow_surr[:, -1:], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Flow Vel']
                    if 'Fourier' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, four_surr, 1)
                        idx_four += [range(pre_len, all_dat.shape[1])]
                        four_shape.append(four_mask_shape)

                elif targ_type == 'Whole':
                    source = e['psth_w']
                    if 'Luminance' in comb:
                        all_dat = append_Nones(all_dat,
                                               lum_mask[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Luminance']
                    if 'Contrast' in comb:
                        all_dat = append_Nones(all_dat,
                                               con_mask[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Contrast']
                    if 'Flow Directions' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, flow_mask[:, :-1], 1)
                        idx_flow += [range(pre_len, all_dat.shape[1])]
                    if 'Flow Strength' in comb:
                        all_dat = append_Nones(all_dat, flow_mask[:, -1:], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Flow Vel']
                    if 'Fourier' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, four_mask, 1)
                        idx_four += [range(pre_len, all_dat.shape[1])]
                        four_shape.append(four_mask_shape)

                    if 'Luminance' in comb:
                        all_dat = append_Nones(all_dat,
                                               lum_surr[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Luminance']
                    if 'Contrast' in comb:
                        all_dat = append_Nones(all_dat,
                                               con_surr[:, np.newaxis], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Contrast']
                    if 'Flow Directions' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, flow_surr[:, :-1], 1)
                        idx_flow += [range(pre_len, all_dat.shape[1])]
                    if 'Flow Strength' in comb:
                        all_dat = append_Nones(all_dat, flow_surr[:, -1:], 1)
                        idx_bar += [all_dat.shape[1] - 1]
                        lbl_bar += ['Flow Vel']
                    if 'Fourier' in comb:
                        if all_dat != None:
                            pre_len = all_dat.shape[1]
                        else:
                            pre_len = 0
                        all_dat = append_Nones(all_dat, four_surr, 1)
                        idx_four += [range(pre_len, all_dat.shape[1])]
                        four_shape.append(four_mask_shape)

                all_dat = np.tile(all_dat, [source.shape[0], 1])
                regr = clf()
                regr.fit(all_dat, source.ravel())
                pred = regr.predict(all_dat[:source.shape[1]])
                mn = source.mean(0)
                std = np.std(source, 0)
                [crr_pred, p_pred] = corr(mn, pred)

                if p_pred < 0.05:
                    crr_pred = np.nan_to_num(crr_pred)
                else:
                    crr_pred = 0
                all_corrs += [crr_pred]

                if targ_type == 'Center':
                    mask_corrs += [crr_pred]
                elif targ_type == 'Surround':
                    surr_corrs += [crr_pred]
                elif targ_type == 'Whole':
                    whole_corrs += [crr_pred]

                xcorr = []
                for i in range(source.shape[0]):
                    xcorr.append(corr(source[i], mn))
                xcorr = np.array(xcorr)
                [crr_targ, p_targ] = xcorr.mean(0)

                # only do plots for the fourier trained classifier
                if len(comb) == 1 and comb[0] == 'Fourier':
                    fig = plt.figure(figsize=(14, 9))
                    fig.set_facecolor('white')
                    plt.suptitle('Experiment: %s Target: %s' %
                                 (expdate, targ_type))
                    lm = np.maximum(np.abs(regr.coef_.min()), regr.coef_.max())
                    ylims = [-lm, lm]

                    plot_mean_std(mn, std, 221)
                    plot_weight_dist(regr.coef_, 243)
                    #plot_bar_weights(regr.coef_[idx_bar], lbl_bar, ylims, 244)
                    plot_prediction(pred, mn,
                        'Prediction Corr to Mean: %.2f  P: %.2f' %
                        (crr_pred, p_pred), 223)
                    #plot_flow(regr.coef_[idx_flow], ylims, 247)
                    for jj, f in enumerate(idx_four):
                        title = None
                        if len(idx_four) > 1:
                            if jj == 0:
                                title = 'Mask Fourier Weights'
                            else:
                                title = 'Surround Fourier Weights'
                        four_weights = regr.coef_[f].reshape(four_shape[0])
                        plot_four(four_weights, ylims, 247 + jj, title=title)
                    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97,
                                        top=0.87, wspace=0.3, hspace=0.34)
                    fig.savefig(fname + '.eps')
                    fig.savefig(fname + '.png')
                    plt.close(fig)

        comb_vals = []
        comb_vals.append(['mask', np.array(mask_corrs).mean(),
                          np.std(np.array(mask_corrs))])
        comb_vals.append(['surround', np.array(surr_corrs).mean(),
                          np.std(np.array(surr_corrs))])
        comb_vals.append(['whole', np.array(whole_corrs).mean(),
                          np.std(np.array(whole_corrs))])
        comb_vals.append(['overall', np.array(all_corrs).mean(),
                          np.std(np.array(all_corrs))])
        print comb
        print comb_vals
        comb_corrs.append([comb, comb_vals])

fig = plt.figure(figsize=(14, 12))
ax = plt.subplot(111)
xaxis = []
for i, [comb, vals] in enumerate(comb_corrs):
    xaxis.append(comb)
    for [name, mn, std] in vals:

        if name == 'mask':
            col = 'r'
            offset = 0
        elif name == 'surround':
            col = 'b'
            offset = 0.2
        elif name == 'whole':
            col = 'g'
            offset = 0.4
        elif name == 'overall':
            col = '0.3'
            offset = 0.6
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
fname = '%s%s_pred' % (fig_path, 'summary')
fig.savefig(fname + '.eps')
fig.savefig(fname + '.png')
plt.show()
plt.close(fig)
