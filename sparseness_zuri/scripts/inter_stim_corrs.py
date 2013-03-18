import sys
sys.path.append('..')
from startup import *
import numpy as np
from data_utils.utils import do_thresh_corr
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot
from data_utils.load_ephys import load_EphysData
import os
# Sub directory of the figure path to put the plots in
corr_type = 'spearmanr'
corr_type = 'pearsonr'
colors = ['r', 'g', 'b']
#fig_path = fig_path + 'Sparseness/%s/initial/' % (exp_type)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

exp_types = ['SOM', 'FS', 'PYR']


corrs = []
cells = []
for exp_type in exp_types:
    dat = load_EphysData(exp_type)
    for d in dat.values():
        cellid = d['cellid']
        print 'doing ', cellid
        psth_s = d['psth_s']
        psth_c = d['psth_c']
        psth_w = d['psth_w']

        mn_c = psth_c.mean(0)
        mn_w = psth_w.mean(0)
        mx = np.maximum(mn_c.max(), mn_w.max())
        if psth_s is not None:
            mn_s = psth_s.mean(0)
            mx = np.maximum(mx, mn_s.max())

        cr_c_w = None
        cr_c_s = None
        cr_w_s = None

        mn_c = mn_c / mx
        mn_w = mn_w / mx
        cr_c_w = do_thresh_corr(mn_w, mn_c, corr_type=corr_type)
        if psth_s is not None:
            mn_s = mn_s / mx
            cr_c_s = do_thresh_corr(mn_s, mn_c, corr_type=corr_type)
            cr_w_s = do_thresh_corr(mn_s, mn_w, corr_type=corr_type)
        corrs.append([cr_c_w, cr_c_s, cr_w_s])
        cells.append([exp_type, cellid])

corrs = np.array(corrs)
print corrs
cells = np.array(cells)

fig = plt.figure(figsize=(10, 12))
fig.set_facecolor('white')

x = np.arange(len(corrs))
ax = plt.subplot(411)
plt.hold(True)
plt.plot([-1, x[-1]], [0, 0], 'k--')
for i, exp_type in enumerate(exp_types):
    idx = cells[:, 0] == exp_type
    plt.scatter(x[idx], corrs[idx, 0], color=colors[i])
plt.ylim(-0.3, 1)
plt.xlim(-1, len(x))
adjust_spines(ax, ['left'])
plt.title('Centre vs Whole')
plt.ylabel('Correlation Coef')

ax = plt.subplot(412)
plt.hold(True)
plt.plot([-1, x[-1]], [0, 0], 'k--')
for i, exp_type in enumerate(exp_types[:-1]):
    idx = cells[:, 0] == exp_type
    plt.scatter(x[idx], corrs[idx, 1], color=colors[i])
plt.ylim(-0.3, 1)
plt.xlim(-1, len(x))
plt.title('Centre vs Surround')
plt.ylabel('Correlation Coef')
adjust_spines(ax, ['left'])

ax = plt.subplot(413)
plt.hold(True)
plt.plot([-1, x[-1]], [0, 0], 'k--')
for i, exp_type in enumerate(exp_types[:-1]):
    idx = cells[:, 0] == exp_type
    plt.scatter(x[idx], corrs[idx, 2], color=colors[i])
plt.ylim(-0.3, 1)
plt.xlim(-1, len(x))
adjust_spines(ax, ['bottom', 'left'])
plt.title('Whole vs Surround')
plt.ylabel('Correlation Coef')
plt.xlabel('Neuron #')

ax = plt.subplot(414)
plt.hold(True)
plt.plot([-1, x[-1]], [0, 0], 'k--')
xval = np.array([1])
lbls = []
tks = []
for i, exp_type in enumerate(exp_types):
    idx = cells[:, 0] == exp_type
    do_box_plot(corrs[idx, 0], xval, colors[i])
    tks.append(xval[0])
    lbls.append('Centre Vs Whole')
    if exp_type != 'PYR':
        xval += 1
        plt.text(xval[0] - 0.25, -0.15, exp_type)
        do_box_plot(corrs[idx, 1], xval, colors[i])
        tks.append(xval[0])
        lbls.append('Centre Vs Surround')
        xval += 1
        do_box_plot(corrs[idx, 2], xval, colors[i])
        tks.append(xval[0])
        lbls.append('Surround Vs Whole')
        xval += 2
    else:
        plt.text(xval[0] - 0.25, -0.15, exp_type)
plt.ylim(-0.3, 1)
plt.xlim(0, xval + 1)
print lbls
plt.xticks(tks, lbls, rotation='vertical')
adjust_spines(ax, ['bottom', 'left'])
plt.title('Whole vs Surround')
plt.ylabel('Correlation Coef')

plt.subplots_adjust(left=0.07, bottom=0.17, right=0.95, top=0.95,
                            wspace=0.2, hspace=0.45)

plt.show()
