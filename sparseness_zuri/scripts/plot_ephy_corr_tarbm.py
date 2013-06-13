
import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import os
import pylab as plt
sys.path.append('..')
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_ephys import load_EphysData
patches = np.load('mask_data_complete.npy')
patches = patches.item()
crr_types = ['crr', 'crr_mn', 'cell_crr']
fig_path = startup.fig_path + 'Sparseness/DBN_preds/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

new_res = {}
mx = 0 
for exp_type in patches.keys():
    e = load_EphysData(exp_type)
    for cell_id in patches[exp_type].keys():
        dat = e[cell_id]['psth_c_shift']
        for rbm_type in patches[exp_type][cell_id]['corrs']:
            if rbm_type not in new_res:
                new_res[rbm_type] = {}
            for act_type in patches[exp_type][cell_id]['corrs'][rbm_type]:
                if act_type not in new_res[rbm_type]:
                    new_res[rbm_type][act_type] = {}
                if exp_type not in new_res[rbm_type][act_type]:
                    new_res[rbm_type][act_type][exp_type] = {}
                for crr_type in crr_types:
                    if crr_type not in new_res[rbm_type][act_type][exp_type]:
                        new_res[rbm_type][act_type][exp_type][crr_type] = {}
                pred = patches[exp_type][cell_id]['responses'][rbm_type][act_type][1].T
                for shift in patches[exp_type][cell_id]['corrs'][rbm_type][act_type]:
                    dt = dat[shift]
                    strt = dt.shape[1] - pred.shape[1]
                    dt = dt[:, strt:]
                    for crr_type in crr_types:
                        if shift not in new_res[rbm_type][act_type][exp_type][crr_type]:
                            new_res[rbm_type][act_type][exp_type][crr_type][shift] = []
                    for cell in patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift]:
                        crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr']
                        crr_mn = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr_mn']
                        cell_crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['cell_crr']
                        active = (pred[cell].max() > (pred[cell].mean() + 2 * np.std(pred[cell])))
                        #if (crr > 0.02 and crr >= (cell_crr * 0.9) and active): or crr_mn > 0.1:
                        if active and crr_mn > 0.2:
                            label = '%s %s %s %s %s: cell: %d, cell corr: %.3f, pred corr: %.3f, crr mn: %.3f' % (
                                            exp_type, cell_id, rbm_type,
                                            act_type, shift, cell, cell_crr, crr, crr_mn)
                            print label
                            plt.figure(figsize=(12, 8))
                            plt.title(label)
                            plt.hold(True)
                            #plt.plot(dt.T, '0.7')
                            mn_cell = dt.mean(0)
                            mn_cell = (mn_cell - mn_cell.mean()) / np.std(mn_cell)
                            mn_pred = (pred[cell] - pred[cell].mean()) / np.std(pred[cell])
                            plt.plot(mn_cell, 'k', lw=2)
                            print pred[cell], active
                            plt.plot(mn_pred, 'r')
                            #plt.ylim([-0.1, 1])
                            plt.show()
                            for crr_type in crr_types:
                                crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell][crr_type]
                                if crr > mx:
                                    mx = crr
                                new_res[rbm_type][act_type][exp_type][crr_type][shift].append(crr)

print mx
bins = np.linspace(0, mx + 0.05, 100)
for rbm_type in new_res:
    for act_type in new_res[rbm_type]:
        fig = plt.figure(figsize=(20, 14))
        fig.set_facecolor('white')
        plt.suptitle('%s  %s' % (rbm_type, act_type))
        rows = len(exp_type) * len(crr_types)
        cols = 16
        cnt = 1
        axes = []
        for e, exp_type in enumerate(sorted(new_res[rbm_type][act_type])):
            for c, crr_type in enumerate(sorted(new_res[rbm_type][act_type][exp_type])):
                for s, shift in enumerate(sorted(new_res[rbm_type][act_type][exp_type][crr_type])):
                    vals = np.array(new_res[rbm_type][act_type][exp_type][crr_type][shift])
                    ax = plt.subplot(rows, cols, cnt)
                    label = '%s %s' % (exp_type, crr_type)
                    if ax.is_first_col() and ax.is_last_row():
                        ax.text(-0.5, 0.5, label, transform=ax.transAxes, rotation='vertical', va='center', ha='center')
                        plt.setp(ax.get_xticklabels(), rotation='vertical')
                        adjust_spines(ax, ['left', 'bottom'])
                    elif ax.is_last_row():
                        plt.setp(ax.get_xticklabels(), rotation='vertical')
                        adjust_spines(ax, ['bottom'])
                    elif ax.is_first_col():
                        adjust_spines(ax, ['left'])
                        ax.text(-0.5, 0.5, label, transform=ax.transAxes, rotation='vertical', va='center', ha='center')
                    else:
                        adjust_spines(ax, [])
                    if ax.is_first_row():
                        plt.title(str(shift))
                    plt.hist(vals, bins=bins)
                    plt.ylim([0, 200])
                    cnt += 1
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.93, wspace=0.25, hspace=0.25)
        fig.savefig(fig_path + '%s_%s.eps' % (rbm_type, act_type))
        fig.savefig(fig_path + '%s_%s.png' % (rbm_type, act_type))
        #plt.show()
        plt.close(fig)
