
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
stim_type = 'psth_w_shift'
patches = np.load('mask_data_complete_%s.npy' % stim_type)
patches = patches.item()
crr_types = ['crr', 'crr_mn', 'cell_crr']
fig_path = startup.fig_path + 'Sparseness/DBN_preds/'
if not os.path.exists(fig_path + '/cells/'):
    os.makedirs(fig_path + '/cells/')

new_res = {}
mx = 0 
for exp_type in patches.keys():
    e = load_EphysData(exp_type)
    for cell_id in patches[exp_type].keys():
        dat = e[cell_id][stim_type]
        for rbm_type in patches[exp_type][cell_id]['corrs']:
            if rbm_type not in new_res:
                new_res[rbm_type] = {}
            for act_type in patches[exp_type][cell_id]['corrs'][rbm_type]:
                if act_type not in new_res[rbm_type]:
                    new_res[rbm_type][act_type] = {}
                if exp_type not in new_res[rbm_type][act_type]:
                    new_res[rbm_type][act_type][exp_type] = {}                
                pred = patches[exp_type][cell_id]['responses'][rbm_type][act_type][1].T
                cell_max = 0
                cell_max_id = None
                cell_max_shift = None
                for shift in patches[exp_type][cell_id]['corrs'][rbm_type][act_type]:
                    if shift not in new_res[rbm_type][act_type][exp_type]:
                        new_res[rbm_type][act_type][exp_type][shift] = []
                    dt = dat[shift]
                    strt = dt.shape[1] - pred.shape[1]
                    dt = dt[:, strt:]
                    cell_shift_max = 0
                    for cell in patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift]:
                        crr_mn = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr_mn']
                        crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr']
                        cell_crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['cell_crr']
                        active = (pred[cell].max() > (pred[cell].mean() + 2 * np.std(pred[cell])))
                        #if (crr > 0.02 and crr >= (cell_crr * 0.9) and active): or crr_mn > 0.1:
                        if active and crr_mn > 0:
                            if crr_mn > mx:
                                mx = crr_mn
                            if crr_mn > cell_shift_max:
                                cell_shift_max = crr_mn
                            if crr_mn > cell_max:
                                cell_max_id = cell
                                cell_max_shift = shift
                                cell_max = crr_mn
                            #print crr_mn
                    new_res[rbm_type][act_type][exp_type][shift].append(cell_max)

                if cell_max_id is not None and cell_max > 0.3:
                    crr_mn = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][cell_max_shift][cell_max_id]['crr_mn']
                    label = 'exp cell: %s, shift: %s, dbn cell: %d, crr mn: %.3f' % (
                                    cell_id, shift, cell, crr_mn)
                    fname = '%s_%s_%s_%s_%s' % (
                                    exp_type, rbm_type, act_type, stim_type, cell_id)
                    print fname
                    print label

                    fig = plt.figure(figsize=(7, 4))
                    fig.set_facecolor('white')
                    ax = plt.subplot(111)
                    plt.suptitle(label)
                    plt.hold(True)
                    #plt.plot(dt.T, '0.7')
                    mn_cell = dt.mean(0)
                    mn_cell = (mn_cell - mn_cell.mean()) / np.std(mn_cell)
                    mn_pred = (pred[cell] - pred[cell].mean()) / np.std(pred[cell])
                    plt.plot(mn_cell, 'k', lw=2, label='Ephys Cell (Mean)')
                    plt.plot(mn_pred, 'r', label='aTRBM Cell')
                    plt.ylabel('Normalised Activation')
                    plt.xlabel('Sample')
                    plt.legend(bbox_to_anchor=(0.1, 0, 0.15, 0.91), bbox_transform=plt.gcf().transFigure, frameon=False, prop={'size':10})
                    plt.subplots_adjust(left=0.28, bottom=0.11, right=0.96, top=0.88, wspace=0.25, hspace=0.25)
                    adjust_spines(ax, ['left', 'bottom'])
                    fig.savefig(fig_path + '/cells/%s.eps' % (fname))
                    fig.savefig(fig_path + '/cells/%s.png' % (fname))
                    #plt.show()
                    plt.close(fig)


print mx
bins = np.linspace(0, mx + 0.05, 10)
print bins
for rbm_type in new_res:
    fig = plt.figure(figsize=(20, 14))
    fig.set_facecolor('white')
    plt.suptitle('%s' % (rbm_type))
    rows = len(exp_type) * len(new_res[rbm_type])
    cols = 16
    cnt = 1
    axes = []
    for act_type in new_res[rbm_type]:
        for e, exp_type in enumerate(sorted(new_res[rbm_type][act_type])):
                for s, shift in enumerate(sorted(new_res[rbm_type][act_type][exp_type])):
                    vals = np.array(new_res[rbm_type][act_type][exp_type][shift])
                    print rbm_type, act_type, exp_type, shift, vals.mean()
                    ax = plt.subplot(rows, cols, cnt)
                    label = '%s %s' % (exp_type, act_type)
                    if ax.is_first_col() and ax.is_last_row():
                        ax.text(-0.5, 0.5, label, transform=ax.transAxes,
                                rotation='vertical', va='center', ha='center')
                        plt.setp(ax.get_xticklabels(), rotation='vertical')
                        adjust_spines(ax, ['left', 'bottom'])
                    elif ax.is_last_row():
                        plt.setp(ax.get_xticklabels(), rotation='vertical')
                        adjust_spines(ax, ['bottom'])
                    elif ax.is_first_col():
                        adjust_spines(ax, ['left'])
                        ax.text(-0.5, 0.5, label, transform=ax.transAxes,
                                rotation='vertical', va='center', ha='center')
                    else:
                        adjust_spines(ax, [])
                    if ax.is_first_row():
                        plt.title(str(shift))
                    plt.hist(vals, bins=bins)
                    plt.ylim([0, len(vals)])
                    plt.title('%.4f' % vals.mean())
                    cnt += 1
    
    fig.savefig(fig_path + '%s_%s.eps' % (rbm_type, stim_type))
    fig.savefig(fig_path + '%s_%s.png' % (rbm_type, stim_type))
    #plt.show()
    plt.close(fig)

print 'CHECK WHAT TYPE OF STIMULUS::: WHOLE FIELD OR CENTRE: DO THE PEAKS OF SHIFT DIFFER BETWEEN THE TWO, WHAT DO THE RECEPTIVE FIELDS LOOK LIKE?'
print 'DO THE SAME PLOT AS THE PRED WITH SCATTER'