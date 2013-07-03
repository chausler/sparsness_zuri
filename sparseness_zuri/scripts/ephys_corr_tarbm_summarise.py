import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import os
import pylab as plt
sys.path.append('..')
import startup
from data_utils.load_ephys import load_EphysData
stim_types = ['psth_c_shift', 'psth_w_shift']
crr_types = ['crr', 'crr_mn', 'cell_crr']
fig_path = startup.fig_path + 'Sparseness/DBN_preds/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
new_res = {}
mx = 0
for stim_type in stim_types:
    patches = np.load('mask_data_complete_%s.npy' % stim_type)
    patches = patches.item() 
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
                    if stim_type not in new_res[rbm_type][act_type][exp_type]:
                        new_res[rbm_type][act_type][exp_type][stim_type] = {}
                    pred = patches[exp_type][cell_id]['responses'][rbm_type][act_type][1].T
                    for shift in patches[exp_type][cell_id]['corrs'][rbm_type][act_type]:
                        if shift not in new_res[rbm_type][act_type][exp_type][stim_type]:
                            new_res[rbm_type][act_type][exp_type][stim_type][shift] = []
                        dt = dat[shift]
                        strt = dt.shape[1] - pred.shape[1]
                        dt = dt[:, strt:]
                        cell_max = 0
                        for cell in patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift]:
                            crr_mn = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr_mn_r2']
                            crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr']
                            cell_crr = patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['cell_crr']
                            active = (pred[cell].max() > (pred[cell].mean() + 2 * np.std(pred[cell])))
                            #if (crr > 0.02 and crr >= (cell_crr * 0.9) and active): or crr_mn > 0.1:
                            if active and crr_mn > 0:
                                if crr_mn > 0.5:
                                    label = '%s %s %s %s %s: cell: %d, cell corr: %.3f, pred corr: %.3f, crr mn: %.3f' % (
                                                    exp_type, cell_id, rbm_type,
                                                    act_type, shift, cell, cell_crr, crr, crr_mn)
                                    print label
#                                    fig = plt.figure(figsize=(12, 8))
#                                    plt.title(label)
#                                    plt.hold(True)
#                                    #plt.plot(dt.T, '0.7')
#                                    mn_cell = dt.mean(0)
#                                    mn_cell = (mn_cell - mn_cell.mean()) / np.std(mn_cell)
#                                    mn_pred = (pred[cell] - pred[cell].mean()) / np.std(pred[cell])
#                                    plt.plot(mn_cell, 'k', lw=2)
#                                    plt.plot(mn_pred, 'r')
#                                    #plt.ylim([-0.1, 1])
#                                    plt.show()
#                                    #plt.close(fig)

                                if crr_mn > mx:
                                    mx = crr_mn
                                if crr_mn > cell_max:
                                    cell_max = crr_mn
                                #print crr_mn
                        new_res[rbm_type][act_type][exp_type][stim_type][shift].append(cell_max)
                        
np.save('mask_data_summary', new_res)