
import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')
from data_utils.movie import load_movie_data
from data_utils.load_ephys import load_EphysData
from data_utils.utils import corr_trial_to_mean, corr_trial_to_trial, do_thresh_corr


patches = np.load('mask_data_complete.npy')
patches = patches.item()

for exp_type in patches.keys():
    e = load_EphysData(exp_type)
    for cell_id in patches[exp_type].keys():
        dat = e[cell_id]['psth_c_shift']
        if 'corrs' not in patches[exp_type][cell_id]:
            patches[exp_type][cell_id]['corrs'] = {}
        for rbm_type in patches[exp_type][cell_id]['responses'].keys():
            if rbm_type == 'movie' or rbm_type == 'responses':
                continue
            if rbm_type not in patches[exp_type][cell_id]['corrs']:
                patches[exp_type][cell_id]['corrs'][rbm_type] = {}
            for act_type in patches[exp_type][cell_id]['responses'][rbm_type].keys():
                print act_type
                if act_type not in patches[exp_type][cell_id]['corrs'][rbm_type]:
                    patches[exp_type][cell_id]['corrs'][rbm_type][act_type]  = {}
                pred = patches[exp_type][cell_id]['responses'][rbm_type][act_type][1]
                for shift in e[cell_id]['shifts']:
                    print shift
                    if shift not in patches[exp_type][cell_id]['corrs'][rbm_type][act_type]:
                        patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift]  = {}
                    dt = dat[shift]
                    strt = dt.shape[1] - pred.shape[0]
                    dt = dt[:, strt:]
                    for cell, cell_dat in enumerate(pred.T):
                        if cell not in patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift]:
                            patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]  = {}
                        if 'cell_crr' in patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]:
                            print 'skipping ', cell
                            #continue
                        crr = corr_trial_to_mean(dt, cell_dat)
                        crr_mn = do_thresh_corr(dt.mean(0), cell_dat)
                        cell_crr = corr_trial_to_trial(dt)
                        patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr'] = crr
                        patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['crr_mn'] = crr_mn
                        patches[exp_type][cell_id]['corrs'][rbm_type][act_type][shift][cell]['cell_crr'] = cell_crr
                        if (crr > 0.01 and crr >= cell_crr) or crr_mn > 0.01:
                            print '%s %s %s %s %s: cell: %d, corr: %.3f, pred corr: %.3f, crr mn: %.3f' % (
                                            exp_type, cell_id, rbm_type,
                                            act_type, shift, cell, cell_crr, crr, crr_mn)
                print 'saving'
                np.save('mask_data_complete', patches)
#    [rbm_type]


