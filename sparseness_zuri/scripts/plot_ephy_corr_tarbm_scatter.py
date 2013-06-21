
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

fig_path = startup.fig_path + 'Sparseness/DBN_preds/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

new_res = np.load('mask_data_summary.npy')
new_res = new_res.item()
exp_types = ['FS', 'SOM', 'PYR']

for rbm_type in new_res:
    for act_type in new_res[rbm_type]:
        fig = plt.figure(figsize=(12, 8))
        fig.set_facecolor('white')
        plt.suptitle('%s %s' % (rbm_type, act_type))
        cnt =1
        for exp_type in exp_types:
            mx = 0
            axes = []
            for stim_type in sorted(new_res[rbm_type][act_type][exp_type]):
                axes.append(plt.subplot(3, 2, cnt))
#                if ax.is_first_col() and ax.is_last_row():
#                    ax.text(-0.5, 0.5, label, transform=ax.transAxes,
#                            rotation='vertical', va='center', ha='center')
#                    plt.setp(ax.get_xticklabels(), rotation='vertical')
#                    adjust_spines(ax, ['left', 'bottom'])
#                elif ax.is_last_row():
#                    plt.setp(ax.get_xticklabels(), rotation='vertical')
#                    adjust_spines(ax, ['bottom'])
#                elif ax.is_first_col():
#                    adjust_spines(ax, ['left'])
#                    ax.text(-0.5, 0.5, label, transform=ax.transAxes,
#                            rotation='vertical', va='center', ha='center')
#                else:
#                    adjust_spines(ax, [])
                ttl = '%s %s' % (exp_type, stim_type)
                plt.title(ttl)
                plt.hold(True)
                for s, shift in enumerate(sorted(new_res[rbm_type][act_type][exp_type][stim_type])):
                    vals = np.array(new_res[rbm_type][act_type][exp_type][stim_type][shift])
                    vals = np.array([vals[vals > 0.].mean()])
                    if vals.max() > mx:
                        mx = vals.max()
                    do_spot_scatter_plot(vals, shift, 'k', 0.4, False, True)
                cnt += 1
            for ax in axes:
                ax.set_ylim(-0.01, mx + (mx * 0.1))
                ax.set_xlim(-15.5, 0.5)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.93, wspace=0.25, hspace=0.25)
        fig.savefig(fig_path + 'scatter_%s_%s.eps' % (rbm_type, act_type))
        fig.savefig(fig_path + 'scatter_%s_%s.png' % (rbm_type, act_type))
        
        plt.close(fig)

print 'CHECK WHAT TYPE OF STIMULUS::: WHOLE FIELD OR CENTRE: DO THE PEAKS OF SHIFT DIFFER BETWEEN THE TWO, WHAT DO THE RECEPTIVE FIELDS LOOK LIKE?'
print 'DO THE SAME PLOT AS THE PRED WITH SCATTER'