import pickle
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, plot_mean_std
import pylab as plt
import numpy as np
import os

randomise = None
filt = 0.1
for exp_type in ['FS', 'PYR', 'SOM']:
    fig_path = startup.fig_path + 'Sparseness/%s/pred/stability/' % (exp_type)
    dat_path = startup.data_path + 'Sparseness/%s/pred/' % (exp_type)

    dat_path = dat_path + str(filt)
    fig_path = fig_path

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    dat_file = dat_path + '/preds.pkl'
    with open(dat_file, 'rb') as infile:
        cell_results = pickle.load(infile)
    print cell_results.keys()

    for cell in cell_results.keys():
        print 'doing ', cell
        xaxis = []
        xvals = []
        fig = plt.figure(figsize=(14, 12))
        ax = plt.subplot(111)
        adjust_spines(ax, ['bottom', 'left'])
        offset = -1
        for k in cell_results[cell].keys():
            offset += 2
            min_x = offset
            for cmb in cell_results[cell][k].keys():
                xaxis.append(cmb)
                xvals.append(offset)
                dat = []
                for s in cell_results[cell][k][cmb].keys():
                    #dat.append([s, cell_results[cell][k][cmb][s]['crr_pred']])
                    dat.append(cell_results[cell][k][cmb][s]['crr_pred'])
                do_spot_scatter_plot(np.array(dat, dtype=np.float), offset,
                                     width=0.4)
                offset += 1
            max_x = offset
            plt.text(min_x + (max_x - min_x) / 2., 1., k, ha='center')

        plt.xticks(xvals, xaxis, rotation='vertical')
        plt.plot([-1, len(xaxis) + 1], [0, 0], '--')
        plt.xlim(0, offset)
        plt.ylim(-0.05, 1)
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.95,
                           wspace=0.3, hspace=0.34)
        plt.ylabel('Correlation between prediction and Experimental Mean')
        fname = '%s%s_pred' % (fig_path, cell)
        fig.savefig(fname + '.eps')
        fig.savefig(fname + '.png')
        #plt.show()
        plt.close(fig)