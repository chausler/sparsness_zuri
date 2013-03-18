import pickle
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, plot_mean_std
import pylab as plt
import numpy as np
import os
from sklearn.decomposition import PCA

randomise = None
filt = 0.1

lags = {'FS': '-4', 'PYR': '-4', 'SOM': '-9'}

fig_path = startup.fig_path + 'Sparseness/summary/'

cells = []
vals = []
for e, exp_type in enumerate(['FS', 'PYR', 'SOM']):
    dat_path = startup.data_path + 'Sparseness/%s/pred/' % (exp_type)
    dat_path += str(filt)
    dat_file = dat_path + '/preds.pkl'
    with open(dat_file, 'rb') as infile:
        cell_results = pickle.load(infile)
    print cell_results.keys()

    for cell in cell_results.keys():
        print 'doing ', cell
        cell_vals = []
        for k in sorted(cell_results[cell]):
            for cmb in sorted(cell_results[cell][k]):
                cell_vals.append(cell_results[cell][k][cmb]
                             [lags[exp_type]]['crr_pred'])
        cell_vals = np.array(cell_vals)
        cell_vals /= cell_vals.max()
        cell_vals = np.nan_to_num(cell_vals)
        vals.append(cell_vals)
        cells.append([e, exp_type, cell])

vals = np.array(vals)
cells = np.array(cells)
y = cells[:, 0]
idx = np.arange(len(y))
np.random.shuffle(idx)
y = y[idx]
vals =vals[idx]

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as clf

print cross_val_score(clf(), vals, y, verbose=True)

##
#
pca = PCA(n_components=2)
pca.fit(vals)
print(pca.explained_variance_ratio_)
vals2 = pca.transform(vals)

fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
plt.hold(True)
for c in range(len(vals2)):
    if cells[c][1] == 'PYR':
        col = 'r'
    elif cells[c][1] == 'FS':
        col = 'b'
    elif cells[c][1] == 'SOM':
        col = 'g'
    plt.scatter(vals2[c, 0], vals2[c, 1], c=col)
plt.show()

#plt.plot([-1, len(xaxis) + 1], [0, 0], '--')
#plt.xlim(0, offset)
#plt.ylim(-0.05, 1)
#plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.95,
#                   wspace=0.3, hspace=0.34)
#plt.ylabel('Correlation between prediction and Experimental Mean')
#fname = '%s%s_pred' % (fig_path, cell)
#fig.savefig(fname + '.eps')
#fig.savefig(fname + '.png')
##plt.show()
#plt.close(fig)
#
#if not os.path.exists(fig_path):
#    os.makedirs(fig_path)
