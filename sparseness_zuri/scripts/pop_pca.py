
import sys
sys.path.append('..')
sys.path = ['/home/chris/programs/scikit-learn'] + sys.path
from sklearn.decomposition import PCA
from startup import *
import numpy as np
import pylab as plt
from data_utils.utils import pairwise_corr
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
import cPickle
from data_utils.tsne import calc_tsne

target = data_path + 'Sparseness/POP/clustering/tsne/'
if not os.path.exists(target):
    os.makedirs(target)
pca_dims = 30
out_dims = 2

exps = list_PopExps()
for exp in exps:
    out_fname = exp + '.pkl'
    dat = load_PopData(exp)
    active = dat['active']
    d = np.where(active[:, 1])[0]
    dat_c = dat['dat_c']
    dat_w = dat['dat_w']
    comb_dat = np.append(dat_c, dat_w, 2)
    comb_shp = comb_dat.shape
    comb_dat = comb_dat.reshape(dat_c.shape[0], -1)
    div = np.array(dat_c.shape[1:]).prod()
#    dat_c = pca.transform(dat_c.reshape(dat_c_shp[0], -1).T).T#.reshape(2, dat_c_shp[1], dat_c_shp[2])
#    dat_w = pca.transform(dat_w.reshape(dat_w_shp[0], -1).T).T#.reshape(2, dat_w_shp[1], dat_w_shp[2])
#    pca = PCA(n_components=2)
#    pca.fit(comb_dat.T)
#    dat_c_shp = dat_c.shape
#    dat_w_shp = dat_w.shape
#    print dat_c.shape, dat_w.shape
    a = calc_tsne(comb_dat.T, NO_DIMS=out_dims, INITIAL_DIMS=pca_dims).T
    print a.shape, div
    dat_c = a[:, :div]
    dat_w = a[:, div:]

    outfile = open(out_fname, 'wb')
    cPickle.dump(a, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    outfile.close()
    
    plt.figure()
    plt.hold(True)
    plt.scatter(dat_c[0], dat_c[1], c='r')
    plt.scatter(dat_w[0], dat_w[1], c='b')
    plt.show()
    
    