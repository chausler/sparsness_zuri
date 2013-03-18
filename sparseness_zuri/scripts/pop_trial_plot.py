import sys
sys.path.append('..')
import numpy as np
import pylab as plt
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
from data_utils.load_pop import load_PopData, list_PopExps
import os
from data_utils.utils import filter

#decay time of 0.3
#voegelstein approch
#ap kernel filter - matlab toolbox. documents/software/matlabscripts
# or 2*std pver baselin, and stays above for a reasonable period
# try setting everything under the 2*std to zero
# what makes correlation switch - center vs whole. ? contrast? - sliding window correlation of network.. only active cells
# rau paper 98
# look if best predictor is stable over time
#  look at pairwise correlations btween predictor data 
# look at prediction relationship between whole and center
# classifying cell types
# look at centre_whole. we would expect the reliability to go up
# decorrelation!!!
# can we cluster cells based on their prediction powers.. does this lead to cell types?


exps = list_PopExps()
for exp in exps:
    print exp
    dat = load_PopData(exp)
    dat_c = dat['dat_c']
    print dat_c.shape
    dat_w = dat['dat_w']
    active = dat['active']
    d = np.where(active[:, 1])[0]
    print d
    for cell in range(len(dat_w)):
        print active[cell]
        dtc = dat_c[cell]
        dtw = dat_w[cell]
        
        fig = plt.figure(figsize=(14, 8))
        fig.set_facecolor('white')

        ax = plt.subplot(211)
        plt.plot(dtc, '0.8')
        plt.plot(dtc.mean(1), '0.4', linewidth=2)
        plt.xlim(0, dat_c.shape[1])
        adjust_spines(ax, ['bottom', 'left'])
        plt.title('Mask')

        ax = plt.subplot(212)
        plt.plot(dtw, '0.8')
        plt.plot(dtw.mean(1), '0.4', linewidth=2)
        plt.xlim(0, dat_w.shape[1])
        adjust_spines(ax, ['bottom', 'left'])
        plt.title('Whole')

        act = None
        if active is not None:
            act = active[cell, 1]

        plt.suptitle('Cell: %d, Active: %s, Trials: %d' % (cell, act == 1,
                                                           dat_c.shape[2]))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9,
                                wspace=0.2, hspace=0.25)
        plt.show()
for e in exps:
    print e

 