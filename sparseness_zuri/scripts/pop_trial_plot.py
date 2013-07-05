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
# look at average prediction of one neuron for all others over time as a measure of correlation/decorrelation
# look at correlation within trials between neurons.
filter = ['120425']
exps = list_PopExps()
for exp in exps:
    if len(filter) > 0 and exp not in filter:
        print exp, ' not in filter, skipping'
        continue
    print exp
    dat = load_PopData(exp)
    dat_c = dat['dat_c']
    print dat_c.shape
    dat_w = dat['dat_w']
    active = dat['active']
    d = np.where(active[:, 1])[0]
    print d

    trl_w = dat_w.mean(0)
    trl_c = dat_c.mean(0)

    fig = plt.figure(figsize=(14, 8))
    fig.set_facecolor('white')

    ax = plt.subplot(211)
    plt.hold(True)
    plt.plot(trl_c)
    plt.plot(trl_c.mean(1), '0.4', linewidth=4)
    plt.xlim(0, trl_c.shape[0])
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Mask')

    ax = plt.subplot(212)
    plt.hold(True)
    plt.plot(trl_w)
    plt.plot(trl_w.mean(1), '0.4', linewidth=4)
    plt.xlim(0, trl_w.shape[0])
    adjust_spines(ax, ['bottom', 'left'])
    plt.title('Whole')
    plt.show()


    for cell in range(len(dat_w)):
        print active[cell]
        dtc = dat_c[cell]
        dtw = dat_w[cell]
        fig = plt.figure(figsize=(14, 8))
        fig.set_facecolor('white')

        ax = plt.subplot(211)
        plt.hold(True)
#        plt.plot(dtc)
        plt.plot(dtc.mean(1), '0.4', linewidth=2)
        plt.xlim(0, dat_c.shape[1])
        adjust_spines(ax, ['bottom', 'left'])
        plt.title('Mask')

        ax = plt.subplot(212)
        plt.hold(True)
        plt.plot(dtw)
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

 