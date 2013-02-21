import pickle
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot
import pylab as plt
import numpy as np


for exp_type in ['FS', 'PYR', 'SOM']:
    fig_path = startup.fig_path + 'ephys/%s/pred/' % (exp_type)

    with open(fig_path + 'shift.pkl', 'rb') as infile:
        dat = pickle.load(infile)
    print dat
    plt.figure(figsize=(20, 10))
    plt.hold(True)
    a = len(dat)
    b = len(dat.values()[0])
    c = 1
    for aa, k in enumerate(sorted(dat.keys())):
        for bb, cmb in enumerate(sorted(dat[k].keys())):
            cnt = (aa * b) + bb + 1 
            print k, cmb, cnt
            ax = plt.subplot(a, b, cnt)
            if aa == 0:
                plt.title(cmb)

            adjust_spines(ax, ['bottom', 'left'])
            shifts = np.array(dat[k][cmb].keys(), dtype=np.int)
            shifts.sort()
            xaxis = np.arange(shifts.min(), shifts.max() + 1, 10)
            xaxis = np.append(xaxis, np.array([0]))
            xaxis.sort()
            for shift in shifts:
                xs = np.array([shift])
                ys = np.array(dat[k][cmb][str(shift)])
                do_box_plot(ys, xs, 'k', widths=[0.9])
            plt.ylim(-1, 1)
            plt.xlim(shifts.min() - 5, shifts.max() + 5)
            plt.xticks(xaxis)
    plt.subplots_adjust(left=0.03, bottom=0.02, right=0.97, top=0.97,
                       wspace=0.3, hspace=0.34)
    plt.show()
