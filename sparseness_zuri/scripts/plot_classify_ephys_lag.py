import pickle
import startup
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, plot_mean_std
import pylab as plt
import numpy as np
from plot_classify_ephys_sub import plot_cell_summary
randomise = None
filt = 0.1
cell_max_time = {}
cell_max_type = {}
cmb_types = {}
stim_types = {}
colors = ['r', 'b', 'g']
style = ['x', 'o', '<']
exp_types = ['FS', 'PYR', 'SOM']
for exp_type in exp_types:
    fig_path = startup.fig_path + 'Sparseness/%s/pred/' % (exp_type)
    dat_path = startup.data_path + 'Sparseness/%s/pred/' % (exp_type)
    if randomise is not None:
        dat_path = dat_path + randomise + '_' + str(filt)
        fig_path = fig_path + randomise + '_' + str(filt)
    else:
        dat_path = dat_path + str(filt)
        fig_path = fig_path

    dat_file = dat_path + '/preds.pkl'
    with open(dat_file, 'rb') as infile:
        cell_results = pickle.load(infile)
    print cell_results.keys()

    # convert data
    dat = {}
    for cell in cell_results.keys():
        for k in cell_results[cell].keys():
            if k not in dat:
                dat[k] = {}
            for cmb in cell_results[cell][k].keys():
                if cmb not in dat[k]:
                    dat[k][cmb] = {}
                for s in cell_results[cell][k][cmb].keys():
                    if s not in dat[k][cmb]:
                        dat[k][cmb][s] = []
                    val = cell_results[cell][k][cmb][s]['crr_pred']
                    dat[k][cmb][s].append(val)

    # do plotting
    fig1 = plt.figure(figsize=(18, 10))
    plt.hold(True)
    fig2 = plt.figure(figsize=(18, 10))
    plt.hold(True)
    a = len(dat)
    b = len(dat.values()[0])
    c = 1
    n = 0
    mx = 0
    axs = []
    crr_sum = {}
    for aa, k in enumerate(sorted(dat.keys())):
        print k
        if k not in stim_types:
            stim_types[k] = None
        for bb, cmb in enumerate(sorted(dat[k].keys())):
            if cmb not in cmb_types:
                cmb_types[cmb] = None
            cnt = (aa * b) + bb + 1 
            print k, cmb, cnt
            title = ''
            if aa == 0:
                title = cmb
            shifts = np.array(dat[k][cmb].keys(), dtype=np.int)
            shifts.sort()
            xaxis = np.arange(shifts.min(), shifts.max() + 1)
            xs = []
            ys = []
            stds = []
            sigs = []
            for shift in shifts:
                xs.append(shift)
                trl = np.array(dat[k][cmb][shift])
                n = len(trl)
                if k not in crr_sum:
                    crr_sum[k] = {}
                if str(shift) in crr_sum[k]:
                    crr_sum[k][str(shift)][0] += len(trl[trl != 0])
                    crr_sum[k][str(shift)][1] += trl.tolist()
                else:
                    crr_sum[k][str(shift)] = [len(trl[trl != 0]),
                                            trl.tolist()]
                trl = trl[trl != 0]
                sigs.append(len(trl))
                if len(trl) == 0:
                    trl = [0]
                ys.append(np.mean(trl))
                stds.append(np.std(trl))
            xs = np.array(xs)
            ys = np.array(ys)
            stds = np.array(stds)
            if (ys + stds).max() > mx:
                mx = (ys + stds).max()
            sigs = np.array(sigs)
            plt.figure(fig1.number)
            axs.append(plot_mean_std(xs, ys, stds, title, [a, b, cnt], legend=False,
                          line='-o'))
            plt.xlim(shifts.min() - 0.5, shifts.max() + 0.5)
            plt.xticks(xaxis)
            plt.figure(fig2.number)
            plot_mean_std(xs, sigs, np.zeros_like(sigs), title, [a, b, cnt], legend=False,
                          line='-o')
            plt.xlim(shifts.min() - 0.5, shifts.max() + 0.5)
            plt.ylim(0, n)
    plt.figure(fig1.number)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.95,
                       wspace=0.23, hspace=0.1)
    mx *= 1.1
    for ax in axs:
        ax.set_ylim(0, mx)

    plt.figure(fig2.number)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.95,
                       wspace=0.23, hspace=0.1)

    fig1.savefig(fig_path + '%.2f_%s_shift.eps' % (filt, exp_type))
    fig1.savefig(fig_path + '%.2f_%s_shift.png' % (filt, exp_type))
    fig2.savefig(fig_path + '%.2f_%s_shift_count.eps' % (filt, exp_type))
    fig2.savefig(fig_path + '%.2f_%s_shift_count.png' % (filt, exp_type))

    plt.close(fig1)
    plt.close(fig2)

    fig3 = plt.figure()
    cnt = 1
    shifts = np.array(crr_sum.values()[0].keys(), dtype=np.int)
    shifts.sort()
    cnt_axs = []
    crr_axs = []
    cnt_lims = [99999, 0]
    crr_lims = [99999, 0]

    for i, k in enumerate(sorted(crr_sum.keys())):
        cnts = []
        crrs = []
        for s in shifts:
            cnts.append(crr_sum[k][str(s)][0])
            tmp = np.array(crr_sum[k][str(s)][1])
            tmp = tmp[tmp!=0]
            #crrs.append([np.mean(tmp), np.std(tmp)])
            crrs.append(tmp)
        crrs = np.array(crrs)
        cnts = np.array(cnts)
        cnt_lims = [np.minimum(cnt_lims[0], cnts.min()),
                    np.maximum(cnt_lims[1], cnts.max())]
#        crr_lims = [np.minimum(crr_lims[0], crrs[:, 0].min()),
#                    np.maximum(crr_lims[1], crrs[:, 0].max() + crrs[:, 1].max())]

        cnt_axs.append(plt.subplot(2, 2, cnt))
        cnt += 2
        plt.title(k)
        plt.plot(shifts, cnts, '-o')
        if i == 0:
            plt.ylabel('# correlated responses')
        crr_axs.append(plt.subplot(2,2, cnt))
        for cr, shift in zip(crrs, shifts):
            do_spot_scatter_plot(cr, shift, 'k', 0.4, False)
        #do_box_plot(crrs, shifts, 'k', np.ones_like(shifts) * 0.5)
        cnt -= 1
        if i == 0:
            plt.ylabel('mean r^2 of responders')
        plt.xlabel('Shift in Frames')
    adjuster = np.array([-0.5, 0.5])
    for ax in cnt_axs:
        ax.set_ylim(cnt_lims + adjuster)
        ax.set_xlim(np.array([shifts.min(), shifts.max()]) + adjuster)
#    for ax in crr_axs:
#        ax.set_ylim(crr_lims + adjuster * 0.05)
#        ax.set_xlim(np.array([shifts.min(), shifts.max()]) + adjuster)
    fig3.savefig(fig_path + '%.2f_%s_shift_avg_count.eps' % (filt, exp_type))
    fig3.savefig(fig_path + '%.2f_%s_shift_avg_count.png' % (filt, exp_type))
    plt.close(fig3)

    shift_max = {}
    for cell in cell_results.keys():
        for k in cell_results[cell].keys():
            for cmb in cell_results[cell][k].keys():
                for s in cell_results[cell][k][cmb].keys():
                    if k not in shift_max:
                        shift_max[k] = {}
                    if s not in shift_max[k]:
                        shift_max[k][s] = {}
                    if cell not in shift_max[k][s]:
                        shift_max[k][s][cell] = []
                    val = cell_results[cell][k][cmb][s]['crr_pred']
                    shift_max[k][s][cell].append(val)

    fig4 = plt.figure(figsize=(14, 8))
    cnt = 1
    shifts = np.array(shift_max.values()[0].keys(), dtype=np.int)
    shifts.sort()
    cnt_axs = []
    crr_axs = []
    cnt_lims = [99999, 0]
    crr_lims = [99999, 0]

    for i, k in enumerate(sorted(shift_max.keys())):
        crrs = []
        # for each shift, get the maximum correlation for each cell
        for s in shifts:
            vals = []
            for cell in shift_max[k][s].keys():
                vals.append(np.array(shift_max[k][s][cell]).max())
            crrs.append(vals)
        crrs = np.array(crrs)
        ax = plt.subplot(1, 2, i + 1)
        plt.title(k)
        for cr, shift in zip(crrs, shifts):
            do_spot_scatter_plot(cr, shift, 'k', 0.5, True)
        #do_box_plot(crrs, shifts, 'k', np.ones_like(shifts) * 0.5)
        if i == 0:
            plt.ylabel('mean r^2 of responders')
        plt.xlabel('Shift in Frames')
        adjuster = np.array([-0.5, 0.5])
        ax.set_ylim(-0.05, 1)
        ax.set_xlim(np.array([shifts.min(), shifts.max()]) + adjuster)
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.95,
                       wspace=0.23, hspace=0.1)
    fig4.savefig(fig_path + '%.2f_%s_shift_max.eps' % (filt, exp_type))
    fig4.savefig(fig_path + '%.2f_%s_shift_max.png' % (filt, exp_type))
    plt.close(fig4)

    cell_max_type[exp_type] = {}
    cell_max_time[exp_type] = {}
    cell_mx = {}
    for cell in cell_results.keys():
        for k in cell_results[cell].keys():
            if k not in cell_max_time[exp_type]:
                cell_max_time[exp_type][k] = []
            if k not in cell_max_type[exp_type]:
                cell_max_type[exp_type][k] = {}
            mx = -1.
            time = None
            clf_type = None
            for cmb in cell_results[cell][k].keys():
                for s in sorted(cell_results[cell][k][cmb].keys())[::-1]:
                    print k, cmb, s, cell
                    val = cell_results[cell][k][cmb][s]['crr_pred']
                    if val > mx:
                        mx = val
                        time = s
                        clf_type = cmb
            cell_max_time[exp_type][k].append([time, mx])
            if clf_type in cell_max_type[exp_type][k]:
                cell_max_type[exp_type][k][clf_type].append(mx)
            else:
                cell_max_type[exp_type][k][clf_type] = [mx]
            cell_mx[k] = [time, clf_type, cell_results[cell][k][clf_type][time]]
        plot_cell_summary(exp_type, cell, cell_mx)
    for k in cell_max_time[exp_type]: 
        cell_max_time[exp_type][k] = np.array(cell_max_time[exp_type][k])


fig5 = plt.figure(figsize=(14, 8))
plt.hold(True)
cnt = 1
for exp_type in exp_types:
    for k in sorted(stim_types):
        ax = plt.subplot(3, 2, cnt)
        plt.scatter(cell_max_time[exp_type][k][:, 0],
                    cell_max_time[exp_type][k][:, 1],
                    c='r', marker='x')
    #                c=colors[i], marker=style[i])
        plt.ylabel('mean r^2 of responders')
        if ax.is_last_row():
            plt.xlabel('Shift in Frames')
            adjust_spines(ax, ['bottom', 'left'])
        else:
            adjust_spines(ax, ['left'])
        if ax.is_first_row():
            plt.title(k)
        if ax.is_first_col():
            ax.text(-0.12, 0.5, exp_type, transform=ax.transAxes,
                    rotation='vertical', va='center', ha='center')
        adjuster = np.array([-0.5, 0.5])
        ax.set_ylim(-0.05, 1)
        ax.set_xlim(np.array([shifts.min(), shifts.max()]) + adjuster)
        cnt += 1

plt.subplots_adjust(left=0.1, bottom=0.06, right=0.97, top=0.95,
                   wspace=0.23, hspace=0.23)
fig_path = startup.fig_path + 'Sparseness/summary/'
fig5.savefig(fig_path + '%.2f_shift_best.eps' % (filt))
fig5.savefig(fig_path + '%.2f_shift_best.png' % (filt))

plt.close(fig5)


fig6 = plt.figure(figsize=(14, 8))
plt.hold(True)
cnt = 1
bins = list(shifts - 0.5) + [0.5]
for exp_type in exp_types:
    for k in sorted(stim_types):
        ax = plt.subplot(3, 2, cnt)
        cnt += 1
        plt.hist(cell_max_time[exp_type][k][:, 0], bins=bins, normed=True)
        if ax.is_last_row():
            plt.xlabel('Shift in Frames')
            adjust_spines(ax, ['bottom', 'left'])
        else:
            adjust_spines(ax, ['left'])
        if ax.is_first_row():
            plt.title(k)
        if ax.is_first_col():
            ax.text(-0.12, 0.5, exp_type, transform=ax.transAxes,
                    rotation='vertical', va='center', ha='center')
            plt.ylabel('# best responders')
        adjuster = np.array([-0.5, 0.5])
        ax.set_xlim(np.array([shifts.min(), shifts.max()]) + adjuster)
        ax.set_ylim(-0.05, 0.5)
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.97, top=0.95,
                   wspace=0.23, hspace=0.23)
fig_path = startup.fig_path + 'Sparseness/summary/'
fig6.savefig(fig_path + '%.2f_shift_best_hist.eps' % (filt))
fig6.savefig(fig_path + '%.2f_shift_best_hist.png' % (filt))
#plt.show()
plt.close(fig6)


fig7 = plt.figure(figsize=(14, 8))
plt.hold(True)
cnt = 1
for exp_type in exp_types:
    for k in sorted(stim_types):
        ax = plt.subplot(3, 2, cnt)
        cnt += 1
        for j, cmb in enumerate(sorted(cmb_types)):
                if cmb in cell_max_type[exp_type][k]:
                    print cmb
                    do_spot_scatter_plot(
                        np.array(cell_max_type[exp_type][k][cmb]), j, 'k',
                        0.4, False)
        if ax.is_last_row():
            adjust_spines(ax, ['bottom', 'left'])
            plt.xlabel('Classifier Type')
            ax.set_xticks(range(len(cmb_types)))
            ax.set_xticklabels(sorted(cmb_types))
            plt.setp(ax.get_xticklabels(), rotation='vertical')
        else:
            adjust_spines(ax, ['left'])
        ax.set_ylim(-0.05, 1)
        ax.set_xlim(-0.5, len(cmb_types) - 0.5)
        if ax.is_first_row():
            plt.title(k)
        if ax.is_first_col():
            ax.text(-0.12, 0.5, exp_type, transform=ax.transAxes,
                    rotation='vertical', va='center', ha='center')
            plt.ylabel('mean r^2 of responders')
plt.subplots_adjust(left=0.06, bottom=0.15, right=0.97, top=0.95,
                   wspace=0.23, hspace=0.23)
fig_path = startup.fig_path + 'Sparseness/summary/'
fig7.savefig(fig_path + '%.2f_cmb_best.eps' % (filt))
fig7.savefig(fig_path + '%.2f_cmb_best.png' % (filt))

plt.close(fig7)


