import os
import startup
import colors
from plotting.utils import adjust_spines, do_box_plot, do_spot_scatter_plot, plot_mean_std
import numpy.fft as fft
import pylab as plt
import numpy as np
from colors import CoolWarm
# Some Colours for plotting
clr1 = '0.3'
clr2 = '0.9'
clr3 = np.array([255, 151, 115]) / 255.
clr4 = np.array([200, 105, 75]) / 255.


def reverse_fft(four_weights):
        A = fft.ifft2(four_weights)
        return np.sqrt(A.real ** 2 + A.imag ** 2)


def plot_four(four_weights, ylims, plt_num=[1, 1, 1], title=None,
              ax_off=False, interpolation='None'):
    """plot the weights of the spatial fourier """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    ext = four_weights.shape[0] / 2.
    _ = plt.imshow(four_weights.T, cmap=colors.CoolWarm,
                    interpolation=interpolation,
               extent=(-ext, ext, -ext, ext))
    plt.clim(ylims)
    if ax_off:
        adjust_spines(ax, [])
    else:
        adjust_spines(ax, ['bottom', 'left'])
    if title is None:
        title = 'Filter'
    plt.title(title)
    plt.colorbar()


def plot_preds_freq(weights, plt_num=[1, 1, 1]):
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.scatter(range(len(weights)), weights)
    adjust_spines(ax, ['bottom', 'left'])


def plot_preds_orient(weights, plt_num=[1, 1, 1]):
    orients = ['Mean Horiz', 'Mean Vert', 'Mean Diag', 'Mean Diag 2']
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    x = 7
    mid = x / 2
    vals_hor = np.zeros([x, x])
    vals_ver = np.zeros([x, x])
    vals_diag = np.zeros([x, x])    
    delta = (mid / 2)
    vals_hor[mid, [mid - delta, mid + delta]] = 1
    vals_ver[[mid - delta, mid + delta], mid] = 1
    x, y = np.diag_indices_from(vals_diag)
    vals_diag[x[mid - delta], y[mid - delta]] = 1
    vals_diag[x[mid + delta], y[mid + delta]] = 1
    vals_diag2 = np.array(zip(*vals_diag[::-1])) 
    
#    delta = mid - mid / 2
#    vals_hor[mid, [mid - delta, mid + delta]] = 1
#    vals_ver[:, mid] = 1
#    np.fill_diagonal(vals_diag, 1)
#    np.fill_diagonal(vals_diag2, 1)
#    vals_diag2 = np.array(zip(*vals_diag2[::-1]))
    
    dat = (vals_hor * weights[0] + vals_ver * weights[1] + vals_diag
           * weights[2] + vals_diag2 * weights[3])
    dat = reverse_fft(dat)
    ext = mid
    _ = plt.imshow(dat, cmap=colors.CoolWarm,
                    interpolation='bilinear',
               extent=(-ext, ext, -ext, ext))
    ylims = np.array([0, np.abs(dat).max()])
    plt.clim(ylims)
    plt.colorbar()
    
    adjust_spines(ax, [])



def plot_cell_summary(exp_type, cell, cell_mx):
    fig = plt.figure(figsize=(14, 8))
    rows = 1
    for k in sorted(cell_mx):
        if (('Fourier' in cell_mx[k][1]) or ('Orientation' in cell_mx[k][1])
            or ('Frequency' in cell_mx[k][1])):
            rows = 2
    cnt = 1
    for k in sorted(cell_mx):
        dat = cell_mx[k][2]
        plot_prediction(dat['pred'],dat['mn'],
                '%s  Time: %d, %s, Corr: %.2f' % (k, cell_mx[k][0],
                                    cell_mx[k][1], dat['crr_pred']),
                                    [rows, 2, cnt], legend=(cnt == 1))
        if rows > 1:
            all_weights = None
            plot_params = dat['plot_params']
            if 'Fourier' in cell_mx[k][1]:
                weights = dat['coefs']
                div = (weights.shape[1] / 2)
                weights = weights[:, :div] + weights[:, div:] * 1j
                mn_weight = None
                cntt = 0
                for w in weights:
                    cntt += 1
                    w = w.reshape([np.sqrt(w.shape[0]), np.sqrt(w.shape[0])])
                    w = reverse_fft(w)
#                    plt.subplot(4, 4, cntt)
#                    plt.imshow(w, cmap=CoolWarm)
#                    plt.colorbar()
#                    plt.title(cntt)
                    if mn_weight is None:
                        mn_weight = w
                    else:
                        mn_weight += w

                mn_weight = mn_weight / cntt
#                plt.subplot(4, 4, cntt + 1)
#                plt.imshow(mn_weight, cmap=CoolWarm)
#                plt.colorbar()
#                plt.show()
                plot_four(mn_weight, [0, np.abs(mn_weight).max()],
                          [rows, 2, cnt + 2],
                              interpolation='bilinear')
#                print weights.shape
#                print 'FIX THIS: TAKE THE AVERAGE OF THE FILTERS AFTER THE FFT REVERSAL'
#                assert False
#                inds = [0, 0]
#                for f in plot_params['idx_four']:
#                    four_weights = weights[f].reshape(plot_params['four_shape'][0])
#                    ff = reverse_fft(four_weights)
#                    if all_weights is None:
#                        nx = int(np.sqrt(len(plot_params['idx_four'])))
#                        x = plot_params['four_shape'][0][0]
#                        xx = x * nx
#                        all_weights = np.zeros([xx, xx])
#                    all_weights[inds[0]: inds[0] + x,
#                                inds[1]: inds[1] + x] = ff
#                    inds[1] += x
#                    if inds[1] == xx:
#                        inds[0] += x
#                        inds[1] = 0
#                    plot_four(all_weights, [0, np.abs(all_weights).max()],
#                          [rows, 2, cnt + 2],
#                              interpolation='bilinear')
            elif 'Orientation' in cell_mx[k][1]:
                weights = dat['coefs'].mean(0)
                plot_preds_orient(weights, [rows, 2, cnt + 2])
            elif'Frequency' in cell_mx[k][1]:
                weights = dat['coefs'].mean(0)
                plot_preds_freq(weights, [rows, 2, cnt + 2])
        cnt += 1
    fig_path = startup.fig_path + 'Sparseness/%s/pred/best/' % (exp_type)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fname = '%s%s' % (fig_path, cell)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def plot_prediction(pred, actual, title, plt_num=[1, 1, 1], legend=True):
    """ plot the prediction of the model vs the mean of the actual
     experiment """
    ax = plt.subplot(plt_num[0], plt_num[1], plt_num[2])
    plt.hold(True)
    #plt.fill_between(range(len(pred)), actual, pred, facecolor=clr3)
    pred = (pred - pred.mean()) / np.std(pred)
    actual = (actual - actual.mean()) / np.std(actual)
    plt.plot(range(len(pred)), pred, color=clr4, linewidth=2,
             label='Predicted')
    plt.plot(range(len(pred)), actual, color=clr1,
             label='Mean Experimental', linewidth=2)
    if legend:
        plt.legend(bbox_to_anchor=(0.55, 0.9, 0.4, 0.1), frameon=False)
    adjust_spines(ax, ['bottom', 'left'])
    plt.title(title)
    