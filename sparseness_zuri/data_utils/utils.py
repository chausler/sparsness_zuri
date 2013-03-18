from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import r2_score
from itertools import combinations
from collections import deque

def do_thresh_corr(x, y, threshold=0.05, corr_type=None):

    if corr_type == 'pearsonr':
        c, p = pearsonr(x, y)
    elif corr_type == 'spearmanr':
        c, p = spearmanr(x, y)
    else:
        c = np.maximum(r2_score(x, y), 0)
        return c
    if p > threshold or np.isnan(c):
        c = 0.

    return c


def gauss_filter(dat, bin_freq, window=300, sigma=100):
    """
        turn psth into firing rate estimate. window size is in ms
    """
    if dat is None:
        return None, None
    window = np.int(1. / bin_freq * window)
    sigma = np.int(1. / bin_freq * sigma)
    r = range(-int(window / 2), int(window / 2) + 1)
    gaus = [1 / (sigma * np.sqrt(2 * np.pi)) *
            np.exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]
    if len(dat.shape) > 1:
        fr = np.zeros_like(dat, dtype=np.float)
        for d in range(len(dat)):
            fr[d] = np.convolve(dat[d], gaus, 'same')
    else:
        fr = np.convolve(dat, gaus, 'same')
#    import pylab as plt
#    print bin_freq
#    plt.subplot(311)
#    plt.plot(gaus)
#    plt.subplot(312)
#    plt.plot(dat[:5].T)
#    plt.subplot(313)
#    plt.plot(fr[:5].T)
#    plt.show()

    return fr, len(gaus) / 2


def filter(dat, bin_freq, type='exp', window=300, prm=0.2):
    """
        turn psth into firing rate estimate. window size is in ms
    """
    if dat is None:
        return None, None
    window = np.int(window / (1. / bin_freq * 1000))
    r = np.arange(-window, window + 1)
    kern = np.exp(r * -prm)
    kern[: len(kern) / 2] = 0
    kern = kern / kern.sum()
    if len(dat.shape) > 1:
        fr = np.zeros_like(dat, dtype=np.float)
        for d in range(len(dat)):
            fr[d] = np.convolve(dat[d], kern, 'same')
    else:
        fr = np.convolve(dat, kern, 'same')
#    import pylab as plt
#    plt.subplot(311)
#    plt.plot(kern)
#    plt.subplot(312)
#    plt.plot(dat[:5].T)
#    plt.subplot(313)
#    plt.plot(fr[:5].T)
#    plt.show()

    return fr, len(kern) / 2


def corr_trial_to_trial(trials, shift):

    samples = trials.shape[1]
    norm_idx = np.arange(samples)
    shift_idx = deque(norm_idx)
    shift_idx.rotate(shift)
    shift_idx = np.array(shift_idx)
    if shift != 0:
        blip = (shift_idx != 0) * (shift_idx != (samples - 1))
        shift_idx = shift_idx[blip]
        norm_idx = norm_idx[blip]

    xcorr = []
    for i, j in combinations(range(len(trials)), 2):
        xcorr.append(do_thresh_corr(trials[i, norm_idx], trials[j, shift_idx], 'spearmanr'))
    crr = np.array(xcorr).mean()
    return crr


def corr_trial_to_mean(trials, mn, edge=None):
    if mn is None:
        mn = trials.mean(0)
    xcorr = []
    for t in trials:
        if edge is None:
            xcorr.append(do_thresh_corr(t, mn))
        else:
            xcorr.append(do_thresh_corr(t[edge: -edge], mn[edge: -edge]))
    crr = np.array(xcorr).mean()
    return crr


if __name__ == "__main__":
    tmp = np.zeros([1, 30])
    tmp[0, 10] = 1
    filter(tmp, 30, prm=1)