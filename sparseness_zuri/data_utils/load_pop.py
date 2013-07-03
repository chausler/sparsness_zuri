import sys
sys.path.append('..')
from startup import *
import scipy.io
from collections import deque
import numpy as np
from data_utils.utils import filter
import movie
import os

scan_freq = 7.81  # scanner sample rate in Hz
mov_freq = 30  # movie frame rate in Hz
baseline = 5  # in seconds
filt_window = 2000.
decay = 0.4

# 5 sec baseline less one frame for shutter start which is already removed
baseline_samples = int((scan_freq * 5)) - 1
maskSizeDeg = 30  # in degrees

ignore_cells = {'120921': [0]}
# movie id for each experiment
exp_mov = {
           '120425': 'movie_43s',
           '120426': 'movie_43s',
           '120920': 'movie_43s',
           '120921': 'movie_43s',
           '121116': 'nc_121115',
           '121122': 'nc_121115',
           '121127': 'nc_121115'
           }


def baseline_filt_old(dat, baselines, flatten=False):
    all_bs = []
    for cell in range(len(dat)):
        cell_bs = []
        for trial in range(dat.shape[2]):
            bs_mn = baselines[cell, :, trial].mean()
            bs_std = np.std(baselines[cell, :, trial])
            if flatten:
                bs = bs_mn + 2 * bs_std
                dat[cell, dat[cell, :, trial] < bs, trial] = 0.
            cell_bs.append([bs_mn, bs_std])
        all_bs.append(cell_bs)
        dat[cell] = filter(dat[cell].T, scan_freq, window=filt_window,
                           prm=decay)[0].T
    return np.array(all_bs)


def baseline_filt(dat, baseline_samples, flatten=False):
    all_bs = []
    all_cell = []
    for cell in range(len(dat)):
        cell_dat = filter(dat[cell].T, scan_freq, window=filt_window,
                           prm=decay)[0].T
        all_cell.append(cell_dat[baseline_samples:])
        all_bs.append(np.array([cell_dat[:baseline_samples].mean(0),
                       np.std(cell_dat[:baseline_samples], 0)]).T)
    return np.array(all_cell), np.array(all_bs)


def list_PopExps():
    pth = extern_data_path + 'Sparseness/PopulationData/Population/'
    return sorted(os.walk(pth).next()[1])


def load_PopData(exp_id, only_active=False):

    #ignore 120201
    dat_dir = (extern_data_path +
            'Sparseness/PopulationData/Population/%s/Analysis/Data/' % exp_id)
    dat = scipy.io.loadmat('%s/%s_dff_movies.mat' % (dat_dir, exp_id))
    rec_dat = scipy.io.loadmat('%s/%s_RF.mat' % (dat_dir, exp_id))
    f_active = '%s/Sparseness/POP/xcorr_active_%s.npy' % (data_path, exp_id)

    cell_idx = np.arange(len(dat['dff_c']))
    if os.path.exists(f_active):
        active = np.load(f_active)
        if only_active:
            cell_idx = (active[:, 1] == 1)
    else:
        active = None

    if 'cellsInRF' in rec_dat:
        rf_cells = np.array(rec_dat['cellsInRF'][0]) - 1
    else:
        rf_cells = []

    dat_c = dat['dff_c'][cell_idx]
    dat_w = dat['dff_w'][cell_idx]
    if exp_id in ignore_cells:
        idx = np.ones(dat_c.shape[0])
        for i in ignore_cells[exp_id]:
            idx[i] = 0
            rf_cells = rf_cells[rf_cells != i]
            rf_cells[rf_cells > i] -= 1
        dat_c = dat_c[idx == 1]
        dat_w = dat_w[idx == 1]

    dat_c, bs_c = baseline_filt(dat_c, baseline_samples)
    dat_w, bs_w = baseline_filt(dat_w, baseline_samples)

    mov_path = data_path + 'Sparseness/POP/'
    mov = scipy.io.loadmat(mov_path + exp_mov[exp_id] + '.mat')
    mov = mov['dat']

    movResolution = np.array(mov.shape[1:])
    vnResolution = np.array(rec_dat['vnResolution'][0])
    #vfSizeDegrees = np.array(rec_dat['vfSizeDegrees'][0])

    maskLocationDeg = np.array(rec_dat['centerRF_actual'][0])
    vfPixelsPerDegree = np.array(rec_dat['vfPixelsPerDegree'][0]).mean()

    # in screen pixels
    maskSizePixel = maskSizeDeg * np.array([vfPixelsPerDegree,
                                            vfPixelsPerDegree])
    #convert to movie pixels
    scale_width = vnResolution[0] / (movResolution[0] * 1.)
    scale_height = vnResolution[1] / (movResolution[1] * 1.)

    #scale to screen with clipping
    if  scale_height < scale_width:
        scale = scale_width
        overshoot = scale_height / scale_width
        # not all of the actual movie is shown anymore
        adjustedMovResolution = movResolution * np.array([1, overshoot])
        adjustedMovResolution = adjustedMovResolution.astype(np.int)
    else:
        scale = scale_height
        overshoot = scale_width / scale_height
        # not all of the actual movie is shown anymore
        adjustedMovResolution = movResolution * np.array([overshoot, 1])
        adjustedMovResolution = adjustedMovResolution.astype(np.int)

    maskSizePixel = (maskSizePixel / scale).astype(np.int)
    if (maskSizePixel[0] % 2) == 0:
        maskSizePixel += 1
    maskLocationPixel = maskLocationDeg * vfPixelsPerDegree / scale
    maskLocationPixel += (adjustedMovResolution / 2.)
    maskLocationPixel = maskLocationPixel.astype(np.int)
    
    ret_dat = {'exp_id': exp_id,
                        'scan_freq': scan_freq,
                        'dat_c': dat_c, 'dat_w': dat_w,
                        'bs_c': bs_c, 'bs_w': bs_w,
                        'active': active,
                        'maskLocationDeg': maskLocationDeg,
                        'maskSizeDeg': maskSizeDeg,
                        'vfPixelsPerDegree': vfPixelsPerDegree,
                        'vnResolution': vnResolution,
                        'movResolution': movResolution,
                        'adjustedMovResolution': adjustedMovResolution,
                        'maskSizePixel': maskSizePixel,
                        'maskLocationPixel': maskLocationPixel,
                        'scale': scale,
                        'movie': exp_mov[exp_id] + '.mat',
                        'rf_cells': rf_cells,
                        'scan_freq': scan_freq
                        
                         }
    return ret_dat

if __name__ == "__main__":
  
#    121122 [12.1, -8.1]
#    121127 [8.4, 2.4]
    exps = list_PopExps()
    for exp in exps:        
        print exp
#        d = load_PopData(exp)
#
#        rf = d['rf_cells']
#        act = d['active'][:,1]
#        act = np.where(act)[0]
#        print rf
#        print act
#        print d['dat_c'].shape[0], len(rf), len(act)
