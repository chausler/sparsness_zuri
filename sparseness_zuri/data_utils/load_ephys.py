import sys
sys.path.append('..')
from startup import *
import scipy.io
from collections import deque
import numpy as np
import os
import Image
from data_utils.utils import filter

cell_data = {  # SOM DATA [Receptive field file, mask size, scale_without_crop]
                '120119_009': ['120119_003_RF', 15, 1],
                '120221_002': ['120221_001_RF', 20, 1],
                '120229_006': ['120229_004_RF', 20, 1],
                '120313_005': ['120313_004_RF', 30, 0],
                '120316_008': ['120316_006_RF', 20, 0],
                '120504_004': ['120504_002_RF', 30, 1],
                '120509_003': ['120509_total_RF', 25, 1],
                '120510_003': ['120510_total_RF', 25, 0],
                '120511_004': ['120511_003_RF', 30, 0],
                '120524_010': ['120524_total_RF', 25, 0],
                '120525_004': ['120525_total_RF', 30, 0],
                '120526_015': ['120526_total_RF', 20, 0],
                '120530_009': ['120530_total_RF', 30, 0],
                '120531_004': ['120531_total_RF', 20, 0],
                '120601_010': ['120601_008_RF', 20, 0],
                '120602_003': ['120602_total_RF', 30, 0],
                '120606_012': ['120606_total_RF', 20, 0],

                #  PYR
                '100610_1': ['100610_008_RF', 25, 0],
                '100610_2': ['100610_012_RF', 35, 0],
                '100610_3': ['100610_015_RF', 25, 0],
                '100611_3': ['100611_010_RF', 15, 0],
                '100611_4': ['100611_013_RF', 25, 0],
                '100611_5': ['100611_020_RF', 35, 0],
                '100611_6': ['100611_024_RF', 35, 0],
                '100622_1': ['100622_006_RF', 25, 0],
                '100622_2': ['100622_011_RF', 25, 0],
                '100622_4': ['100622_019_RF', 15, 0],
                #'100622_5': ['100622_021', None, 0],
                '100623_2': ['100623_007_RF', 25, 0],
                '100623_3': ['100623_011_RF', 25, 0],
                '100623_5': ['100623_018_RF', 15, 0],
                '100623_7': ['100623_026_RF', 25, 0],
                '100623_10': ['100623_039_RF', 25, 0],
                '100624_1': ['100624_003_RF', 25, 0],
                '100624_4': ['100624_017_RF', 25, 0],
                '100902_4': ['100902_007_RF', 25, 0],
                '100922_1': ['100922_001_RF', 25, 0],
                '100928_1': ['100928_005_RF', 15, 0],
                '100928_2': ['100928_006_RF', 25, 0],
                '100930_1': ['100930_001_RF', 25, 0],
                '100930_3': ['100930_007_RF', 35, 0],
                '101008_1': ['101008_001_005_RF', 15, 0],
                '101008_2': ['101008_008_RF', 25, 0],
                '101008_3': ['101008_009_012_RF', 25, 0],
                '101013_3': ['101013_007_RF', 40, 0],
                '101020_1': ['101020_005_RF', 15, 0],

                # FS DATA
                '120412_006': ['120412_003_004_RF', 25, 0],
                '120413_003': ['120413_000_RF', 25, 0],
                '120413_006': ['120413_005_RF', 25, 0],
                '120413_009': ['120413_008_RF', 25, 0],
                '120417_006': ['120417_004_005_RF', 25, 0],
                '120420_008': ['120420_007_RF', 30, 0],
                '120420_012': ['120420_010_011_RF', 30, 0],
                '120420_014': ['120420_013_RF', 25, 0],
            }


#analysis_mask_size_som 
# whether the stimulus was cropped or not
#scale_without_crop_SOM 

def load_movie_data(cellid, exp_type='SOM'):
    dat = np.load(data_path + 'ephys/' + exp_type + '/' + cellid + '_processed.npz',
              'rb')
    return dat


def downsample_four(four, size):
    new_four = []
    for f in four:
        fr = []
        for ff in f:
            ff = Image.fromarray(ff)
            ff = ff.resize([size, size], Image.ANTIALIAS)
            fr.append(np.array(ff))
        new_four.append(fr)
    new_four = np.array(new_four)
    return new_four


def load_parsed_movie_dat(cellid, exp_type='SOM', four_downsample=None):
    mov = load_movie_data(cellid, exp_type)
    #mask movie data
    lum_mask = mov['lum_mask']
    con_mask = mov['con_mask']
    flow_mask = mov['flow_mask']
    four_mask = mov['four_mask']
    freq_mask = mov['freq_mask']
    orient_mask = mov['orient_mask']
    if four_downsample != None:
        four_mask = downsample_four(four_mask, four_downsample)
    four_mask_shape = four_mask.shape[2:]
    four_mask = four_mask.reshape(four_mask.shape[0],
                           four_mask.shape[1], -1)
    #surround movie data
    lum_surr = mov['lum_surr']
    con_surr = mov['con_surr']
    flow_surr = mov['flow_surr']
    four_surr = mov['four_surr']
    freq_surr = mov['freq_surr']
    orient_surr = mov['orient_surr']
    if four_downsample != None:
        four_surr = downsample_four(four_surr, four_downsample)
    four_surr_shape = four_surr.shape[2:]
    four_surr = four_surr.reshape(four_surr.shape[0],
                                  four_surr.shape[1], -1)
    #whole movie data
    lum_whole = mov['lum_whole']
    con_whole = mov['con_whole']
    flow_whole = mov['flow_whole']
    four_whole = mov['four_whole']
    freq_whole = mov['freq_whole']
    orient_whole = mov['orient_whole']
    if four_downsample != None:
        four_whole = downsample_four(four_whole, four_downsample)
    four_whole_shape = four_whole.shape[2:]
    four_whole = four_whole.reshape(four_whole.shape[0],
                                    four_whole.shape[1], -1)
    mov.close()
    return lum_mask, con_mask, flow_mask, four_mask, four_mask_shape,\
            freq_mask, orient_mask,\
            lum_surr, con_surr, flow_surr, four_surr, four_surr_shape,\
            freq_surr, orient_surr,\
            lum_whole, con_whole, flow_whole, four_whole, four_whole_shape,\
            freq_whole, orient_whole


def generate_psth(src):
    if src is None:
        return None
    targ = []
    for t in src:
            psth = np.zeros_like(t)
            spk_times = np.where(t)[0]
            spikes = t[t != 0]
            if len(spikes) > 0:
                isi = np.array([spk_times[0] + 1] +
                               np.diff(spk_times).tolist())
                np.random.shuffle(spikes)
                np.random.shuffle(isi)
                cnt = 0
                for i, j in zip(isi, spikes):
                    cnt += i
                    if cnt >= (len(psth) - 1):
                        cnt -= 1
                    psth[cnt] += j
            targ.append(psth)
    return np.array(targ)


def load_EphysData(exp_type='SOM', filt=0.1):

    #ignore 120201
    mat = scipy.io.loadmat(extern_data_path +
                           'Sparseness/EphysData/analysis/EphysData_%s.mat'
                           % exp_type)
    dat_dir = extern_data_path + 'Sparseness/EphysData/%s/' % exp_type
    mov_path = data_path + 'ephys/%s/' % exp_type
    dat = mat['EphysData_%s' % exp_type]
    print len(dat[0])
    #do_firing_rate([1], 10)
    all_dat = {}
    for dt in dat[0]:
        expdate = str(dt[0][0])
        if exp_type == 'SOM':
            cellid = dt[0][0] + '_' + dt[1][0]
        elif exp_type == 'PYR':
            cellid = dt[0][0] + '_' + str(dt[1][0][0])
        else:
            cellid = dt[1][0]
        print cellid

        if expdate == '120201':
            print 'skipping'
            continue
        if not cell_data.has_key(cellid):
            print 'no mask size found'
            continue
        #files = os.listdir(dat_dir + '/' + expdate + '/Spreadsheets@100Hz/')
        fname = cell_data[cellid][0]
#        for ff in files:
#            if 'RF.mat' in ff:
#                fname = ff
#                break

        rec_dat = scipy.io.loadmat(dat_dir + expdate +
                                   '/Spreadsheets@100Hz/' + fname)
        mov = scipy.io.loadmat(mov_path + expdate + '.mat')
        mov = mov['dat']
        movResolution = np.array(mov.shape[1:])        
        vnResolution = np.array(rec_dat['vnResolution'][0])
        #vfSizeDegrees = np.array(rec_dat['vfSizeDegrees'][0])

        maskLocationDeg = np.array(rec_dat['maskLocationDeg'][0])
        vfPixelsPerDegree = np.array(rec_dat['vfPixelsPerDegree'][0]).mean()
        maskSizeDeg = cell_data[cellid][1]

        # in screen pixels
        maskSizePixel = maskSizeDeg * np.array([vfPixelsPerDegree,
                                                vfPixelsPerDegree])
        #convert to movie pixels
        scale_width = vnResolution[0] / (movResolution[0] * 1.)
        scale_height = vnResolution[1] / (movResolution[1] * 1.)

        ## check this
        if cell_data[cellid][2] == 1:
            # scale to screen, no clipping
            if  scale_height < scale_width:
                scale = scale_height
            else:
                scale = scale_width
            maskSizePixel = (maskSizePixel / scale).astype(np.int)
            maskLocationPixel = maskLocationDeg * vfPixelsPerDegree / scale
            maskLocationPixel += (movResolution / 2.)
            maskLocationPixel = maskLocationPixel.astype(np.int)
            adjustedMovResolution = movResolution

        else:
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

        bin_freq = dt[5][0][0]
        movie_duration = dt[6][0][0]
        if exp_type == 'PYR':
            psth_c = np.array(dt[6], dtype=np.int)
            psth_w = np.array(dt[7], dtype=np.int)
            psth_s = None
        else:
            psth_c = np.array(dt[7], dtype=np.int)
            psth_w = np.array(dt[8], dtype=np.int)
            psth_s = np.array(dt[9], dtype=np.int)

        # do shift
        psth_c_shift = []
        psth_w_shift = []
        psth_s_shift = []
        #shifts = np.arange(-15, 16, 5).tolist()
        shifts = range(-10, 1)
#        shifts = np.array(list(set(shifts)))
#        shifts.sort()

        for shift in shifts:
            idx = deque(np.arange(psth_c.shape[1]))
            idx.rotate(shift)
            c, _ = filter(psth_c[:, idx], bin_freq, prm=filt)
            psth_c_shift.append(c)
            w, _ = filter(psth_w[:, idx], bin_freq, prm=filt)
            psth_w_shift.append(w)
            if psth_s is not None:
                s, _ = filter(psth_s[:, idx], bin_freq, prm=filt)
                psth_s_shift.append(s)
        psth_c_shift = np.array(psth_c_shift)
        psth_w_shift = np.array(psth_w_shift)
        psth_s_shift = np.array(psth_s_shift)

        idx = deque(np.arange(psth_c.shape[1]))
        idx.rotate(-4)
        psth_c = psth_c[:, idx]
        psth_w = psth_w[:, idx]
        if psth_s is not None:
            psth_s = psth_s[:, idx]

        # do generate
        psth_c_gen = generate_psth(psth_c)
        psth_w_gen = generate_psth(psth_w)
        psth_s_gen = generate_psth(psth_s)
        psth_c_gen, _ = filter(psth_c_gen, bin_freq, prm=filt)
        psth_w_gen, _ = filter(psth_w_gen, bin_freq, prm=filt)
        psth_s_gen, _ = filter(psth_s_gen, bin_freq, prm=filt)

        psth_c, edge = filter(psth_c, bin_freq, prm=filt)
        psth_w, _ = filter(psth_w, bin_freq, prm=filt)
        psth_s, _ = filter(psth_s, bin_freq, prm=filt)

        # random version
        idx = np.arange(psth_c.shape[1])
        np.random.shuffle(idx)
        psth_c_rand = psth_c[:, idx]
        psth_w_rand = psth_w[:, idx]
        if psth_s is not None:
            psth_s_rand = psth_s[:, idx]
        else:
            psth_s_rand = None
#           idx = np.arange(y.shape[1])
#                        if randomise == 'shift':
#                            idx = deque(idx)
#                            idx.rotate(-23)
#                        elif randomise == 'random':
#                            np.random.shuffle(idx)
#                        y = y[:, idx]
        all_dat[cellid] = {'expdate': expdate, 'cellid': cellid,
                            #'matfile': matfile,
                            #'conditions': conditions,
                            #'conditions_used': conditions_used,
                            'bin_freq': bin_freq,
                            'movie_duration': movie_duration,
                            'psth_c': psth_c, 'psth_w': psth_w,
                            'psth_s': psth_s,
                            'psth_c_rand': psth_c_rand,
                            'psth_w_rand': psth_w_rand,
                            'psth_s_rand': psth_s_rand,
                            'psth_c_gen': psth_c_gen,
                            'psth_w_gen': psth_w_gen,
                            'psth_s_gen': psth_s_gen,
                            'psth_c_shift': psth_c_shift,
                            'psth_w_shift': psth_w_shift,
                            'psth_s_shift': psth_s_shift,
                            'shifts': shifts,
                            'maskLocationDeg': maskLocationDeg,
                            'maskSizeDeg': maskSizeDeg,
                            #'vfSizeDegrees': vfSizeDegrees,
                            'vfPixelsPerDegree': vfPixelsPerDegree,
                            'vnResolution': vnResolution,
                            'movResolution': movResolution,
                            'adjustedMovResolution': adjustedMovResolution,
                            'maskSizePixel': maskSizePixel,
                            'maskLocationPixel': maskLocationPixel,
                            'scale': scale,
                            'edge': edge
                             }
    return all_dat

if __name__ == "__main__":
    exp_type = 'FS'
    dat = load_EphysData(exp_type)
    for e in dat.values():
        cellid = e['cellid']
        mov = load_movie_data(cellid, exp_type)
        load_parsed_movie_dat(cellid, exp_type)
        four_mask = mov['four_mask']
        four_whole = mov['four_whole']
        print four_mask.shape, four_whole.shape
#        import pylab as plt
#        plt.figure()
#        plt.subplot(121)
#        plt.hist(np.log(four_mask.ravel()), 100)
#        plt.subplot(122)
#        plt.hist(four_whole.ravel(), 100)
#        plt.show()
