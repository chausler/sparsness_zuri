import sys
sys.path.append('..')
from startup import *
import scipy.io
import numpy as np
import os
import Image

# mask sizes of each of the cells for analysis
analysis_mask_size_som ={'120119': 15,
                        '120221': 20,
                        '120229': 20,
                        '120313': 30,
                        '120316': 20,
                        '120504': 30,
                        '120509': 25,
                        '120510': 25,
                        '120511': 30,
                        '120524': 25,
                        '120525': 30,
                        '120526': 20,
                        '120530': 30,
                        '120531': 20,
                        '120601': 20,
                        '120602': 30,
                        '120606': 20,
                        }

# whether the stimulus was cropped or not
scale_without_crop_SOM = {'120119': 1,
                    '120221': 1,
                    '120229': 1,
                    '120313': 1,
                    '120316': 0,
                    '120504': 0,
                    '120509': 1,
                    '120510': 1,
                    '120511': 0,
                    '120524': 0,
                    '120525': 0,
                    '120526': 0,
                    '120530': 0,
                    '120531': 0,
                    '120601': 0,
                    '120602': 0,
                    '120606': 0
                    }


def load_movie_data(expdate, exp_type='SOM'):
    if exp_type == 'SOM':
        dat = np.load(data_path + 'ephys/som/' + expdate + '_processed.npz',
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


def load_parsed_movie_dat(expdate, exp_type='SOM', four_downsample=None):
    mov = load_movie_data(expdate, exp_type)
    #mask movie data
    lum_mask = mov['lum_mask']
    con_mask = mov['con_mask']
    flow_mask = mov['flow_mask']
    four_mask = mov['four_mask']
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
    if four_downsample != None:
        four_whole = downsample_four(four_whole, four_downsample)
    four_whole_shape = four_whole.shape[2:]
    four_whole = four_whole.reshape(four_whole.shape[0],
                                    four_whole.shape[1], -1)

    return lum_mask, con_mask, flow_mask, four_mask, four_mask_shape,\
            lum_surr, con_surr, flow_surr, four_surr, four_surr_shape,\
            lum_whole, con_whole, flow_whole, four_whole, four_whole_shape


def load_EphysData_SOM():

    #ignore 120201
    mat = scipy.io.loadmat(extern_data_path +
                           'Sparseness/EphysData/analysis/EphysData_SOM.mat')
    som_dir = extern_data_path + 'Sparseness/EphysData/SOM/'
    mov_path = data_path + 'ephys/som/'
    dat = mat['EphysData_SOM']

    all_dat = {}
    for dt in dat[0]:
        expdate = str(dt[0][0])
        if expdate == '120201':
            continue
        if not analysis_mask_size_som.has_key(expdate):
            print 'no mask size found'
            continue
        files = os.listdir(som_dir + '/' + expdate + '/Spreadsheets@100Hz/')
        fname = None
        for ff in files:
            if 'RF.mat' in ff:
                fname = ff
                break
        print fname

        som_dat = scipy.io.loadmat(som_dir + '/' + expdate +
                                   '/Spreadsheets@100Hz/' + fname)
        mov = scipy.io.loadmat(mov_path + expdate + '.mat')
        mov = mov['dat']
        movResolution = np.array(mov.shape[1:])
        vnResolution = np.array(som_dat['vnResolution'][0])
        vfSizeDegrees = np.array(som_dat['vfSizeDegrees'][0])

        maskLocationDeg = np.array(som_dat['maskLocationDeg'][0])
        vfPixelsPerDegree = np.array(som_dat['vfPixelsPerDegree'][0]).mean()
        maskSizeDeg = analysis_mask_size_som[expdate]

        # in screen pixels
        maskSizePixel = maskSizeDeg * np.array([vfPixelsPerDegree,
                                                vfPixelsPerDegree])
        #convert to movie pixels
        scale_width = vnResolution[0] / (movResolution[0] * 1.)
        scale_height = vnResolution[1] / (movResolution[1] * 1.)

        ## check this
        if scale_without_crop_SOM[expdate] == 1:
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
            maskLocationPixel = maskLocationDeg * vfPixelsPerDegree / scale
            maskLocationPixel += (adjustedMovResolution / 2.)
            maskLocationPixel = maskLocationPixel.astype(np.int)

        cellid = dt[1][0]
        matfile = dt[2][0]
        conditions = []
        for c in dt[3][0]:
            conditions.append(str(c[0]))
        conditions_used = []
        for c in dt[4][0]:
            val = str(c[0])
            if 'Inverse' in val:
                val = 'Surround'
            elif 'Deg Mask' in val:
                val = 'RF'
            elif 'Whole' in val:
                val = 'WF'
            conditions_used.append(val)

        bin_freq = dt[5][0][0]
        movie_duration = dt[6][0][0]
        psth_c = np.array(dt[7], dtype=np.int)
        psth_w = np.array(dt[8], dtype=np.int)
        psth_s = np.array(dt[9], dtype=np.int)

        all_dat[expdate] = {'expdate': expdate, 'cellid': cellid,
                            'matfile': matfile, 'conditions': conditions,
                            'conditions_used': conditions_used,
                            'bin_freq': bin_freq,
                            'movie_duration': movie_duration,
                            'psth_c': psth_c, 'psth_w': psth_w,
                            'psth_s': psth_s,
                            'maskLocationDeg': maskLocationDeg,
                            'maskSizeDeg': maskSizeDeg,
                            'vfSizeDegrees': vfSizeDegrees,
                            'vfPixelsPerDegree': vfPixelsPerDegree,
                            'vnResolution': vnResolution,
                            'movResolution': movResolution,
                            'adjustedMovResolution': adjustedMovResolution,
                            'maskSizePixel': maskSizePixel,
                            'maskLocationPixel': maskLocationPixel,
                            'scale': scale
                             }
    return all_dat

if __name__ == "__main__":
    load_EphysData_SOM()
