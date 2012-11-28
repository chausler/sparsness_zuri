import sys
sys.path.append('..')
from startup import *
import scipy.io
import numpy as np
import os


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


def load_movie_data(expdate):
    dat = np.load(data_path + 'ephys/som_movies/' + expdate + '_processed.npz', 
                  'rb')
    return dat


def load_EphysData_SOM():

    #ignore 120201
    mat = scipy.io.loadmat(extern_data_path +
                           'Sparseness/EphysData/analysis/EphysData_SOM.mat')
    som_dir = extern_data_path + 'Sparseness/EphysData/SOM/'
    mov_path = data_path + 'ephys/som_movies/'
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
        psth_c = np.array(dt[7])
        psth_w = np.array(dt[8])
        psth_s = np.array(dt[9])

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
