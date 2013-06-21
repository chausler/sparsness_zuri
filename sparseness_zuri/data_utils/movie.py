import numpy as np
from startup import *
import Image


def load_movie_data(cellid, exp_type='SOM'):
    
    dat = np.load(data_path + 'Sparseness/' + exp_type + '/' + cellid +
                      '_processed.npz', 'rb')
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
    if 'lum_surr' in mov:
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
    else:
        lum_surr = None
        con_surr = None
        flow_surr = None
        four_surr = None
        freq_surr = None
        orient_surr = None
        four_surr_shape = None
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


if __name__ == "__main__":
    for exp_type in ['SOM', 'FS', 'PYR']:
        from data_utils.load_ephys import load_EphysData
        import Image
        import pylab as plt
        size = 15
        ephys = load_EphysData(exp_type)
        for e in ephys.values():
            cellid = e['cellid']
            mov = load_movie_data(cellid, exp_type)
            mov = mov['masked'][0]
            print mov.shape
            for i in range(5):
                plt.subplot(2, 5, i + 1)
                plt.imshow(mov[i*10], cmap=plt.cm.gray)
                plt.subplot(2, 5, i + 6)
                xx = mov[i*10]
                xx = np.array(Image.fromarray(xx).resize([size, size]).getdata()).reshape([size, size])
                plt.imshow(xx, cmap=plt.cm.gray)
            plt.show()
