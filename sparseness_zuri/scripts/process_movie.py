import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy.fft as fft
import numpy as np
import sys
import math
import scipy.io
import scipy.stats as spt
import pylab as plt
sys.path.append('..')
import startup
import os
from plotting.utils import adjust_spines
from data_utils.load_ephys import load_EphysData
from data_utils.load_pop import load_PopData, list_PopExps, scan_freq
from data_utils.utils import downsample, downsample_multi_dim
frame_rate = 30.

def ellipse(width, height):

    horpow = 2
    verpow = horpow
    # to produce a Ellipse with horizontal axis == ceil(2*hor semi axis)
    width = width + 0.5
    #to produce a Ellipse with vertical axis == ceil(2*vert semi axis)
    height = height + 0.5
    [x, y] = np.meshgrid(np.arange(-width, width + 1),
                        np.arange(-height, height + 1))
    #print width, height, x.shape, y.shape
    bll = (abs(x / width) ** horpow + abs(y / height) ** verpow) < 1
    xs = plt.find(bll.sum(0) == 0)
    ys = plt.find(bll.sum(1) == 0)
    bll = bll[ys[0] + 1:ys[1], :]
    bll = bll[:, xs[0] + 1:xs[1]]
    bll = bll.T
    mask_inds = np.where(bll > 0)
    mask_inds = np.array(mask_inds).T
    inv_mask_inds = np.where(bll == 0)
    inv_mask_inds = np.array(inv_mask_inds).T
    box_inds = np.array(np.where(bll != 5555)).T
    box_edges = np.array(
                [[box_inds[:, 0].min(), box_inds[:, 1].min()],
                [box_inds[:, 0].max(), box_inds[:, 1].max()]])
    return mask_inds, inv_mask_inds, box_edges
    #bool = bool(cropcoords(3):cropcoords(4),cropcoords(1):cropcoords(2));


def get_contrast(movie):
    print 'get contrast ...'
    contrast = np.zeros(len(movie))
    for i, frame in enumerate(movie):
        contrast[i] = np.std(frame.ravel())
    return contrast


def get_luminance(movie):
    print 'get luminence ...'
    luminance = np.zeros(len(movie))
    for i, frame in enumerate(movie):
        luminance[i] = frame.mean()
    return luminance


def get_flow(flow, maskSizePixel, maskLocationPixel):
    print 'get flow ...'
    masked, surround = get_masked_data(flow, maskSizePixel, maskLocationPixel)
    masked_sum = []
    whole_sum = []

    for m in masked:
        m = m.reshape(-1, 2)
        [x, y] = spt.nanmean(m, 0)
        r = np.sqrt(x ** 2 + y ** 2)
        theta = math.atan2(y, x)
        masked_sum.append([theta, r])

    for m in flow:
        m = m.reshape(-1, 2)
        [x, y] = spt.nanmean(m, 0)
        r = np.sqrt(x ** 2 + y ** 2)
        theta = math.atan2(y, x)
        whole_sum.append([theta, r])

    surround_sum = []
    for s in surround:
        tmp_sum = []
        for m in s:
            m = m.reshape(-1, 2)
            [x, y] = spt.nanmean(m, 0)
            r = np.sqrt(x ** 2 + y ** 2)
            theta = math.atan2(y, x)
            tmp_sum.append([theta, r])
        surround_sum.append(tmp_sum)

    return np.array(whole_sum)[np.newaxis, :, :],\
         np.array(masked_sum)[np.newaxis, :, :], np.array(surround_sum)

def get_frequencies(dat):
    assert(dat.shape[0] == dat.shape[1])
    mid = (dat.shape[0] / 2)
    end = dat.shape[0]
    freqs = []    
#    plt.figure()
    for i in range(1, end - mid):
        ring = []
        ring.append(dat[mid - i, mid - i: mid + i + 1])
        ring.append(dat[mid + i, mid - i: mid + i + 1])
        ring.append(dat[mid - i: mid + i + 1, mid - i])
        ring.append(dat[mid - i: mid + i + 1, mid + i])
        tmp = np.array(ring)
        ring = [tmp.max(), tmp.mean()]
        freqs.append(ring)
    return np.array(freqs).ravel()

#        datc = dat.copy()
#        datc[mid - i, mid - i: mid + i + 1] = 1
#        datc[mid + i, mid - i: mid + i + 1] = 1
#        datc[mid - i: mid + i + 1, mid - i] = 1
#        datc[mid - i: mid + i + 1, mid + i] = 1
#        plt.subplot(3, 3, i)
#        plt.imshow(datc)
#    plt.show()


def get_orientations(dat):
    assert(dat.shape[0] == dat.shape[1])
    mid = (dat.shape[0] / 2)
    orients = np.array([
                    dat[mid, :].max(),
                    dat[:, mid].max(),
                    dat.diagonal().max(),
                    np.array(zip(*dat[::-1])).diagonal().max(),
                    dat[mid, :].mean(),
                    dat[:, mid].mean(),
                    dat.diagonal().mean(),
                    np.array(zip(*dat[::-1])).diagonal().mean()
                    ])
    return orients


def get_fourier2D(movie, dim_lim=7):
    fouriers = []
    orients = []
    freqs = []
    print 'doing fourier2D'
    for i, frame in enumerate(movie):
        if i % 50 == 0:
            print 'frame %d of %d' % (i, len(movie))
        frame = frame / 255.
        A = fft.fftshift(fft.fft2(frame))
        added = np.sqrt(A.real ** 2 + A.imag ** 2)
        #added *= np.sign(A.imag)
        fouriers.append(added)
    fouriers = np.array(fouriers)
    tmp = fouriers.max(0)
    thresh = tmp.mean() + 4 * np.std(tmp)
    tmp = tmp > thresh
    midx = np.ceil(tmp.shape[0] / 2.)
    midy = np.ceil(tmp.shape[1] / 2.)
    diff_x = dim_lim
    diff_y = dim_lim
    if diff_x < 1:
        diff_x = 1
    if diff_y < 1:
        diff_y = 1
    fouriers = fouriers[:, np.maximum(midx - diff_x, 0):midx + diff_x + 1, :]
    fouriers = fouriers[:, :,  np.maximum(midy - diff_y, 0):midy + diff_y + 1]

    for f in fouriers:
        orients.append(get_orientations(f))
        freqs.append(get_frequencies(f))
    freqs = np.array(freqs)
    orients = np.array(orients)
    # correct for really big values
 #   thresh = fouriers.mean() + 4 * np.std(fouriers)
#    fouriers[fouriers > thresh] = thresh
#    fouriers[fouriers < -thresh] = -thresh
    return fouriers, freqs, orients


def plot_movie(lum, con, flow, four, movie, fname):
    num_frames = lum.shape[1]
    fig = plt.figure(figsize=(16, 7))
    fig.set_facecolor('white')

    ax = plt.subplot(231)
    plt.plot(lum[0])
    plt.xlim(0, num_frames)
    plt.title('Luminence')
    adjust_spines(ax, ['left', 'bottom'])

    ax = plt.subplot(234)
    plt.plot(con[0])
    plt.xlim(0, num_frames)
    plt.title('Contrast')
    adjust_spines(ax, ['left', 'bottom'])

    total_plots = 4
    frames = np.linspace(0, num_frames - 1, total_plots).astype(np.int)
    print 'plotting frames ', frames
    lims = [four[0, frames].min(), four[0, frames].max()]
    for i, _ in enumerate(frames):
        ax = plt.subplot(3, 6, 3 + i, aspect='equal')
        plt.imshow(movie[0, i].T, cmap=plt.cm.gray)
        adjust_spines(ax, [])
        if i == 0:
            plt.title('Movie')

        ax = plt.subplot(3, 6, 9 + i, aspect='equal')
        plt.imshow(four[0, i].T, cmap=plt.cm.gray, interpolation='None')
        plt.clim(lims)
        if i == 0:
            adjust_spines(ax, ['left', 'bottom'])
            plt.title('Spatial Fourier')
            plt.xticks(np.linspace(0, four[0, i].shape[1], 4).astype(np.int))
            plt.yticks(np.linspace(0, four[0, i].shape[0], 4).astype(np.int))
            plt.colorbar()
        else:
            adjust_spines(ax, [])

        if len(flow) > 0:
            ax = plt.subplot(3, 6, 15 + i, polar=True)
            plt.plot([0, flow[0, i, 0]], [0, flow[0, i, 1]], '-o', color='0.3',
                     linewidth=2)
            plt.ylim(0, flow[0, :, 1].max())
            for _, spine in ax.spines.iteritems():
                    spine.set_color('none')  # don't draw spine
            #plt.adjust_spines(ax,[])
            if i != 0:
                plt.gca().set_yticklabels([])
                plt.gca().set_xticklabels([])
            else:
                plt.title('Flow')
            plt.xticks([0, np.pi / 2, np.pi, np.pi * 3. / 2])
            plt.yticks(np.linspace(0, flow[0, :, 1].max(), 4).astype(np.int))

    plt.subplots_adjust(left=0.04, bottom=0.06, right=0.96, top=0.95,
                         wspace=0.20, hspace=0.35)
    #adjust_spines(ax,[])
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def plot_surround(lum, con, flow, four, movie, fname):
    num_frames = lum.shape[1]
    fig = plt.figure(figsize=(16, 9))
    fig.set_facecolor('white')

    parts = lum.shape[0]
    time = np.arange(0, num_frames, 20)
    for s in range(parts):
        ax = plt.subplot(5, parts, s + 1)
        plt.plot(lum[s])
        plt.xlim(0, num_frames)
        if s == 0:
            plt.title('Luminence')
        plt.xticks(time)
        adjust_spines(ax, ['left', 'bottom'])

        ax = plt.subplot(5, parts, parts + (s + 1))
        plt.plot(con[s])
        plt.xlim(0, num_frames)
        if s == 0:
            plt.title('Contrast')
        plt.xticks(time)
        adjust_spines(ax, ['left', 'bottom'])

        frame = num_frames / 2
        lims = [four[s, frame].min(), four[s, frame].max()]

        ax = plt.subplot(5, parts, 2 * parts + (s + 1), aspect='equal')
        plt.imshow(movie[s, frame].T, cmap=plt.cm.gray)
        adjust_spines(ax, [])
        if s == 0:
            plt.title('Movie')

        ax = plt.subplot(5, parts, 3 * parts + (s + 1), aspect='equal')
        plt.imshow(four[s, frame].T, cmap=plt.cm.gray, interpolation='None')
        plt.clim(lims)
        if s == 0:
            adjust_spines(ax, ['left', 'bottom'])
            plt.title('Spatial Fourier')
            tix = np.array(plt.xticks()[0], dtype=np.float)
            plt.xticks(np.linspace(tix.min(), tix.max(), 4).astype(np.int))
            tix = np.array(plt.yticks()[0])
            plt.yticks(np.linspace(tix.min(), tix.max(), 4).astype(np.int))
            #plt.colorbar()
        else:
            adjust_spines(ax, [])
        if len(flow) > 0:
            ax = plt.subplot(5, parts, 4 * parts + (s + 1), polar=True)
            plt.plot([0, flow[s, frame, 0]], [0, flow[s, frame, 1]], '-o',
                     color='0.3', linewidth=2)
            for _, spine in ax.spines.iteritems():
                    spine.set_color('none')  # don't draw spine
            if s != 0:
                plt.gca().set_yticklabels([])
                plt.gca().set_xticklabels([])
            else:
                plt.title('Flow')
            plt.xticks([0, np.pi / 2, np.pi, np.pi * 3. / 2])
            plt.yticks(np.linspace(0, flow[s, frame, 1].max(), 4).astype(np.int))

    plt.subplots_adjust(left=0.04, bottom=0.06, right=0.96, top=0.95,
                         wspace=0.50, hspace=0.5)
    fig.savefig(fname + '.eps')
    fig.savefig(fname + '.png')
    plt.close(fig)


def get_masked_data(data, maskSizePixel, maskLocationPixel, parts=[3, 3]):

    mask_inds, inv_mask_inds, box_edges = ellipse(maskSizePixel[0] / 2,
                                   maskSizePixel[1] / 2)
    box_edges += (maskLocationPixel - maskSizePixel / 2).astype(np.int)
    mask_inds += (maskLocationPixel - maskSizePixel / 2).astype(np.int)
    #masked = np.copy(data)
    surround = np.copy(data)

    dim_x = box_edges[1, 0] - box_edges[0, 0]
    dim_y = box_edges[1, 1] - box_edges[0, 1]
    dims = np.array(data.shape)
    dims[1] = dim_x
    dims[2] = dim_y
    masked = np.zeros(dims)
    for x in range(dim_x):
        for y in range(dim_y):
            xind = box_edges[0, 0] + x
            yind = box_edges[0, 1] + y
            if (xind < 0 or xind >= data.shape[1] or
                yind < 0 or yind >= data.shape[2]):
                continue
            masked[:, x, y] = data[:, xind, yind]


#    masked = masked[:, np.maximum(box_edges[0, 0], 0):box_edges[1, 0] + 1, :]
#    masked = masked[:, :, np.maximum(box_edges[0, 1], 0):box_edges[1, 1] + 1]
    for ind in inv_mask_inds:
        if (ind[0] > 0 and ind[0] < masked.shape[1]
            and ind[1] > 0 and ind[1] < masked.shape[2]):
            masked[:, ind[0], ind[1]] = 0

    for ind in mask_inds:
        if (ind[0] > 0 and ind[0] < surround.shape[1]
            and ind[1] > 0 and ind[1] < surround.shape[2]):
            surround[:, ind[0], ind[1]] = 0

    # split surround
    xs = surround.shape[1] / parts[0]
    ys = surround.shape[2] / parts[1]
    xs = np.arange(0, surround.shape[1] + 1, xs)
    ys = np.arange(0, surround.shape[2] + 1, ys)
    split_surround = None
    for s in surround:
        splits = []
        for ii in range(len(xs) - 1):
            for jj in range(len(ys) - 1):
                tmp = s[xs[ii]:xs[ii + 1]]
                tmp = tmp[:, ys[jj]:ys[jj + 1]]
                splits.append(tmp)
        splits = np.array(splits)[:, np.newaxis, :, :]

        if split_surround is None:
            split_surround = splits
        else:
            split_surround = np.append(split_surround,
                                       splits, 1)
    return masked, split_surround


def process_movie_ephys(exp_type='SOM'):
    failed_count = 0
    success_count = 0
    ephys = load_EphysData(exp_type)
    dat_path = startup.data_path + 'Sparseness/%s/' % exp_type
    fig_path = startup.fig_path + 'Sparseness/%s/movies/' % exp_type

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for e in ephys.values():
        print 'recording ', e['cellid']
        target_fname = e['cellid'] + '_processed.npz'
        flow_fname = e['expdate'] + '_flow.mat'
        if os.path.exists(dat_path + target_fname):
            print 'already exists, skipping ', target_fname
            success_count += 1
#            continue

        if not os.path.exists(dat_path + flow_fname):
            print 'flow data missing, skipping ', flow_fname
            continue

        movie = scipy.io.loadmat(dat_path + e['expdate'] + '.mat')
        movie = movie['dat']
        flow = scipy.io.loadmat(dat_path + flow_fname)
        flow = flow['uv']
        flow = np.swapaxes(flow, 1, 2)
        actual_len = e['psth_c'].shape[1]
        flow = flow[:actual_len]
        movie = movie[:actual_len]

        if (np.abs(e['adjustedMovResolution']
                   - e['movResolution'])).sum() != 0:
            movie = movie[:, :e['adjustedMovResolution'][0],
                      :e['adjustedMovResolution'][1]]
            flow = flow[:, :e['adjustedMovResolution'][0],
                      :e['adjustedMovResolution'][1]]
        #try:
        masked, surround = get_masked_data(movie,
                                    e['maskSizePixel'], e['maskLocationPixel'])
#        except:
#            print 'FAILED: ', e['cellid']
#            failed_count += 1
#            continue
        success_count += 1

        lum_mask = get_luminance(masked)[np.newaxis, :]
        con_mask = get_contrast(masked)[np.newaxis, :]
        four_mask, freq_mask, orient_mask = get_fourier2D(masked)
        four_mask = four_mask[np.newaxis, :, :, :]
        freq_mask = freq_mask[np.newaxis, :, :]
        orient_mask = orient_mask[np.newaxis, :, :]

        lum_surr = []
        con_surr = []
        four_surr = []
        freq_surr = []
        orient_surr = []
        for s in range(surround.shape[0]):
            lum_surr.append(get_luminance(surround[s]))
            con_surr.append(get_contrast(surround[s]))
            f, fr, o = get_fourier2D(surround[s])
            four_surr.append(f)
            freq_surr.append(fr)
            orient_surr.append(o)
        lum_surr = np.array(lum_surr)
        con_surr = np.array(con_surr)
        four_surr = np.array(four_surr)
        freq_surr = np.array(freq_surr)
        orient_surr = np.array(orient_surr)

        lum_whole = get_luminance(movie)[np.newaxis, :]
        con_whole = get_contrast(movie)[np.newaxis, :]
        four_whole, freq_whole, orient_whole = get_fourier2D(movie)
        four_whole = four_whole[np.newaxis, :, :, :]
        freq_whole = freq_whole[np.newaxis, :, :]
        orient_whole = orient_whole[np.newaxis, :, :]

        flow_whole, flow_mask, flow_surr = get_flow(flow,
                                    e['maskSizePixel'], e['maskLocationPixel'])
        movie = movie[np.newaxis, :, :, :]
        masked = masked[np.newaxis, :, :, :]
        np.savez(dat_path + target_fname,
                   lum_mask=lum_mask, con_mask=con_mask, flow_mask=flow_mask,
                   four_mask=four_mask,
                   freq_mask=freq_mask, orient_mask=orient_mask,
                    masked=masked,
                   lum_whole=lum_whole, con_whole=con_whole,
                   flow_whole=flow_whole,
                   four_whole=four_whole, freq_whole=freq_whole,
                   orient_whole=orient_whole, movie=movie,
                   lum_surr=lum_surr, con_surr=con_surr, flow_surr=flow_surr,
                   four_surr=four_surr, freq_surr=freq_surr,
                   orient_surr=orient_surr, surround=surround)

        scipy.io.savemat(dat_path + target_fname[:-3] + '.mat',
            {'lum_mask': lum_mask, 'con_mask': con_mask,
            'flow_mask': flow_mask, 'four_mask': four_mask,
            'freq_mask': freq_mask, 'orient_mask': orient_mask,
            'masked': masked,
            'lum_whole': lum_whole, 'con_whole': con_whole,
            'flow_whole': flow_whole, 'four_whole': four_whole, 
            'freq_whole': freq_whole,
            'orient_whole': orient_whole, 'movie': movie,
            'lum_surr': lum_surr, 'con_surr': con_surr, 'flow_surr': flow_surr,
            'four_surr': four_surr, 'freq_surr': freq_surr,
                   'orient_surr': orient_surr, 'surround': surround})

        plot_movie(lum_mask, con_mask, flow_mask, four_mask, masked,
                   fig_path + e['cellid'] + '_masked')
        plot_movie(lum_whole, con_whole, flow_whole, four_whole, movie,
                   fig_path + e['cellid'] + '_whole')
        plot_surround(lum_surr, con_surr, flow_surr, four_surr, surround,
                   fig_path + e['cellid'] + '_surround')
    print 'Failed: ', failed_count
    print 'Successful: ', success_count



def process_movie_pop(exp_type='POP'):
    failed_count = 0
    success_count = 0
    exps = list_PopExps()
    dat_path = startup.data_path + 'Sparseness/%s/' % exp_type
    fig_path = startup.fig_path + 'Sparseness/%s/movies/' % exp_type

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for exp_id in exps:
#        print 'recording ', exp_id
        target_fname = exp_id + '_processed.npz'
        flow_fname = exp_id + '_flow.mat'
        e = load_PopData(exp_id)
        if os.path.exists(dat_path + target_fname):
        #    print 'already exists, skipping ', target_fname
            success_count += 1
#            continue
        flow_found = True
        if not os.path.exists(dat_path + flow_fname):
            flow_found = False
#            print 'flow data missing ', flow_fname
#            continue

        movie = scipy.io.loadmat(dat_path + e['movie'])
        movie = movie['dat']

        exp_len = e['dat_c'].shape[1] / e['scan_freq']
        movie_len = movie.shape[0] / frame_rate
        if exp_len > movie_len:
            mov_frames = movie.shape[0]
            exp_samples = np.ceil(e['scan_freq'] * movie_len)
        else:
            mov_frames = np.ceil(frame_rate * exp_len)
            exp_samples = e['dat_c'].shape[1]

        print
        print 'Exp: %s,\tScan Rate: %.2f \tFrame Rate: %.2f' % (
                            exp_id, e['scan_freq'],  frame_rate)
        print '\t\t#Samples: %d\t\t#Frames: %d' % (e['dat_c'].shape[1],
                                                   movie.shape[0])
        print '\t\tExp Length: %.2fsec\tMovie Length: %.2fsec' % (exp_len,
                                                                  movie_len)
        print 'Selected\t#Samples: %d\t\t#Frames: %d' % (exp_samples,
                                                         mov_frames)

        movie = movie[:mov_frames]
        if flow_found:
            flow = scipy.io.loadmat(dat_path + flow_fname)
            flow = flow['uv']
            flow = np.swapaxes(flow, 1, 2)
            flow = flow[:mov_frames]

        if (np.abs(e['adjustedMovResolution']
                   - e['movResolution'])).sum() != 0:
            movie = movie[:, :e['adjustedMovResolution'][0],
                      :e['adjustedMovResolution'][1]]
            if flow_found:
                flow = flow[:, :e['adjustedMovResolution'][0],
                      :e['adjustedMovResolution'][1]]
        #try:
        masked, surround = get_masked_data(movie,
                                    e['maskSizePixel'], e['maskLocationPixel'])
#        except:
#            print 'FAILED: ', e['cellid']
#            failed_count += 1
#            continue
        success_count += 1
        exp_time = np.arange(exp_samples) / scan_freq
        mov_time = np.arange(mov_frames) / frame_rate

        lum_mask = downsample(get_luminance(masked),
                              mov_time, exp_time)[np.newaxis, :]
        con_mask = downsample(get_contrast(masked),
                              mov_time, exp_time)[np.newaxis, :]

        four_mask, freq_mask, orient_mask = get_fourier2D(masked)
        four_mask = downsample_multi_dim(four_mask, mov_time,
                               exp_time)[np.newaxis, :, :, :]
        freq_mask = downsample_multi_dim(freq_mask, mov_time,
                               exp_time)[np.newaxis, :, :]
        orient_mask = downsample_multi_dim(orient_mask, mov_time,
                               exp_time)[np.newaxis, :, :]

#        lum_surr = []
#        con_surr = []
#        four_surr = []
#        freq_surr = []
#        orient_surr = []
#        for s in range(surround.shape[0]):
#            lum_surr.append(get_luminance(surround[s]))
#            con_surr.append(get_contrast(surround[s]))
#            f, fr, o = get_fourier2D(surround[s])
#            four_surr.append(f)
#            freq_surr.append(fr)
#            orient_surr.append(o)
#        lum_surr = np.array(lum_surr)
#        con_surr = np.array(con_surr)
#        four_surr = np.array(four_surr)
#        freq_surr = np.array(freq_surr)
#        orient_surr = np.array(orient_surr)

        lum_whole = downsample(get_luminance(movie), mov_time,
                               exp_time)[np.newaxis, :]
        con_whole = downsample(get_contrast(movie), mov_time,
                               exp_time)[np.newaxis, :]
        four_whole, freq_whole, orient_whole = get_fourier2D(movie)
        four_whole = downsample_multi_dim(four_whole, mov_time,
                               exp_time)[np.newaxis, :, :, :]
        freq_whole = downsample_multi_dim(freq_whole, mov_time,
                               exp_time)[np.newaxis, :, :]
        orient_whole = downsample_multi_dim(orient_whole, mov_time,
                               exp_time)[np.newaxis, :, :]
        if flow_found:
            flow_whole, flow_mask, flow_surr = get_flow(flow,
                                e['maskSizePixel'], e['maskLocationPixel'])
        else:
            flow_whole, flow_mask, flow_surr = np.array([]), np.array([]), np.array([])

        movie = movie[np.newaxis, :, :, :]
        masked = masked[np.newaxis, :, :, :]
        np.savez(dat_path + target_fname,
                   lum_mask=lum_mask, con_mask=con_mask, flow_mask=flow_mask,
                   four_mask=four_mask,
                   freq_mask=freq_mask, orient_mask=orient_mask,
                    masked=masked,
                   lum_whole=lum_whole, con_whole=con_whole,
                   flow_whole=flow_whole,
                   four_whole=four_whole, freq_whole=freq_whole,
                   orient_whole=orient_whole, movie=movie,
                   mov_frames=mov_frames, exp_samples=exp_samples
#                   lum_surr=lum_surr, con_surr=con_surr, flow_surr=flow_surr,
#                   four_surr=four_surr, freq_surr=freq_surr,
#                   orient_surr=orient_surr, surround=surround
        )

        scipy.io.savemat(dat_path + target_fname[:-3] + '.mat',
            {'lum_mask': lum_mask, 'con_mask': con_mask,
            'flow_mask': flow_mask, 'four_mask': four_mask,
            'freq_mask': freq_mask, 'orient_mask': orient_mask,
            'masked': masked,
            'lum_whole': lum_whole, 'con_whole': con_whole,
            'flow_whole': flow_whole, 'four_whole': four_whole,
            'freq_whole': freq_whole,
            'orient_whole': orient_whole, 'movie': movie,
            'mov_frames': mov_frames, 'exp_samples': exp_samples
#            'lum_surr': lum_surr, 'con_surr': con_surr, 'flow_surr': flow_surr,
#            'four_surr': four_surr, 'freq_surr': freq_surr,
#                   'orient_surr': orient_surr, 'surround': surround
            })

        plot_movie(lum_mask, con_mask, flow_mask, four_mask, masked,
                   fig_path + exp_id + '_masked')
        plot_movie(lum_whole, con_whole, flow_whole, four_whole, movie,
                   fig_path + exp_id + '_whole')
#        plot_surround(lum_surr, con_surr, flow_surr, four_surr, surround,
#                   fig_path + exp_id + '_surround')
    print 'Failed: ', failed_count
    print 'Successful: ', success_count


if __name__ == "__main__":
    exp_type = 'POP'
    process_movie_pop()
#        animate_matrix_multi([masked, four_mask])
