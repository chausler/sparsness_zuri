import numpy.fft as fft
import numpy as np
import cv2
from cv2 import cv
import sys
import math
import scipy.io
import scipy.stats as spt
import pylab as plt
sys.path.append('..')
import startup
import os
from plotting.utils import adjust_spines
from data_utils.load_ephys import load_EphysData_SOM


lk_params = dict( winSize  = (3, 3), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  derivLambda = 0.0 )    

feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.1,
                       minDistance = 5,
                       blockSize = 5 )


def ellipse(width, height):
    
    horpow = 2;
    verpow = horpow;
    width = width + 0.5 # to produce a Ellipse with horizontal axis == ceil(2*hor semi axis)
    height = height + 0.5 #to produce a Ellipse with vertical axis == ceil(2*vert semi axis)
    [x,y] = np.meshgrid(np.arange(-width, width+1), np.arange(-height, height+1));
    #print width, height, x.shape, y.shape
    bool    = (abs(x / width) ** horpow + abs(y / height) ** verpow)  < 1;    
    xs = plt.find(bool.sum(0)==0)
    ys = plt.find(bool.sum(1)==0)
    bool = bool[ys[0]+1:ys[1],:]
    bool = bool[:, xs[0] + 1 : xs[1]]
    bool = bool.T
    mask_inds = np.where(bool>0)
    mask_inds = np.array(mask_inds).T
    inv_mask_inds = np.where(bool==0)
    inv_mask_inds = np.array(inv_mask_inds).T
    
    box_inds = np.array(np.where(bool != 5555)).T    
    box_edges = np.array(
                [[box_inds[:,0].min(), box_inds[:,1].min()], 
                [box_inds[:,0].max(), box_inds[:,1].max()]])
    return mask_inds, inv_mask_inds, box_edges
    
    #bool = bool(cropcoords(3):cropcoords(4),cropcoords(1):cropcoords(2));


def get_contrast(movie):
    print 'get contrast ...'
    contrast = np.zeros(len(movie))
    for i, frame in enumerate(movie):
        contrast[i] = np.std(frame.ravel()) #frame.max() - frame.min() / frame.mean()
    return contrast

def get_luminance(movie):
    print 'get luminence ...'
    luminance = np.zeros(len(movie))
    for i, frame in enumerate(movie):
        luminance[i] = frame.mean()
    return luminance

def get_flow(flow,  maskSizePixel, maskLocationPixel):
    print 'get flow ...' 
    masked, surround = get_masked_data(flow, maskSizePixel, maskLocationPixel)
    
    masked_sum = []
    whole_sum = []
    surround_sum = []
    
    for m in masked:
        m = m.reshape(-1,2)
        [x,y] = spt.nanmean(m, 0)
        r = np.sqrt(x**2 + y**2)
        theta = math.atan2(y,x)
        masked_sum.append([theta, r])
    
    for m in flow:
        m = m.reshape(-1,2)
        [x,y] = spt.nanmean(m, 0)
        r = np.sqrt(x**2 + y**2)
        theta = math.atan2(y,x)
        whole_sum.append([theta, r])
    
    for m in surround:
        m = m.reshape(-1,2)
        [x,y] = spt.nanmean(m, 0)
        r = np.sqrt(x**2 + y**2)
        theta = math.atan2(y,x)
        surround_sum.append([theta, r])
            
    return np.array(whole_sum), np.array(masked_sum), np.array(surround_sum)
    

def get_flow_old(movie):
    print 'get flow ...'
    change = [[0,0]] 
    old_mode = True    
    for i, frame in enumerate(movie):        
        frame = np.array(frame)
        vis = np.array(frame)
        if i > 0:
            p0 = cv2.goodFeaturesToTrack(prev_frame, **feature_params)
            img0 = prev_frame 
            img1 = frame            
            p1,  st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            local_change = []
            the_change = np.array([0,0])
            cnt = 0.
            for (x0, y0), (x, y), (s) in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), st):               
                if x >= 0 and y >= 0 and s == 1:
                    local_change.append([x, y, x0, y0])
                    the_change += np.array([x0-x, y0-y])
                    cnt += 1.
            if cnt > 0:
                the_change = np.array(the_change / cnt)
            #tracks.append(local_change)
            #print the_change
            r = np.sqrt(the_change[0]**2 + the_change[1]**2)
            theta = math.atan2(the_change[1],the_change[0])
            change.append([theta, r])
        prev_frame = np.array(frame) 
        
    return np.array(change)

def get_fourier2D(movie):
    fouriers = []
    print 'doing fourier2D'
    for i, frame in enumerate(movie):
        if i % 100 == 0:
            print 'frame %d of %d' % (i, len(movie))
        frame = frame / 255.        
        A = fft.fftshift(fft.fft2(frame))
        added = np.sqrt(A.real**2 + A.imag**2)
        added *= np.sign(A.imag)
        fouriers.append(added)
    fouriers = np.array(fouriers)
    tmp = fouriers.max(0)
    thresh = tmp.mean() + 4 * np.std(tmp)
    tmp = tmp > thresh
    ys = tmp.sum(1)
    xs = tmp.sum(0)
    midx = np.ceil(tmp.shape[0]/2.)
    midy = np.ceil(tmp.shape[1]/2.)
    diff_x = 7#(plt.find(xs>0).max() - plt.find(xs>0).min()) / 2
    diff_y = 7#(plt.find(ys>0).max() - plt.find(ys>0).min()) / 2
    if diff_x < 1:
        diff_x = 1
    if diff_y < 1:
        diff_y = 1
    if fouriers.shape[1] < (diff_x * 2 + 1) or fouriers.shape[2] < (diff_x * 2 + 1):
        print 'FUCK!'
        h = 1/0  
    fouriers = fouriers[:, midx - diff_x : midx + diff_x + 1, :]
    fouriers = fouriers[:, :,  midy - diff_y : midy + diff_y + 1]    
    #print fouriers.mean(), np.std(fouriers)
    thresh = fouriers.mean() + 4 * np.std(fouriers)
    #fouriers = np.log(fouriers)
    print 
    fouriers[fouriers>thresh] = thresh
#    plt.imshow(fouriers.max(0))
#    plt.colorbar()
#    plt.show()    
#    animate_matrix_multi([movie, fouriers])
#        print A 
#        plt.imshow(added)
#        plt.show()
    return fouriers
    
def plot_movie(lum, con, flow, four, movie, fname):
    num_frames = len(lum)
    fig = plt.figure(figsize=(16,7))
    fig.set_facecolor('white')
    
    ax = plt.subplot(231)
    plt.plot(lum)
    plt.xlim(0,num_frames)
    plt.title('Luminence')
    adjust_spines(ax,['left', 'bottom'])
    
    ax = plt.subplot(234)
    plt.plot(con)
    plt.xlim(0,num_frames)
    plt.title('Contrast')
    adjust_spines(ax,['left', 'bottom'])
    
    total_plots = 4
    frames = np.linspace(0, num_frames-1, total_plots).astype(np.int)
    print 'plotting frames ', frames
    lims = [four[frames].min(), four[frames].max()]
    for i, f in enumerate(frames):        
        ax = plt.subplot(3,6,3+i,  aspect='equal')  
        plt.imshow(movie[i].T, cmap=plt.cm.gray)        
        adjust_spines(ax,[])   
        if i==0:            
            plt.title('Movie')
             
        ax = plt.subplot(3,6,9+i,  aspect='equal')
        plt.imshow(four[i].T, cmap=plt.cm.gray, interpolation='None')
        plt.clim(lims)
        if i==0:
            adjust_spines(ax,['left', 'bottom'])
            plt.title('Spatial Fourier')            
            tix = np.array(plt.xticks()[0], dtype=np.float)
            plt.xticks(np.linspace(tix.min(), tix.max(), 4).astype(np.int))
            tix = np.array(plt.yticks()[0])
            plt.yticks(np.linspace(tix.min(), tix.max(), 4).astype(np.int))
            plt.colorbar()
        else:
            adjust_spines(ax,[])
            
        ax = plt.subplot(3,6,15+i, polar=True)        
        plt.plot([0,flow[i,0]], [0,flow[i,1]], '-o', color='0.3', linewidth=2)
        plt.ylim(0, flow[:,1].max())
        for loc, spine in ax.spines.iteritems():
                spine.set_color('none') # don't draw spine
        #plt.adjust_spines(ax,[])
        if i != 0:
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
        else:
            plt.title('Flow')
        plt.xticks([0, np.pi/2, np.pi, np.pi * 3./2,])
        plt.yticks(np.linspace(0, flow[:,1].max(), 4).astype(np.int))
    
    plt.subplots_adjust(left=0.04, bottom=0.06, right=0.96, top=0.95,
                         wspace=0.20, hspace=0.35)
        #adjust_spines(ax,[])
    fig.savefig( fname + '.eps')
    fig.savefig( fname + '.png')
    plt.close(fig)   
    
    
def get_masked_data(data, maskSizePixel, maskLocationPixel):      
    
    mask_inds, inv_mask_inds, box_edges = ellipse(maskSizePixel[0] / 2 ,
                                   maskSizePixel[1] / 2 )
    
    box_edges += (maskLocationPixel - maskSizePixel / 2).astype(np.int)
    mask_inds += (maskLocationPixel - maskSizePixel / 2).astype(np.int)
    masked = np.copy(data)
    surround = np.copy(data)
    masked = masked[:,box_edges[0,0]:box_edges[1,0] + 1,:]    
    masked = masked[:,:,box_edges[0,1]:box_edges[1,1] + 1]    
    #mock_masked = np.ones_like(masked[0]) 
    #mock_surround = np.ones_like(masked[0]) 
    for ind in inv_mask_inds:        
        masked[:,ind[0], ind[1]] = np.nan
    
    for ind in mask_inds:
        surround[:,ind[0], ind[1]] = np.nan
  
    return masked, surround
    

    
    

if __name__=="__main__":    
    
    ellipse(20, 20)
    ephys = load_EphysData_SOM()
    dat_path = startup.data_path + 'ephys/som/'
    fig_path = startup.fig_path + 'ephys/som/movies/'
    for e in ephys.values():
        print 'recording ', e['expdate']
#        if e['expdate'] != '120601':
#            continue
        target_fname = e['expdate'] + '_processed.npz'
        flow_fname = e['expdate'] + '_flow.mat'
        if os.path.exists(dat_path + target_fname):
            print 'already exists, skipping ', target_fname
            
        if not os.path.exists(dat_path + flow_fname):
            print 'flow data missing, skipping ', flow_fname
            continue
        
        movie = scipy.io.loadmat(dat_path + e['expdate'] + '.mat')
        movie = movie['dat']
        flow = scipy.io.loadmat(dat_path + flow_fname)
        flow = flow['uv']
        flow = np.swapaxes(flow, 1,2)
        actual_len = e['psth_c'].shape[1]
        flow = flow[:actual_len]
        movie = movie[:actual_len]
        
        if (np.abs(e['adjustedMovResolution'] - e['movResolution'])).sum() != 0:
            movie = movie[:, :e['adjustedMovResolution'][0], 
                      :e['adjustedMovResolution'][1]]
            flow = flow[:, :e['adjustedMovResolution'][0], 
                      :e['adjustedMovResolution'][1]]
        masked, surround = get_masked_data(movie, e['maskSizePixel'], e['maskLocationPixel'])        
        lum_mask = get_luminance(masked)
        con_mask = get_contrast(masked)
        four_mask = get_fourier2D(masked)
        
        lum_surr = get_luminance(surround)
        con_surr = get_contrast(surround)
        four_surr = get_fourier2D(surround)
        
        lum_whole = get_luminance(surround)
        con_whole = get_contrast(surround)
        four_whole = get_fourier2D(surround)
               
                        
        flow_whole, flow_mask, flow_surr = get_flow(flow, 
                                    e['maskSizePixel'], e['maskLocationPixel'])
        
        np.savez(dat_path + e['expdate'] + '_processed.npz', 
                   lum_mask=lum_mask, con_mask=con_mask, flow_mask=flow_mask, 
                   four_mask=four_mask, masked=masked,
                   lum_whole=lum_whole, con_whole=con_whole, flow_whole=flow_whole, 
                   four_whole=four_whole, movie=movie,
                   lum_surr=lum_surr, con_surr=con_surr, flow_surr=flow_surr, 
                   four_surr=four_surr, surround=surround )
        plot_movie(lum_mask, con_mask, flow_mask, four_mask, masked, 
                   fig_path + e['expdate'] + '_masked')
        plot_movie(lum_whole, con_whole, flow_whole, four_whole, movie, 
                   fig_path + e['expdate'] + '_whole')
        plot_movie(lum_surr, con_surr, flow_surr, four_surr, surround, 
                   fig_path + e['expdate'] + '_surround')
#        animate_matrix_multi([masked, four_mask])
        
