
import matplotlib
# force plots to file. no display. comment out to use plt.show()
#matplotlib.use('Agg')
import numpy as np
import sys
import pylab as plt
sys.path.append('..')
from data_utils.movie import load_movie_data
from data_utils.load_ephys import load_EphysData
import Image

def make_patches(data, dim=20):
    res = np.zeros([data.shape[0], dim, dim])
    for i, d in enumerate(data):
        im = Image.fromarray(d)
        im = im.resize((dim, dim), Image.BICUBIC)
        res[i] = np.asarray(im)
    return res



exp_types = ['FS', 'PYR', 'SOM']
results = {}
for exp_type in exp_types:
    results[exp_type] = {}
    ephys = load_EphysData(exp_type)
    for e in ephys.values():
        cellid = e['cellid']
        mov = load_movie_data(cellid, exp_type)
        mov = mov['masked']
        mov = make_patches(mov[0])
        results[exp_type][cellid] = {'movie': mov, 'responses': {}}

np.save('mask_data', results)
results = np.load('mask_data.npy')
print results.item().keys()
