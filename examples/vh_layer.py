"""
Learn a single-layer sparse autoencoder on Van Hateren data
(as in CogSci 2013 paper)
"""

import sys, os, time

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
import theano

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
plt.ion()

import deepnet
import deepnet.autoencoder
from deepnet.autoencoder import Autoencoder, SparseAutoencoder
from deepnet.autoencoder import SparseTrainer, sgd
from deepnet.functions import Linear, NoisyLIFApprox
import deepnet.image_tools

import skdata.vanhateren.dataset
data = skdata.vanhateren.dataset.Calibrated(50)
data.meta      # accessing this forces data arrays to be built

N = 60000
S = 32
patches = data.raw_patches((N, S, S), items=data.meta[:data.n_item_limit])
patches = patches.astype('float32')

def normalize_patch(p):
    p = (p - p.mean()) / (p.std() + 1)
    return p

for p in patches:
    p[:] = normalize_patch(p)

imshape = patches.shape[1:]

# seed = 7

################################################################################
### train one layer

filename = 'vh_layer.npz'

if 'filename' not in locals() or not os.path.exists(filename):
    linear = Linear(slope=1.0)
    noisylif = NoisyLIFApprox(
        tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)

    # layer = SparseAutoencoder(visshape=imshape, hidshape=(50,50),
    #                           rfshape=(9,9), f=noisylif, g=linear, seed=seed)
    layer = SparseAutoencoder(visshape=imshape, hidshape=(60,60),
                              rfshape=(11,11), f=noisylif, g=linear)

    # train_params = {'rho': 0.01, 'lamb': 25, 'noise_std': 0.2}
    train_params = {'rho': 0.01, 'lamb': 5, 'noise_std': 0.2}
    trainer = SparseTrainer(layer, **train_params)

    sgd(trainer, patches, nepochs=30, rate=0.05, vlims=(-2,2))

    if 'filename' in locals():
        layer.to_file(filename)

else:
    layer = deepnet.CacheObject.from_file(filename)

### test the layer
if 1:
    test = patches[:100]
    recs = layer.compVHV(test)
    rmse = np.sqrt(((recs - test)**2).mean())
    print rmse

    plt.figure(1)
    plt.clf()
    deepnet.image_tools.compare([test, recs], vlims=(-2,2))

