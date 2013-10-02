
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
import deepnet.autoencoder as auto
import deepnet.functions as func
import deepnet.image_tools as imtools

import skdata.vanhateren.dataset
data = skdata.vanhateren.dataset.Calibrated(50)
data.meta      # accessing this forces data arrays to be built

N = 60000
S = 32
patches = data.raw_patches((N, S, S), items=data.meta[:data.n_item_limit])
patches = patches.astype('float32')
patch_shape = patches.shape[1:]

### intensities are essentially log-normally distributed. So take the log
patches = np.log1p(patches)

def normalize(p):
    std0 = patches.std()
    mean, std = p.mean(axis=(1,2)), p.std(axis=(1,2))
    return ((p - mean[:,None,None]) / np.maximum(std, 0.01*std0)[:,None,None])
patches = normalize(patches)
patches = patches.clip(-3, 3)

################################################################################
loadfile = 'results/vh_tied.npz'
if not os.path.exists(loadfile):

    linear = func.Linear(slope=1.0)
    noisylif = func.NoisyLIFApprox(
        tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)

    params = [(auto.SparseAutoencoder, (50, 50), {'rfshape': (9,9), 'f': noisylif, 'g': linear}),
              (auto.Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
              (auto.Autoencoder, (20, 20), {'f': noisylif, 'g': noisylif}),
              (auto.Autoencoder, (15, 15), {'f': linear, 'g': noisylif})]

    layers = []
    for param in params:
        visshape = patch_shape if len(layers) == 0 else layers[-1].hidshape
        EncoderClass, hidshape, p = param
        enc = EncoderClass(visshape=visshape, hidshape=hidshape, **p)
        layers.append(enc)

    net = auto.DeepAutoencoder(layers)
else:
    net = auto.DeepAutoencoder.from_file(loadfile)

# assert 0

# sgd_epochs = [layer for layer in net.layers if layer.train_stats

def algo_epochs(layer, algo):
    return sum([s['n_epochs'] for s in layer.train_stats
                if s['algorithm'] == algo])

sgd_params = [dict(n_epochs=30, rate=0.05, clip=(-1,1)) for i in net.layers]

if any(algo_epochs(layer, 'sgd') < sgd_params[i]['n_epochs']
       for i, layer in enumerate(net.layers)):
    images = patches[:]
    timages = patches[:500]

    train_params = [{'rho': 0.01, 'lamb': 5, 'noise_std': 0.2},
                    {'rho': 0.05, 'lamb': 0.1, 'noise_std': 0.2},
                    {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2},
                    {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2}]

    for i, layer in enumerate(net.layers):
        sgd_param = sgd_params[i]
        train_param = train_params[i]

        # subtract completed epochs
        sgd_param['n_epochs'] -= algo_epochs(layer, 'sgd')
        if sgd_param['n_epochs'] > 0:
            trainer = auto.SparseTrainer(layer, **train_param)
            test_fn = net.propVHV_fn(maxlayer=i)
            save_fn = lambda: net.to_file(loadfile)
            auto.sgd(trainer, images, timages, test_fn=test_fn,
                     vlims=(-2,2), save_fn=save_fn,
                     **sgd_param)

        images = layer.compup(images)


if 1:
    results = net.compVHV(patches)
    rmses = imtools.rmse(patches, results)
    print "rmse", rmses.mean(), rmses.std()

    if imtools.display_available():
        imtools.compare([patches, results], vlims=(-2,2))
