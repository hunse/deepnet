
"""
Learn a single-layer sparse autoencoder on Van Hateren data
(as in CogSci 2013 paper)
"""

import sys, os, time, datetime

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

# patches = (2*(patches > 0) - 1).astype('float32')

################################################################################
# loadfile = 'results/vh_tied.npz'
# loadfile = 'results/vh_binary.npz'
loadfile = 'results/vh_flatlif.npz'
if not os.path.exists(loadfile):

    linear = func.Linear(slope=1.0)
    # noisylif = func.NoisyLIFApprox(
    #     tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)
    noisylif = func.NoisyLIFApprox(
        tRef=0.002, tauRC=0.05, alpha=0.7, xint=-0.5, amp=1./50, sigma=0.001)

    # params = [(auto.SparseAutoencoder, (50, 50), {'rfshape': (9,9), 'f': noisylif, 'g': linear}),
    #           (auto.Autoencoder, (40, 40), {'f': noisylif, 'g': noisylif}),
    #           (auto.Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
    #           (auto.Autoencoder, (20, 20), {'f': linear, 'g': noisylif})]

    params = [
        #(auto.SparseAutoencoder, (50, 50), {'rfshape': (9,9), 'f': noisylif, 'g': linear}),
        "results/vh_flatlif.npz.layer_0_2013-10-07_12:00:39.npz",
        # (auto.Autoencoder, (40, 40), {'f': noisylif, 'g': noisylif}),
        "results/vh_flatlif.npz.layer_1_2013-10-07_12:20:10.npz",
        # (auto.Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
        "results/vh_flatlif.npz.layer_2_2013-10-07_12:25:06.npz",
        (auto.Autoencoder, (20, 20), {'f': linear, 'g': noisylif})]

    # params = ['results/vh_flatlif.npz.layer_0_2013-10-04_16:07:47.npz',
    #           # (auto.Autoencoder, (40, 40), {'f': noisylif, 'g': noisylif}),
    #           # (auto.SparseAutoencoder, (40, 40), {'rfshape': (13, 13), 'f': noisylif, 'g': noisylif}),
    #           (auto.Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
    #           (auto.Autoencoder, (20, 20), {'f': linear, 'g': noisylif})]

    # params = ['results/vh_layer.npz',
    #           # (auto.SparseAutoencoder, (40, 40), {'rfshape': (13, 13), 'f': noisylif, 'g': noisylif}),
    #           'results/vh_tied.npz.layer_1.npz',
    #           # (auto.Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
    #           # 'results/vh_tied.npz.layer_2.npz',
    #           'results/vh_tied.npz.layer_2_2013-10-02_13:33:09.npz',
    #           # (auto.Autoencoder, (20, 20), {'f': linear, 'g': noisylif})
    #           'results/vh_tied.npz.layer_3_2013-10-02_13:36:00.npz'
    #           ]

    layers = []
    for param in params:
        if isinstance(param, str):
            # load from file
            enc = deepnet.base.CacheObject.from_file(param)
        else:
            # make a new layer
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

# sgd_params = [dict(n_epochs=30, rate=0.05, clip=(-1,1)) for i in net.layers]
sgd_params = [dict(n_epochs=30, rate=0.05, clip=(-1,1)),
              dict(n_epochs=30, rate=0.05, clip=(-1,1)),
              dict(n_epochs=30, rate=0.01, clip=(-1,1)),
              dict(n_epochs=50, rate=0.005, clip=(-1,1))]


if any(algo_epochs(layer, 'sgd') < sgd_params[i]['n_epochs']
       for i, layer in enumerate(net.layers)):

    if imtools.display_available():
        ### set figure size and position
        fig = plt.figure(101, figsize=(11.925, 12.425))
        # figman = plt.get_current_fig_manager()
        # figman.window.wm_geometry('954x1028+2880+0')
        # fig.set_size_inches([11.925, 12.425])

    images = patches[:]
    timages = patches[:500]

    # train_params = [{'rho': 0.01, 'lamb': 5, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 0.1, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2}]

    # train_params = [{'rho': 0.05, 'lamb': 5, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 1, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2},
    #                 {'rho': 0.05, 'lamb': 0, 'noise_std': 0.2}]

    train_params = [{'rho': 0.01, 'lamb': 5, 'noise_std': 0.1},
                    {'rho': 0.05, 'lamb': 0, 'noise_std': 0.1},
                    {'rho': 0.05, 'lamb': 0, 'noise_std': 0.1},
                    {'rho': 0.05, 'lamb': 0, 'noise_std': 0.0}]

    for i, layer in enumerate(net.layers):
        sgd_param = sgd_params[i]
        train_param = train_params[i]

        # subtract completed epochs
        sgd_param['n_epochs'] -= algo_epochs(layer, 'sgd')
        if sgd_param['n_epochs'] > 0:
            trainer = auto.SparseTrainer(layer, **train_param)
            test_fn = net.propVHV_fn(maxlayer=i)
            # save_fn = lambda: net.to_file(loadfile)
            save_fn = None
            auto.sgd(trainer, images, timages, test_fn=test_fn,
                     vlims=(-2,2), save_fn=save_fn,
                     **sgd_param)

            if save_fn is None:
                # net.to_file(loadfile)
                layer_file = "%s.layer_%d_%s.npz" % (
                        loadfile, i,
                        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                layer.to_file(layer_file)

        images = layer.compup(images)

    net.to_file(loadfile)

if 1:
    results = net.compVHV(patches)
    rmses = imtools.rmse(patches, results)
    print "rmse", rmses.mean(), rmses.std()

    if imtools.display_available():
        plt.figure(figsize=(11.925, 12.425))
        imtools.compare([patches, results], rows=8, cols=12, vlims=(-2,2))
