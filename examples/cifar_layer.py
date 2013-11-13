"""
Learn a single-layer sparse autoencoder on CIFAR-10
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
# from deepnet.autoencoder import Autoencoder, SparseAutoencoder
# from deepnet.autoencoder import SparseTrainer, sgd
# from deepnet.functions import Linear, NoisyLIFApprox
import deepnet.image_tools as imtools

from skdata.cifar10.dataset import CIFAR10
data = CIFAR10()
data.meta      # accessing this forces data arrays to be built

test_mask = np.asarray([m['split'] == 'test' for m in data.meta])
# images = data._pixels[~test_mask]
# timages = data._pixels[test_mask]
images = data._pixels[~test_mask].astype('float32') / 255.
timages = data._pixels[test_mask].astype('float32') / 255.
labels = np.asarray([m['label'] for m in data.meta if m['split'] == 'train'])
tlabels = np.asarray([m['label'] for m in data.meta if m['split'] == 'test'])

imshape = images.shape[1:]

# def luminance(images):
#     return np.sqrt((images**2).sum(axis=-1))

def normalize(images):
    # mean = luminance(images).mean(axis=(1,2), keepdims=1)
    # std = luminance(images).std(axis=(1,2), keepdims=1)
    mean = images.mean(axis=(1,2,3), keepdims=1)
    std = images.std(axis=(1,2,3), keepdims=1)
    return (images - mean) / np.maximum(std, 1e-3)

images = normalize(images)
timages = normalize(timages)

if 0:
    plt.figure(1)
    plt.clf()
    imtools.tile(images, vlims=(-2,2))
    plt.show()

################################################################################
### train one layer

filename = 'results/cifar_layer.npz'
if 'filename' not in locals() or not os.path.exists(filename):
    linear = func.Linear(slope=1.0)
    # noisylif = NoisyLIFApprox(
    #     tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)
    noisylif = func.NoisyLIFApprox(
        tRef=0.002, tauRC=0.05, alpha=0.7, xint=-0.5, amp=1./50, sigma=0.001)

    layer = auto.SparseAutoencoder(visshape=imshape, hidshape=(70,70),
                                   rfshape=(11,11), f=noisylif, g=linear)

    # train_params = {'rho': 0.01, 'lamb': 25, 'noise_std': 0.2}
    # train_params = {'rho': 0.01, 'lamb': 5, 'noise_std': 0.2}
    train_params = {'rho': 0.05, 'lamb': 5, 'noise_std': 0.2}
    trainer = auto.SparseTrainer(layer, **train_params)

    plt.figure(101)
    raw_input("Please place the figure...")

    auto.sgd(trainer, images, n_epochs=30, rate=0.05, vlims=(-2,2))

    if 'filename' in locals():
        layer.to_file(filename)

else:
    layer = deepnet.CacheObject.from_file(filename)

### test the layer
if 1:
    # test = timages[:200]
    test = timages
    recs = layer.compVHV(test)
    rmse = np.sqrt(((recs - test)**2).mean())
    print "rmse", rmse

    plt.figure(1)
    plt.clf()
    imtools.compare([test, recs], vlims=(-2,2))


