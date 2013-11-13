




# def MNISTTrainer(Trainer):
#     def __init__(self, network, 


#     def cost(self, x, y):
#         lamb = T.cast(self.lamb, self.dtype)
#         rho = T.cast(self.rho, self.dtype)
#         return ((x - y)**2).mean(axis=0).sum() + lamb*(T.abs_(self.q - rho)).sum()



# import collections
import os
# import sys
# import time

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
plt.ion()

import deepnet
import deepnet.autoencoder
from deepnet.autoencoder import Autoencoder, SparseAutoencoder
import deepnet.functions
import deepnet.image_tools

from skdata.mnist.dataset import MNIST
mnist = MNIST()
mnist.meta      # accessing this forces data arrays to be built

images = mnist.arrays['train_images']
labels = np.asarray([m['label'] for m in mnist.meta if m['split'] == 'train'])

plt.figure(1)
plt.clf()
deepnet.image_tools.tile(images, rows=5, cols=10)

################################################################################
loadfile = ''
if not os.path.exists(loadfile):

    linear = deepnet.functions.Linear(slope=1.0)
    noisylif = deepnet.functions.NoisyLIFApprox(
        tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)

    params = [(SparseAutoencoder, (50, 50), {'rfshape': (9,9), 'f': noisylif, 'g': linear}),
              (Autoencoder, (30, 30), {'f': noisylif, 'g': noisylif}),
              (Autoencoder, (20, 20), {'f': noisylif, 'g': noisylif}),
              (Autoencoder, (15, 15), {'f': linear, 'g': noisylif})]

    layers = []
    for param in params:
        visshape = images.shape if len(layers) == 0 else layers[-1].hidshape
        EncoderClass, hidshape, p = param
        enc = EncoderClass(visshape=visshape, hidshape=hidshape, **p)
        layers.append(enc)

    net = autoencoder.DeepAutoencoder(layers)
else:
    net = autoencoder.DeepAutoencoder.fromfile(loadfile)

################################################################################
train_params = [{'rho': 0.01, 'lamb': 25, 'noisestd': 0.2},
                {'rho': 0.05, 'lamb': 0, 'noisestd': 0.2},
                {'rho': 0.05, 'lamb': 0, 'noisestd': 0.2},
                {'rho': 0.05, 'lamb': 0, 'noisestd': 0.2}]

for layer, params in zip(net.layers, train_params):
    if not layer.pretrained:
        layer.set_train_params(**params)

sgd_params = [{'nepochs': 30, 'rate': 0.05},
              {'nepochs': 30, 'rate': 0.05},
              {'nepochs': 30, 'rate': 0.01},
              {'nepochs': 30, 'rate': 0.01}]

net.sgd_layers(images, images.subset(500), sgd_params=sgd_params, savename=loadfile)
# net.save(loadfile)

results = net.compVHV(images)
images.compare(results, fignum=998)
print images.rmse(results)

if True:
    net.untie()
    net.lbfgs(images, nevals=100)
    results = net.compVHV(images)
    images.compare(results, fignum=999)
    print images.rmse(results)

