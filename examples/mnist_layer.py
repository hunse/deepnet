
import os
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

from skdata.mnist.dataset import MNIST
mnist = MNIST()
mnist.meta      # accessing this forces data arrays to be built

images = mnist.arrays['train_images'].astype('float32')
images = (images - images.mean()) / images.std()

labels = np.asarray([m['label'] for m in mnist.meta if m['split'] == 'train'])
imshape = images.shape[1:]

plt.figure(1)
plt.clf()
deepnet.image_tools.tile(images, rows=5, cols=10)

################################################################################
### train one layer

# loadfile = None
loadfile = 'mnist_layer.pkl'

if loadfile is None or not os.path.exists(loadfile):

    linear = Linear(slope=1.0)
    noisylif = NoisyLIFApprox(
        tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)

    # layer = SparseAutoencoder(visshape=imshape, hidshape=(50,50),
    #                           rfshape=(9,9), f=noisylif, g=linear)
    layer = SparseAutoencoder(visshape=imshape, hidshape=(40,40),
                              rfshape=(9,9), f=noisylif, g=linear)

    if loadfile is not None:
        layer.tofile(loadfile)
else:
    layer = deepnet.CacheObject.fromfile(loadfile)

################################################################################
train_params = {'rho': 0.01, 'lamb': 25, 'noise_std': 0.2}
trainer = SparseTrainer(layer, **train_params)

sgd(trainer, images, nepochs=30, rate=0.05)

if 0:
    ### untied training
    sgd(trainer, images, nepochs=1, rate=0.05)
    layer.untie()

    trainer = SparseTrainer(layer, **train_params)
    sgd(trainer, images, nepochs=30, rate=0.05)

results = layer.compVHV(images)

plt.figure(1)
plt.clf()
deepnet.image_tools.compare([images, results], vlims=(-1,1))
