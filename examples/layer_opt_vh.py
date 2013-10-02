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

import hyperopt

import deepnet
import deepnet.autoencoder
from deepnet.autoencoder import Autoencoder, SparseAutoencoder
from deepnet.autoencoder import SparseTrainer, sgd
from deepnet.functions import Linear, NoisyLIFApprox
import deepnet.image_tools as imtools

import skdata.vanhateren.dataset
data = skdata.vanhateren.dataset.Calibrated(50)
data.meta      # accessing this forces data arrays to be built

N = 60000
S = 32
patches = data.raw_patches((N, S, S), items=data.meta[:data.n_item_limit])
patches = patches.astype('float32')
imshape = patches.shape[1:]

### intensities are essentially log-normally distributed. So take the log
patches = np.log1p(patches)

def normalize(p):
    std0 = patches.std()
    mean, std = p.mean(axis=(1,2)), p.std(axis=(1,2))
    return ((p - mean[:,None,None]) / np.maximum(std, 0.01*std0)[:,None,None])
patches = normalize(patches)
patches = patches.clip(-3, 3)

################################################################################
### train one layer

linear = Linear(slope=1.0)
noisylif = NoisyLIFApprox(
    tRef=0.02, tauRC=0.06, alpha=10.0, xint=-0.5, amp=1./41, sigma=0.05)

def objective(args):
    n_epochs = 10
    rho = 0.01
    lamb = 5
    noise_std = 0.2
    hidshape = (50, 50)

    rflen, rate, clip = args
    rfshape = (rflen, rflen)
    clip = (-clip, clip)

    layer = SparseAutoencoder(
        visshape=imshape, hidshape=hidshape, rfshape=rfshape,
        f=noisylif, g=linear)

    train_params = {'rho': rho, 'lamb': lamb, 'noise_std': noise_std}
    trainer = SparseTrainer(layer, **train_params)

    stats = sgd(trainer, patches,
                n_epochs=n_epochs, rate=rate, clip=clip, show=False)

    cost = stats['cost'][-1]

    filename = 'results/layer_opt_vh_cost=%0.3e.npz' % cost
    layer.to_file(filename)

    return cost

ratio = np.log(10)
space = (hyperopt.hp.randint('rflen', 14) + 7,
         hyperopt.hp.loguniform('rate', -5*ratio, 0),
         hyperopt.hp.loguniform('clip', -3*ratio, 3*ratio))

best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=20)
print best

