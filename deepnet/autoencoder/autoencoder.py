
import collections
import os
import re
import subprocess
import sys
import time

import numpy as np
import numpy.random as npr
import scipy as sp

import matplotlib.pyplot as plt

# gpuflags = 'device=gpu,floatX=float32'
# if os.environ.has_key('THEANO_FLAGS'):
#     os.environ['THEANO_FLAGS'] += gpuflags
# else:
#     os.environ['THEANO_FLAGS'] = gpuflags
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from ..base import CacheObject
from ..functions.functions import Logistic

def print_gpu_memory():
    outstr = subprocess.check_output(['nvidia-smi', '-q', '--id=0'])
    m = re.findall('([0-9]+) MB', outstr)
    mems = [float(i) for i in m]
    print "GPU memory: %0.1f%% used" % (100*mems[1] / mems[0])

def check_shape(shape):
    if isinstance(shape, tuple):
        assert len(shape) == 2
        return shape, shape[0]*shape[1]
    else:
        size = int(shape)
        return (size, 1), size

def numel_shape(shape):
    p = 1
    for d in shape:
        p *= d
    return p

def shared_to_symbolic(shared):
    value = shared.get_value(borrow=True)
    if value.ndim == 2: return T.matrix(name=shared.name, dtype=value.dtype)
    elif value.ndim == 1: return T.vector(name=shared.name, dtype=value.dtype)
    else: raise NotImplementedError("Higher (> 2) dimensional tensors not implemented")

def batch_call(func, data, batchlen=5000):
    ### assume each row of data is an example
    outdata = None
    nexamples = data.shape[0]

    i = 0
    while True:
        iend = min(i + batchlen, nexamples)
        outval = func(data[i:iend])
        if outdata is None:
            outdata = np.zeros((nexamples, outval.shape[1]), dtype=data.dtype)

        outdata[i:iend] = outval
        i += batchlen
        if i >= nexamples:
            break

    return outdata




# def theano_pack(args, dtype='float64'):
#     """Pack Theano.shared arrays into a Numpy vector"""
#     args = [w.get_value(
#     x_size = sum([w.size for w in args])
#     x = np.empty(x_size, dtype=dtype) # has to be float64 for fmin_l_bfgs_b
#     i = 0
#     for w in args:
#         x[i:i + w.size] = w.flatten()
#         i += w.size
#     return x

# def unpack(x, orig_args, dtype=theano.config.floatX):
#     args = []
#     for

################################################################################
class Autoencoder(CacheObject):
    """
    An autoencoder
      W = weights
      b = visual node biases
      c = hidden node biases
    """
    def __init__(self, visshape, hidshape, tied=True,
                 W=None, V=None, b=None, c=None,
                 f=None, g=None,
                 Wstd=0.1, dtype=theano.config.floatX, seed=None):

        super(Autoencoder, self).__init__()

        self.dtype = dtype
        self.visshape, self.nvis = check_shape(visshape)
        self.hidshape, self.nhid = check_shape(hidshape)
        self.shape = (self.nvis, self.nhid)
        self.train_stats = {}    # dictionary of training statistics

        ### Set up random number generators
        if seed is None:
            self.rng = theano.sandbox.rng_mrg.MRG_RandomStreams()
        else:
            self.rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=seed)
            npr.seed(seed)

        ### initialize functions
        self.f = f if f is not None else Logistic(scale=5.0)
        self.g = g if g is not None else Logistic(scale=5.0)

        ### initialize cost params
        # self.set_train_params(rho=rho, lamb=lamb, noisestd=noisestd)

        ### initialize parameters
        if W is None:
            # W = npr.normal(size=self.shape, scale=Wstd)
            limit = 4*np.sqrt(6. / (self.nvis + self.nhid))
            W = npr.uniform(size=self.shape, low=-limit, high=limit)
        # if V is None and not tied:
        #     V = np.zeros(W.T.shape, dtype=dtype)
        if b is None:
            b = np.zeros(self.nvis, dtype=dtype)
        if c is None:
            c = np.zeros(self.nhid, dtype=dtype)

        assert W.shape[0] == self.nvis and W.shape[1] == self.nhid
        assert b.shape[0] == self.nvis
        assert c.shape[0] == self.nhid

        # self.tied = tied
        self.W = theano.shared(name='W', value=np.asarray(W, dtype=dtype))
        # self.V = self.W.T
        self.V = theano.shared(name='V', value=np.asarray(V, dtype=dtype)) \
            if not tied else self.W.T
        self.b = theano.shared(name='b', value=np.asarray(b, dtype=dtype))
        self.c = theano.shared(name='c', value=np.asarray(c, dtype=dtype))

        self.params = [self.W, self.b, self.c]
        if not tied: self.params.append(self.V)

        self.param_masks = {}

        ### initialize parameter increments
        # self.Winc = theano.shared(name='Winc',
        #                           value=np.zeros((nvis,nhid), dtype=dtype))
        # self.binc = theano.shared(name='binc', value=np.zeros(nvis, dtype=dtype))
        # self.cinc = theano.shared(name='cinc', value=np.zeros(nhid, dtype=dtype))
        # self.increments = [self.Winc, self.binc, self.cinc]

    def __getstate__(self):
        d = dict(self.__dict__)
        for name in ['W', 'b', 'c']:
            d[name] = d[name].get_value()
        d['V'] = d['V'].get_value() if not self.tied else None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name in ['W', 'b', 'c']:
            self.__dict__[name] = theano.shared(name=name, value=state[name])
        self.params = [self.W, self.b, self.c]
        if not self.tied:
            self.V = theano.shared(name='V', value=state['V'])
            self.params.append(self.V)
        else:
            self.V = self.W.T

    @property
    def filters(self):
        filters = self.W.get_value(borrow=True).T
        return filters.reshape((self.nhid,) + self.visshape)

    @property
    def tied(self):
        return not (self.V in self.params)

    def untie(self):
        if self.tied:
            V = self.W.get_value(borrow=False).T
            self.V = theano.shared(name='V', value=np.asarray(V, dtype=self.dtype))
            self.params.append(self.V)
            self._cache.clear()
        else:
            raise Exception("Autoencoder is already untied")

    def propup(self, vis, noisestd=0.0):
        a = T.dot(vis, self.W) + self.c
        if noisestd > 0.0:
            a += self.rng.normal(size=a.shape, std=noisestd)
        return a, self.f.theano(a)

    def propdown(self, hid, noisestd=0.0):
        a = T.dot(hid, self.V) + self.b
        if noisestd > 0.0:
            a += self.rng.normal(size=a.shape, std=noisestd)
        return a, self.g.theano(a)

    def propVHV(self, x, noisestd=0.0):
        ha, h = self.propup(x, noisestd=noisestd)
        ya, y = self.propdown(h)
        return ha, h, ya, y

    def compup(self, images):
        assert images.shape[-2:] == self.visshape
        shape = images.shape[:-2]
        images = images.reshape((numel_shape(shape), self.nvis))

        if not self._cache.has_key('compup'):
            v = T.matrix('v', dtype=self.dtype)
            ha, h = self.propup(v)
            self._cache['compup'] = theano.function([v], h, allow_input_downcast=True)

        h = batch_call(self._cache['compup'], images)
        return h.reshape(shape + self.hidshape)

    def compVHV(self, images):
        assert images.shape[-2:] == self.visshape
        shape = images.shape[:-2]
        images = images.reshape((numel_shape(shape), self.nvis))

        if not self._cache.has_key('compVHV'):
            x = T.matrix('x', dtype=self.dtype)
            ha, h, ya, y = self.propVHV(x)
            self._cache['compVHV'] = theano.function([x], y, allow_input_downcast=True)

        y = batch_call(self._cache['compVHV'], images)
        return y.reshape(shape + self.visshape)

    def compVHVraw(self, rawimages):
        if not self._cache.has_key('compVHVraw'):
            x = T.matrix('x', dtype=self.dtype)
            ha, h, ya, y = self.propVHV(x)
            self._cache['compVHVraw'] = theano.function([x], [ha, h, ya, y], allow_input_downcast=True)
        return self._cache['compVHVraw'](rawimages)
        return batch_call(self._cache['compVHVraw'], rawimages, batchlen=2000)


class SparseAutoencoder(Autoencoder):
    def __init__(self, rfshape=(7,7), mask=None, **kwargs):
        super(SparseAutoencoder, self).__init__(**kwargs)

        self.rfshape = rfshape
        M, N = self.visshape
        m, n = self.rfshape

        if mask is None:
            # find positions of top-left corner
            a = npr.randint(low=0, high=M-m+1, size=self.nhid)
            b = npr.randint(low=0, high=N-n+1, size=self.nhid)

            mask = np.zeros((M, N, self.nhid), dtype='bool')
            for i in xrange(self.nhid):
                mask[a[i]:a[i]+m, b[i]:b[i]+n, i] = True

            mask = mask.reshape((self.nvis, self.nhid))

        self.mask = mask
        # self.param_masks[self.W] = T.cast(self.mask, dtype='int8')
        self.param_masks[self.W] = self.mask
        self.W.set_value(self.W.get_value() * mask)

        # if not self.tied:
        #     self.param_masks[self.V] = T.cast(self.mask.T, dtype='int8')
        #     self.V.set_value(self.V.get_value() * mask.T)

    # def __getstate__(self):

    @property
    def filters(self):
        filters = self.W.get_value(borrow=True).T[self.param_masks[self.W].T]
        return filters.reshape((self.nhid,) + self.rfshape)

    def untie(self):
        super(SparseAutoencoder, self).untie()
        self.param_masks[self.V] = self.param_masks[self.W].T
