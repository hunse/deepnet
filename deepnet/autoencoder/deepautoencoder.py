
import collections
import os
import sys
import time

import numpy as np
import numpy.random as npr
import scipy as sp

import matplotlib.pyplot as plt

# gpuflags = 'device=gpu,floatX=float32'
# os.environ['THEANO_FLAGS'] = gpuflags

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from ..base import CacheObject
from . import autoencoder
from .. import image_tools
from .. import functions

################################################################################
class DeepAutoencoder(CacheObject):
    def __init__(self, layers):
        super(DeepAutoencoder, self).__init__()
        self.layers = layers
        # self.nlayers = len(self.layers)
        self.dtype = layers[0].dtype
        self.train_stats = {}

        self._init_params()

    def _init_params(self):
        self.params = []
        self.param_masks = {}
        for layer in self.layers:
            self.params.extend(layer.params)
            for param in layer.params:
                if param in layer.param_masks:
                    self.param_masks[param] = layer.param_masks[param]

    def __getstate__(self):
        self._cache.clear()
        self.params = []
        self.param_masks.clear()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_params()

    def untie(self):
        for layer in self.layers:
            layer.untie()
        self._cache.clear()

    def get_params_view(self):
        return [p.get_value(borrow=True) for p in self.params]

    # def tofile(self, filename):
    #     np.savez(filename, [layer.todict() for layer in self.layers])

    # def tocsvs(self, fileroot):
    #     for i in xrange(self.nlayers):
    #         name = fileroot + ("layer%d" % i)
    #         self.layers[i].tocsvs(name)

    # @classmethod
    # def fromfile(cls, filename):
    #     lst = np.load(filename)['arr_0']
    #     layers = [autoencoder.Autoencoder.fromdict(d) for d in lst]
    #     return cls(layers)

    def propVHV(self, x, maxlayer=-1, noise_std=None):
        """ Propagate input to top of net and back down """
        if noise_std is not None:
            raise NotImplementedError()
        if maxlayer < 0:
            maxlayer += len(self.layers)

        ha = [None] * (maxlayer + 1)
        h =  [None] * (maxlayer + 1)
        va = [None] * (maxlayer + 1)
        v =  [None] * (maxlayer + 1)

        for i in xrange(0, maxlayer + 1):
            layer = self.layers[i]
            ha[i], h[i] = layer.propup(x)
            x = h[i]

        for i in xrange(maxlayer, -1, -1):
            layer = self.layers[i]
            # noisestd = layer.noisestd if i > 0 else 0.0
            va[i], v[i] = layer.propdown(x)
            x = v[i]

        return ha, h, va, v

    def propVHV_fn(self, **kwargs):
        x = T.matrix('x', dtype=self.dtype)
        ha, h, va, v = self.propVHV(x, **kwargs)
        return theano.function(
            [x], [ha[-1], h[-1], va[0], v[0]], allow_input_downcast=True)

    def compup(self, images, maxlayer=-1):
        visshape = self.layers[0].visshape
        assert images.shape[-2:] == visshape
        shape = images.shape[:-2]
        images = images.reshape((np.prod(shape), np.prod(visshape)))

        key = 'compup%d' % maxlayer
        if key not in self._cache:
            x = T.matrix('x', dtype=self.dtype)
            ha, h, va, v = self.propVHV(x, maxlayer=maxlayer)
            self._cache[key] = theano.function(
                [x], h[-1], allow_input_downcast=True)
        y = autoencoder.batch_call(self._cache[key], images, batchlen=2000)
        # return imageset.imageset_like(newimages, shape=self.layers[-1].hidshape)
        return y.reshape(shape + self.layers[maxlayer].hidshape)

    def compVHV(self, images):
        visshape = self.layers[0].visshape
        assert images.shape[-2:] == visshape
        shape = images.shape[:-2]
        images = images.reshape((np.prod(shape), np.prod(visshape)))

        if not self._cache.has_key('compVHV'):
            x = T.matrix('x', dtype=self.dtype)
            ha, h, va, v = self.propVHV(x)
            self._cache['compVHV'] = theano.function(
                [x], v[0], allow_input_downcast=True)
        y = autoencoder.batch_call(
            self._cache['compVHV'], images, batchlen=2000)
        return y.reshape(shape + visshape)

    def compVHVraw(self, flatimages):
        if not self._cache.has_key('compVHVraw'):
            x = T.matrix('x', dtype=self.dtype)
            ha, h, va, v = self.propVHV(x)
            self._cache['compVHVraw'] = theano.function(
                [x], ha + h + va + v, allow_input_downcast=True)
        return self._cache['compVHVraw'](flatimages)
        # return autoencoder.batch_call(self._cache['compVHVraw'], rawimages, batchlen=2000)

    # def train_layers(self, images, timages=None, train_fn=None, params=None,
    #                  force_retrain=False, savename=None):

    #     if timages is None: timages = images.subset(500)
    #     if params is None: params = [{}]*self.nlayers

    #     for i in xrange(self.nlayers):
    #         layer = self.layers[i]
    #         param = params[i]

    #         if 'test_fn' not in param:
    #             test_fn = self.propVHV_fn(maxlayer=i)
    #             param['test_fn'] = test_fn
    #         param['images'] = images
    #         param['timages'] = timages

    #         if force_retrain or not layer.pretrained:
    #             # layer.set_train_params(**params[i])
    #             autoencoder.print_gpu_memory()
    #             train_fn(layer, param)
    #             # layer.sgd(images, timages, test_fn=test, **sgd_params[i])
    #             autoencoder.print_gpu_memory()
    #             if savename is not None:
    #                 self.to_file(savename)

    #         images = layer.compup(images)       # propagate images up the layer

    # def sgd_layers(self, images, timages, sgd_params=None, force_retrain=False, savename=None):
    #     if sgd_params is None:
    #         params = [{}]*self.nlayers

    #     # for layer, param in zip(self.layers, params):
    #     # curimages = images
    #     for i in xrange(self.nlayers):
    #         layer = self.layers[i]
    #         test = self.propVHV_fn(maxlayer=i)

    #         if force_retrain or not layer.pretrained:
    #             # layer.set_train_params(**params[i])
    #             autoencoder.print_gpu_memory()
    #             layer.sgd(images, timages, test_fn=test, **sgd_params[i])
    #             autoencoder.print_gpu_memory()
    #             if savename is not None: self.tofile(savename)

    #         images = layer.compup(images)       # propagate images up the layer

    # def cost(self, x, q, y):
    #     return T.sum(T.mean((x - y)**2, axis=0))
    #     # return T.sum(T.mean((x - y)**2, axis=0)) + T.sum((T.mean(h, axis=0) - 0.05)**2)
    #     # return T.sum(T.mean((x - y)**2, axis=0)) + lamb*T.sum((q - rho)**2)

    # def getGrads(self, cost):
    #     grads = T.grad(cost, self.params)

    #     for i in xrange(len(self.params)):
    #         grads[i] = T.switch(T.isnan(grads[i]), 0.0, grads[i])
    #         if self.params[i] in self.param_masks:
    #             grads[i] = grads[i] * self.param_masks[self.params[i]]

    #     return grads

    # def getCostGradsUpdates(self, x, rate):
    #     ha, h, ya, y = self.propVHV(x)
    #     # q = 0.9*self.q + 0.1*T.mean(h, axis=0)
    #     cost = self.cost(x, h[-1], y[0])
    #     grads = self.getGrads(cost)

    #     updates = collections.OrderedDict()
    #     rate = T.cast(rate, dtype=self.dtype)
    #     for param, grad in zip(self.params, grads):
    #         updates[param] = param - rate*grad

    #     rmse = T.mean(T.sqrt(T.mean((x - y[0])**2, axis=1)))
    #     act = T.mean(h[0])
    #     measures = [rmse, act]

    #     return cost, measures, grads, updates

    # def test(self, timages, test_fn=None, show=True, fignum=None):

    #     if test_fn is None:
    #         test_fn = self.compVHVraw

    #     x = timages.images
    #     outs = test_fn(x)
    #     n = self.nlayers
    #     ha, h, ya, y = outs[:n], outs[n:2*n], outs[2*n:3*n], outs[3*n:]

    #     rmse = np.sqrt(((x - y[0])**2).mean(axis=1)).mean()
    #     act = h[-1].mean()
    #     test_stats = {'rmse': rmse, 'hidact': act}

    #     ### Show current results
    #     if show:
    #         image_tools.figure(fignum=fignum, figsize=(12,12))
    #         rows = 4
    #         cols = self.nlayers

    #         ax = plt.subplot(rows, 1, 1)
    #         image_tools.compare(ax, [x, y[0]], timages.shape, vlims=timages.vlims)

    #         for i in xrange(cols):
    #             layer = self.layers[i]

    #             ax = plt.subplot(rows, cols, cols+i+1)
    #             image_tools.activations(ax, ha[i], layer.compF)

    #             ax = plt.subplot(rows, cols, 2*cols+i+1)
    #             image_tools.activations(ax, ya[i], layer.compG)

    #             ax = plt.subplot(rows, cols, 3*cols+i+1)
    #             layer.plotweights(ax)

    #         plt.tight_layout()
    #         plt.draw()

    #     return test_stats

    # def sgd(self, images, timages=None, nepochs=30, rate=0.05, show=True):
    #     """
    #     Perform Stochastic Gradient Descent on the whole network as one unit
    #     """

    #     print "Performing SGD on a %s x %s DeepAutoencoder for %d epochs"\
    #         % (str(self.layers[0].visshape), str(self.layers[-1].hidshape), nepochs)

    #     if timages is None:
    #         timages = images.subset(500)

    #     ### Create train function
    #     s_data = T.matrix('data', dtype=self.dtype)
    #     _, measures, _, updates = self.getCostGradsUpdates(s_data, rate=rate)
    #     train = theano.function([s_data], measures, updates=updates, allow_input_downcast=True)

    #     stats = {}
    #     for epoch in xrange(nepochs):
    #         t = time.time()
    #         for i in xrange(images.nbatches):
    #             train(images.batch(i))
    #         t = time.time() - t

    #         test_stats = self.test(timages, show=True, fignum=101)

    #         for k, v in test_stats.items():
    #             if k in stats: stats[k].append(v)
    #             else: stats[k] = [v]

    #         print "Epoch %d finished, t = %0.2f s, %s" \
    #             % (epoch, t, str(["%s = %0.2e" % (k,v) for k,v in test_stats.items()]))

    #     self.train_stats['pretrain'] = stats

    # # def pack_params(self, params):
    # #     size = sum([p.size for p in params])
    # #     x = np.zeros(size, dtype=params[0].dtype)

    # #     i = 0
    # #     for p in params:
    # #         x[i : i + p.size] = p.flatten()
    # #         i += p.size

    # #     return x
    # #     # return np.hstack([p.flatten() for p in params])

    # # def unpack_params(self, x):
    # #     unpacked = []
    # #     for layer in self.layers:


    # #     i = 0
    # #     for w in self.params:
    # #         if w in self.param_masks:
    # #             mask[i: i + w.size] = self.param_masks[w]

    # #         s_pi = s_p[i: i + w.size].reshape(w.shape)
    # #         s_args.append(s_pi)
    # #         i += w.size
    # #     return package

    # def lbfgs(self, images, nevals=30):
    #     """
    #     Unsupervised training using limited-memory BFGS (L-BFGS)
    #     """

    #     # nevals = 30
    #     # nevals = 2

    #     print "Performing L-BFGS on a %s x %s DeepAutoencoder for %d function evals"\
    #         % (str(self.layers[0].visshape), str(self.layers[-1].hidshape), nevals)
    #     # print self.get_train_params()
    #     autoencoder.print_gpu_memory()

    #     test_stats = self.test(images.subset(1000), show=True, fignum=101)
    #     print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]

    #     # images.batchlen = 5000
    #     images.batchlen = 2000
    #     params_view = self.get_params_view()
    #     p0 = np.hstack([p.flatten() for p in params_view]).astype('float64')

    #     ### make Theano function
    #     x = T.matrix('x', dtype=self.dtype)
    #     s_p = T.vector(dtype=self.dtype)
    #     s_args = []
    #     i = 0
    #     for w in self.params:
    #         s_pi = s_p[i: i + w.size].reshape(w.shape)
    #         s_args.append( s_pi )
    #         i += w.size

    #     ha, h, ya, y = self.propVHV(x)
    #     cost = self.cost(x, h[-1], y[0])
    #     grads = self.getGrads(cost)

    #     f_df = theano.function([x, s_p], [cost] + grads, givens=zip(self.params, s_args))

    #     def f_df_cast(p):
    #         cost = 0
    #         grad = np.zeros_like(p0)
    #         for i in xrange(images.nbatches):
    #             outs = f_df(images.batch(i).astype(self.dtype), p.astype(self.dtype))
    #             icost, igrads = outs[0], outs[1:]

    #             cost += icost
    #             j = 0
    #             for g in igrads:
    #                 grad[j: j + g.size] += g.flatten()
    #                 j += g.size

    #         return cost.astype('float64'), grad.astype('float64')


    #     t = time.time()
    #     p_opt, mincost, info_dct = sp.optimize.lbfgsb.fmin_l_bfgs_b(
    #         f_df_cast, p0, maxfun=nevals, iprint=1)
    #     t = time.time() - t

    #     i = 0
    #     for w, v in zip(self.params, params_view):
    #         w.set_value(p_opt[i: i + v.size].reshape(v.shape).astype(self.dtype), borrow=False)
    #         i += v.size

    #     print "Done. t = %0.3f s, cost = %0.2e" % (t, mincost)
    #     autoencoder.print_gpu_memory()

    #     test_stats = self.test(images.subset(1000), show=True, fignum=101)
    #     print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]

    # def simulate(self, X):
    #     raise NotImplementedError()
