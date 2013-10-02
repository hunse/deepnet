
import collections
import time

import numpy as np
import numpy.random

import theano
import theano.tensor as T

from .. import image_tools as imtools

# def sgd_minibatch_fn(trainer, rate, clip=None):
#     x = T.matrix('x', dtype=trainer.dtype)
#     cost, ups = trainer.get_cost_updates(x)

#     grads = trainer.grads(cost)

#     updates = collections.OrderedDict(ups)

#     rate = T.cast(rate, dtype=trainer.dtype)
#     for param, grad in zip(trainer.params, grads):
#         if clip is not None:
#             grad = grad.clip(*clip)
#         updates[param] = param - rate*grad

#     return theano.function([x], cost, updates=updates,
#                            allow_input_downcast=True)

def sgd_minibatch_fn(trainer, rate, clip=None):
    x = T.matrix('x', dtype=trainer.dtype)
    cost, grads, updates = trainer.get_cost_grads_updates(x)

    rate = T.cast(rate, dtype=trainer.dtype)
    for param in trainer.params:
        grad = grads[param]
        if clip is not None:
            grad = grad.clip(*clip)
        updates[param] = param - rate*grad

    return theano.function([x], cost, updates=updates.items(),
                           allow_input_downcast=True)

def sgd(trainer, images, timages=None, test_fn=None,
        n_epochs=30, rate=0.05, clip=(-1,1),
        show=imtools.display_available(), vlims=None,
        save_fn=None):
    """
    Unsupervised training using Stochasitc Gradient Descent (SGD)
    """

    if timages is None:
        timages = images[:500]

    print "Performing SGD on a %d x %d autoencoder for %d epochs"\
        % (trainer.network.nvis, trainer.network.nhid, n_epochs)
    print "SGD params: %s" % dict(n_epochs=n_epochs, rate=rate, clip=clip)
    print "Trainer params: %s" % trainer.train_hypers

    ### create minibatch learning function
    train = sgd_minibatch_fn(trainer, rate=rate, clip=clip)

    exshape = images.shape[:-2]
    imshape = images.shape[-2:]
    if len(exshape) == 1:
        ### split into batches
        batchlen = 100
        batches = images.reshape((-1, batchlen, np.prod(imshape)))
    elif len(exshape) == 2:
        ### already in batches, so just collapse the shape
        batches = images.reshape(exshape + (np.prod(imshape),))
    else:
        raise ValueError("Invalid input image shape %s" % images.shape)

    stats = dict(algorithm='sgd', n_epochs=0, cost=[],
                 hypers=dict(trainer.train_hypers))
    trainer.network.train_stats.append(stats)

    for epoch in xrange(n_epochs):
        # rate = rates[epoch] if epoch < len(rates) else rates[-1]
        cost = 0

        t = time.time()
        for batch in batches:
            cost += train(batch)
        t = time.time() - t

        test_stats = test(trainer, timages,
                          test_fn=test_fn, show=show, fignum=101, vlims=vlims)

        stats['n_epochs'] += 1
        stats['cost'].append(cost)
        for k, v in test_stats.items():
            if k not in stats: stats[k] = []
            stats[k].append(v)

        print "Epoch %d finished, t = %0.2f s, cost = %0.3e, %s" \
            % (epoch, t, cost, str(["%s = %0.2e" % (k,v) for k,v in test_stats.items()]))

        if save_fn is not None:
            save_fn()
            # trainer.network.to_file(save_name)

    return stats

def test(trainer, timages, test_fn=None,
         show=imtools.display_available(), fignum=None, vlims=None):
    # from ..image_tools import *
    # from ..image_tools import compare, activations, filters
    from .. import image_tools as imtools

    if test_fn is None:
        test_fn = trainer.network.compVHVraw

    imshape = timages.shape[1:]
    ims_shape = (-1,) + imshape
    x = timages.reshape((len(timages), -1))
    ha, h, ya, y = test_fn(x)

    rmse = imtools.rmse(x, y).mean()
    act = h.mean()
    # test_stats = {'rmse': rmse, 'hidact': act}
    test_stats = collections.OrderedDict(
        rmse=rmse, hid_mean=h.mean(), hid_min=h.min(), hid_max=h.max())

    ### Show current results
    if show:
        import matplotlib.pyplot as plt

        # image_tools.figure(fignum=fignum, figsize=(12,12))
        plt.figure(fignum)
        plt.clf()
        rows, cols = 3, 2

        ax = plt.subplot(rows, 1, 1)
        imtools.compare([x.reshape(ims_shape), y.reshape(ims_shape)], vlims=vlims)

        ax = plt.subplot(rows, cols, cols+1)
        imtools.activations(ha, trainer.network.f.eval)

        ax = plt.subplot(rows, cols, cols+2)
        imtools.activations(ya, trainer.network.g.eval)

        ax = plt.subplot(rows, 1, 3)
        imtools.filters(trainer.network.filters, rows=8, cols=16)

        plt.tight_layout()
        plt.draw()

    return test_stats


# def lbfgs(self, images, nevals=10):
#     """
#     Unsupervised training using limited-memory BFGS (L-BFGS)
#     """

#     print "Performing L-BFGS on a %d x %d autoencoder for %d function evals"\
#         % (self.nvis, self.nhid, nevals)
#     print self.get_train_params()
#     print_gpu_memory()

#     test_stats = self.test(images.subset(1000), show=True, fignum=101)
#     print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]

#     images.batchlen = 5000
#     params_view = [p.get_value(borrow=False) for p in self.params]
#     p0 = np.hstack([p.flatten() for p in params_view]).astype('float64')

#     ### make Theano function
#     x = T.matrix('x', dtype=self.dtype)
#     s_p = T.vector(dtype=self.dtype)
#     s_args = []
#     i = 0
#     for w in self.params:
#         s_pi = s_p[i: i + w.size].reshape(w.shape)
#         # g_pi = theano.sandbox.cuda.gpu_from_host(s_pi)
#         s_args.append( s_pi )
#         i += w.size

#     ha, h, ya, y = self.propVHV(x, noisestd=self.noisestd)
#     cost = self.cost(x, h.mean(axis=0), y)
#     # grads = T.grad(cost, self.params)
#     grads = self.getGrads(cost)

#     # def host_from_gpu(x):
#     #     if isinstance(x.type, theano.tensor.TensorType):
#     #         return x
#     #     else:
#     #         return theano.sandbox.cuda.host_from_gpu(x)
#     # grads = map(host_from_gpu, grads)

#     f_df = theano.function([x, s_p], [cost] + grads, givens=zip(self.params, s_args))

#     # def f_df_cast(p):
#     #     # cost, grads = f_df(p.astype('float32'))
#     #     outs = f_df(p.astype(self.dtype))
#     #     cost, grads = outs[0], outs[1:]
#     #     grad = np.hstack([g.flatten() for g in grads])
#     #     return cost.astype('float64'), grad.astype('float64')

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
#             # grad = np.hstack([g.flatten() for g in grads])
#         return cost.astype('float64'), grad.astype('float64')

#     t = time.time()
#     p_opt, mincost, info_dct = sp.optimize.lbfgsb.fmin_l_bfgs_b(
#         f_df_cast, p0, maxfun=nevals, iprint=1)
#     t = time.time() - t

#     i = 0
#     # params = [p.get_value(borrow=True) for p in self.params]
#     for w, v in zip(self.params, params_view):
#         w.set_value(p_opt[i: i + v.size].reshape(v.shape).astype(self.dtype), borrow=False)
#         i += v.size

#     # x.set_value(np.zeros((0,0), dtype=self.dtype))

#     print "Done. t = %0.3f s, cost = %0.2e" % (t, mincost)
#     print_gpu_memory()

#     test_stats = self.test(images.subset(1000), show=True, fignum=101)
#     print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]
