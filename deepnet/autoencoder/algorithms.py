
import collections
import time

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from ..image_tools import *

class Trainer(object):

    def __init__(self, network):
        self.network = network

        self.train_params = []
        self.train_hypers = {}

    @property
    def dtype(self):
        return self.network.dtype

    @property
    def params(self):
        return self.network.params

    @property
    def param_masks(self):
        return self.network.param_masks

    def cost(self, x, y):
        raise NotImplementedError("Trainer children must implement the cost function")

    def state_updates(self):
        pass # children can update states here

    def grads(self, cost):
        grads = T.grad(cost, self.params)

        for i, p in enumerate(self.params):
            grads[i] = T.switch(T.isnan(grads[i]), 0.0, grads[i])
            # grads[i] = grads[i].clip(-1,1)
            # grads[i] = grads[i].clip(-0.01,0.01)
            if p in self.param_masks:
                grads[i] = grads[i] * self.param_masks[p]

        return grads


# class ReconTrainer(Trainer):
#     def cost(self, x, y):

class SparseTrainer(Trainer):
    def __init__(self, network, **train_hypers):
        super(SparseTrainer, self).__init__(network)

        # self.rho = rho
        # self.lamb = lamb
        # self.noise_std = noise_std
        # self.train_hypers = dict(rho=rho, lamb=lamb, noise_std=noise_std]
        self.train_hypers = dict(train_hypers)

        ### initialize training parameters
        q = np.zeros(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_activs(self, x):
        ha, h, ya, y = self.network.propVHV(x, noisestd=self.train_hypers['noise_std'])
        return ha, h, ya, y

    def get_cost_updates(self, x, ha, h, ya, y):
        q = 0.9*self.q + 0.1*h.mean(axis=0)

        lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        rho = T.cast(self.train_hypers['rho'], self.dtype)
        cost = ((x - y)**2).mean(axis=0).sum() + lamb*(T.abs_(q - rho)).sum()

        updates = [(self.q, q)]
        return cost, updates


def sgd_minibatch_fn(trainer, rate, clip=None):
    x = T.matrix('x', dtype=trainer.dtype)
    ha, h, ya, y = trainer.get_activs(x)
    cost, ups = trainer.get_cost_updates(x, ha, h, ya, y)
    grads = trainer.grads(cost)

    updates = collections.OrderedDict(ups)

    rate = T.cast(rate, dtype=trainer.dtype)
    for param, grad in zip(trainer.params, grads):
        if clip is not None:
            grad = grad.clip(*clip)
        updates[param] = param - rate*grad

    # rmse = T.mean(T.sqrt(T.mean((x - y)**2, axis=1)))
    # act = T.mean(h)

    return theano.function([x], cost, updates=updates,
                           allow_input_downcast=True)


def sgd(trainer, images, timages=None, test_fn=None,
        nepochs=30, rate=0.05, clip=(-1,1), show=True, vlims=None):
    """
    Unsupervised training using Stochasitc Gradient Descent (SGD)
    """

    if timages is None:
        timages = images[:500]

    print "Performing SGD on a %d x %d autoencoder for %d epochs"\
        % (trainer.network.nvis, trainer.network.nhid, nepochs)
    print trainer.train_hypers
    # print trainer.get_train_params()

    ### create minibatch learning function
    train = sgd_minibatch_fn(trainer, rate=rate, clip=clip)

    imshape = images.shape[1:]
    batchlen = 100
    batches = images.reshape((-1, batchlen, imshape[0]*imshape[1]))

    # stats = {'rmse': [], 'hidact': []}
    stats = {}
    for epoch in xrange(nepochs):
        # rate = rates[epoch] if epoch < len(rates) else rates[-1]
        cost = 0

        t = time.time()
        for batch in batches:
            cost += train(batch)
        t = time.time() - t

        test_stats = test(trainer, timages,
                          test_fn=test_fn, show=show, fignum=101, vlims=vlims)

        for k, v in test_stats.items():
            if k in stats: stats[k].append(v)
            else: stats[k] = [v]

        print "Epoch %d finished, t = %0.2f s, cost = %0.3e, %s" \
            % (epoch, t, cost, str(["%s = %0.2e" % (k,v) for k,v in test_stats.items()]))

    # trainer.train_stats['pretrain'] = stats
    return stats


# @property
# def pretrained(self):
#     return self.train_stats.has_key('pretrain')

# def plot_filters(ax, filters):
    # image_tools.tile(ax, self.Warray.T, self.visshape,
    #                      rows=rows, cols=cols, vlims=(-2*r,2*r), grid=True)


def test(trainer, timages, test_fn=None, show=True, fignum=None, vlims=None):

    if test_fn is None:
        test_fn = trainer.network.compVHVraw

    imshape = timages.shape[1:]
    ims_shape = (-1,) + imshape
    x = timages.reshape((len(timages), -1))
    ha, h, ya, y = test_fn(x)

    rmse = np.sqrt(((x - y)**2).mean(axis=1)).mean()
    act = h.mean()
    test_stats = {'rmse': rmse, 'hidact': act}

    ### Show current results
    if show:
        # image_tools.figure(fignum=fignum, figsize=(12,12))
        plt.figure(fignum)
        plt.clf()
        rows, cols = 3, 2

        ax = plt.subplot(rows, 1, 1)
        compare([x.reshape(ims_shape), y.reshape(ims_shape)], vlims=vlims)

        ax = plt.subplot(rows, cols, cols+1)
        activations(ha, trainer.network.f.eval)

        ax = plt.subplot(rows, cols, cols+2)
        activations(ya, trainer.network.g.eval)

        ax = plt.subplot(rows, 1, 3)
        filters(trainer.network.filters)

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
