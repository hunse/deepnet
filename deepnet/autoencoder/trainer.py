
import collections
import time

import numpy as np
import numpy.random

import theano
import theano.tensor as T

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

    # def get_activs(self, x):
    #     raise NotImplementedError(
    #         "Trainer child must implement the 'get_activs' function")

    # def cost(self, x, y):
    #     raise NotImplementedError(
            # "Trainer child must implement the 'cost' function")

    # def state_updates(self):
    #     pass # children can update states here

    def grads(self, cost):
        grad_dict = {}

        grads = T.grad(cost, self.params)
        for param, grad in zip(self.params, grads):
            grad = T.switch(T.isnan(grad), 0.0, grad)
            if param in self.param_masks:
                grad = grad * self.param_masks[param]
            grad_dict[param] = grad

        return grad_dict

    def get_cost_grads_updates(self, x):
        raise NotImplementedError(
            "Trainer child must implement the 'get_cost_grads_updates' function")


class SparseTrainer(Trainer):
    """Train an autoencoder with sparse hidden-node activation"""

    def __init__(self, network, **train_hypers):
        super(SparseTrainer, self).__init__(network)

        # self.train_hypers = dict(train_hypers)
        self.train_hypers.update(train_hypers)

        ### initialize training parameters
        q = np.nan * np.ones(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_cost_grads_updates(self, x):
        ha, h, ya, y = self.network.propVHV(x, noise_std=self.train_hypers['noise_std'])

        q = T.switch(T.isnan(self.q), h.mean(axis=0),
                     0.9*self.q + 0.1*h.mean(axis=0))

        lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        rho = T.cast(self.train_hypers['rho'], self.dtype)
        cost = ((x - y)**2).mean(axis=0).sum() + lamb*(T.abs_(q - rho)).sum()

        updates = {self.q: q}
        return cost, self.grads(cost), updates


class DoubleSparseTrainer(Trainer):
    def __init__(self, network, **train_hypers):
        super(DoubleSparseTrainer, self).__init__(network)

        # self.train_hypers = dict(train_hypers)
        self.train_hypers.update(train_hypers)

        ### initialize training parameters
        q = np.zeros(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_cost_grads_updates(self, x):
        ha, h, ya, y = self.network.propVHV(x, noise_std=self.train_hypers['noise_std'])

        # q = 0.9*self.q + 0.1*h.mean(axis=0)
        q = 0.7*self.q + 0.3*h.mean(axis=0)

        p = dict((k, T.cast(v, self.dtype)) for k, v in self.train_hypers.items())

        mse = ((x - y)**2).mean(axis=0).sum()
        cost_low = T.exp(-q / p['rho1']).sum()
        cost_high = T.exp(q / p['rho2']).sum()
        cost = mse + p['lamb1']*cost_low + p['lamb2']*cost_high

        updates = {self.q: q}
        return cost, self.grads(cost), updates


class ComplexTrainer(Trainer):
    """Learn complex cells from videos"""

    def __init__(self, network, **train_hypers):
        super(ComplexTrainer, self).__init__(network)

        # self.train_hypers = dict(train_hypers)
        self.train_hypers.update(train_hypers)

        ### initialize training parameters
        q = np.zeros(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_cost_updates(self, x):

        ha, h = self.network.propup(x, noise_std=self.train_hypers['noise_std'])

        ### take diffs across frames
        diffs = T.extra_ops.diff(h, axis=0)

        cost = (diffs**2).mean(0).sum()

        # q = 0.9*self.q + 0.1*h.mean(axis=0)

        # lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        # rho = T.cast(self.train_hypers['rho'], self.dtype)
        # cost = ((x - y)**2).mean(axis=0).sum() + lamb*(T.abs_(q - rho)).sum()

        # updates = [(self.q, q)]
        updates = []
        return cost, updates
