
import numpy as np
import numpy.random

import theano
import theano.tensor as T

from .trainer import Trainer

class DecorrelateTrainer(Trainer):
    def __init__(self, network, **train_hypers):
        super(DecorrelateTrainer, self).__init__(network)

        self.train_hypers.update(train_hypers)

        ### initialize training parameters
        q = np.zeros(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_cost_grads_updates(self, x):

        ha, h = self.network.propup(x, noisestd=self.train_hypers['noise_std'])
        q = 0.9*self.q + 0.1*h.mean(axis=0)

        ### get correlation matrix for examples
        # C = T.dot(x.T, h) / x.shape[0]
        x_std = x.std(axis=0)
        h_std = h.std(axis=0)
        xz = (x - x.mean(0)) / (x.std(0) + 1e-2)
        hz = (h - h.mean(0)) / (h.std(0) + 1e-2)
        # C = T.dot(xz.T, hz) / x.shape[0]
        C = T.dot(xz.T, hz)

        lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        rho = T.cast(self.train_hypers['rho'], self.dtype)
        # cost = (C**2).sum() + lamb*(T.abs_(q - rho)).sum()
        # cost = (C**2).sum() / x.shape[0]**2 + lamb*(T.abs_(q - rho)).sum()
        cost = (C**2).sum() / x.shape[0]**2 + lamb*((q - rho)**2).sum()

        # lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        # rho = T.cast(self.train_hypers['rho'], self.dtype)
        # cost = ((x - y)**2).mean(axis=0).sum() + lamb*(T.abs_(q - rho)).sum()

        updates = {self.q: q}
        return cost, self.grads(cost), updates

class DecorrelateVTrainer(Trainer):
    def __init__(self, network, **train_hypers):
        super(DecorrelateTrainer, self).__init__(network)

        self.train_hypers.update(train_hypers)

        ### initialize training parameters
        q = np.zeros(self.network.nhid, dtype=self.dtype)
        self.q = theano.shared(name='q', value=q)

        self.train_params = [q]

    def get_cost_grads_updates(self, x):

        ha, h = self.network.propup(x, noisestd=self.train_hypers['noise_std'])
        ya, y = self.network.propdown(h)
        q = 0.9*self.q + 0.1*h.mean(axis=0)

        ### get correlation matrix for examples
        # C = T.dot(x.T, h) / x.shape[0]
        x_std = x.std(axis=0)
        h_std = h.std(axis=0)
        xz = (x - x.mean(0)) / (x.std(0) + 1e-2)
        hz = (h - h.mean(0)) / (h.std(0) + 1e-2)
        # C = T.dot(xz.T, hz) / x.shape[0]
        C = T.dot(xz.T, hz)

        lamb = T.cast(self.train_hypers['lamb'], self.dtype)
        rho = T.cast(self.train_hypers['rho'], self.dtype)
        # cost = (C**2).sum() + lamb*(T.abs_(q - rho)).sum()
        # cost = (C**2).sum() / x.shape[0]**2 + lamb*(T.abs_(q - rho)).sum()
        # cost = (C**2).sum() / x.shape[0]**2 + lamb*((q - rho)**2).sum()

        cost_up = (C**2).sum() / x.shape[0]**2 + lamb*((q - rho)**2).sum()
        cost_down = ((x - y)**2).mean(0).sum()
        cost = cost_up + cost_down

        ### get grads
        grads = {}

        params_up = [self.network.W, self.network.c]
        grads_up = T.grad(cost_up, params_up)
        grads.update(zip(params_up, grads_up))

        params_down = [self.network.V, self.network.b]
        grads_down = T.grad(cost_down, params_down, consider_constant=params_up)
        grads.update(zip(params_down, grads_down))

        for p, g in grads.items():
            g = T.switch(T.isnan(g), 0.0, g)
            if p in self.param_masks:
                g = g * self.param_masks[p]

            grads[p] = g

        updates = {self.q: q}
        return cost, grads, updates
