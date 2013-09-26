
import theano
import theano.tensor as T

from ..base import CacheObject

class Function(CacheObject):
    def __init__(self):
        super(Function, self).__init__()

    def theano(self, x):
        """Produce Theano code for the function"""
        raise NotImplementedError("Function must generate Theano code")

    def eval(self, x):
        if 'f' not in self._cache:
            # t = T.dvector()
            t = T.vector()
            self._cache['f'] = theano.function(
                [t], self.theano(t), allow_input_downcast=True)

        ### flatten x and reshape, to deal with general tensors
        return self._cache['f'](x.flatten()).reshape(x.shape)

    def __call__(self, x):
        return self.eval(x)


class Linear(Function):
    """Linear function"""

    def __init__(self, slope=1.0):
        super(Linear, self).__init__()
        self.slope = slope

    def theano(self, x):
        return self.slope * x


class Logistic(Function):
    """Logistic (sigmoid) function"""

    def __init__(self, scale=5.0):
        super(Logistic, self).__init__()
        self.scale = scale

    def theano(self, x):
        return T.nnet.sigmoid(self.scale * x)


class LIF(Function):
    """Leaky integrate-and-fire tuning curve function"""

    def __init__(self, tRef=20e-3, tauRC=60e-3, alpha=10.0, xint=-0.5, amp=1./41):
        super(LIF, self).__init__()
        self.tRef = tRef
        self.tauRC = tauRC
        self.alpha = alpha
        self.xint = xint
        self.amp = amp

    def theano(self, x):
        j = self.alpha*(x - self.xint)
        v = self.amp / (self.tRef + self.tauRC*T.log1p(1./j))
        return T.switch(j > 0, v, 0.0)


# class NoisyLIFApprox(LIF):
#     """Imitation noisy LIF tuning curve function

#     NOTE: sigma has not been calibrated to reflect actual noise levels
#     """

#     def __init__(self, sigma=0.05, **kwargs):
#         super(NoisyLIFApprox, self).__init__(**kwargs)
#         self.sigma = sigma

#     def theano(self, x):
#         j = self.sigma*T.log1p(T.exp(self.alpha*(x - self.xint)/self.sigma))
#         v = self.amp / (self.tRef + self.tauRC*T.log1p(1./j))
#         return T.switch(j > 0, v, 0.0)

class NoisyLIFApprox(LIF):
    """Imitation noisy LIF tuning curve function

    NOTE: sigma has not been calibrated to reflect actual noise levels
    """

    def __init__(self, sigma=0.05, **kwargs):
        super(NoisyLIFApprox, self).__init__(**kwargs)
        self.sigma = sigma

    def theano(self, x):
        dtype = theano.config.floatX
        alpha = T.cast(self.alpha, dtype=dtype)
        xint = T.cast(self.xint, dtype=dtype)
        sigma = T.cast(self.sigma, dtype=dtype)
        t_ref = T.cast(self.tRef, dtype=dtype)
        tau_rc = T.cast(self.tauRC, dtype=dtype)
        amp = T.cast(self.amp, dtype=dtype)

        j = sigma*T.log1p(T.exp(alpha*(x - xint)/sigma))
        v = amp / (t_ref + tau_rc*T.log1p(1./j))
        return T.switch(j > 0, v, 0.0)



# class NoisyLIFApprox2(LIF):
#     """Imitation noisy LIF tuning curve function

#     NOTE: sigma has not been calibrated to reflect actual noise levels
#     """

#     def __init__(self, sigma=0.05, **kwargs):
#         super(NoisyLIFApprox2, self).__init__(**kwargs)
#         self.sigma = sigma

#     def theano(self, x):
#         j = self.sigma*T.log1p(T.exp(self.alpha*(x - self.xint)/self.sigma))
#         v = self.amp / (self.tRef + self.tauRC*T.log1p(1./j))

#         ### THIS DOESN'T WORK. THEANO SEEMS TO OPTIMIZE IT WEIRDLY, COMPARED
#         ### WITH RETRUNING v ONLY IF j > 0
#         return v

################################################################################

def file_test():
    import os
    import tempfile
    import cPickle as pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from util.timing import tic, toc

    tic("Creation")
    func = NoisyLIFApprox(amp=5)
    toc()

    tic("Writing")
    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    pickle.dump(func, f)
    f.close()
    del func
    toc()

    tic("Loading")
    f = file(fname, 'r')
    func = pickle.load(f)
    toc()

    tic("Evaluation")
    radius = 2.0
    x = np.linspace(-radius,radius,101)
    y = func(x)
    toc()

    tic("Evaluation 2")
    y2 = func.eval(x)
    toc()

    plt.figure(1)
    plt.clf()
    plt.plot(x, y)
    plt.show()
    

if __name__ == '__main__':
    file_test()
