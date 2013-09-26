
import os

import numpy as np
import numpy.random as npr

def rmse(x, y):
    assert x.shape == y.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[:-2] + (-1,))
        y = y.reshape(y.shape[:-2] + (-1,))
    return np.sqrt(((x - y)**2).mean(axis=-1))
