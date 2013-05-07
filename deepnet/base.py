
import cPickle as pickle

import theano
import theano.tensor as T

class CacheObject(object):
    """
    A object that can be pickled, with a cache that is not pickled
    """
    def __init__(self):
        self._cache = {}

    def __getstate__(self):
        self._cache.clear()
        return self.__dict__

    def tofile(self, filename):
        f = file(filename, 'w')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def fromfile(filename):
        f = file(filename, 'r')
        obj = pickle.load(f)
        f.close()
        return obj

# def tofile(obj, filename):
#     f = file(filename, 'w')
#     pickle.dump(obj, f)

# def fromfile(filename):
#     f = file(filename, 'r')
#     return pickle.load(f)

