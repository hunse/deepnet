
import numpy as np

# import ipdb

class CacheObject(object):
    """
    A object that can be saved to file, with a cache that is not saved
    """
    def __init__(self):
        self._cache = {}

    def __getstate__(self):
        self._cache.clear()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def to_file(self, file_name):
        d = {}
        d['__class__'] = self.__class__
        d['__dict__'] = self.__getstate__()
        np.savez(file_name, **d)

    @staticmethod
    def from_file(file_name):
        npzfile = np.load(file_name)
        cls = npzfile['__class__'].item()
        d = npzfile['__dict__'].item()

        self = cls.__new__(cls)
        self.__setstate__(d)
        return self
