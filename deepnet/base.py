
import numpy as np

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
        np.savez(file_name, **self.__getstate__())

    @classmethod
    def from_file(cls, file_name):
        npzfile = np.load(file_name)

        self = cls.__new__(cls)
        self.__setstate__(npzfile)
        return self
