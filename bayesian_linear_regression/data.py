"""
Data class that consists input, output and it's respective
modalities
"""
import numpy as np


class Data:
    """
    Data class
    """

    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self._data = data

    @property
    def input(self):
        return self._data[:, 0]

    @property
    def output(self):
        return self._data[:, 1]

    def __len__(self):
        return len(self.input)
