"""
Various noise models
"""
import numpy as np


class NoiseModel:
    """NoiseModel

    Class that generates different types of noise

    Parameters
    ----------
    n_samples: The size of the dependent variable/output
    """

    def __init__(self, n_samples):
        assert isinstance(n_samples, int)
        self.n_samples = n_samples

    def gaussian_noise(self, sigma, seed=None):
        n_samples = self.n_samples

        if seed is not None:
            np.random.seed(seed)

        return np.random.standard_normal(n_samples) * sigma
