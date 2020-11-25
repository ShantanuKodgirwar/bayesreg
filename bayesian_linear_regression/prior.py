"""
Collection of prior function classes that is to be maximized along with likelihood
function i.e., minimizing the total cost
"""
import numpy as np
from .model import LinearModel


class Prior:
    """Prior

    An abstract superclass that adds a prior probability distribution

    Parameters
    ----------
    model: A model called from the model class
    """

    def __init__(self, model, *args):
        assert isinstance(model, LinearModel)
        self.model = model

    def __call__(self, params):
        msg = 'Needs to be implemented in subclass'
        assert NotImplementedError, msg


class GammaPrior(Prior):
    """GammaPrior

    A prior based on Gamma distribution consists of two parameters
    'shape' and 'rate'. Gamma distribution is a conjugate prior to several
    likelihood distributions (Gaussian, Poisson, exponential)

    """

    def __init__(self, model, shape, rate, beta):
        pass