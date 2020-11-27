"""
Collection of prior function classes that is to be maximized along with likelihood
function i.e., minimizing the total cost
"""
import numpy as np
from .model import LinearModel
from .estimator import JeffreysPrecisionEstimator, JeffreysHyperparameterEstimator


class Prior:
    """Prior

    An abstract superclass that adds a prior probability distribution

    Parameters
    ----------
    model: model called from the model class
    """

    # TODO: A good way to design an abstract class where no arguments are previously known?!
    def __init__(self, *args):
        self.args = args

    def __call__(self, *args):
        return self._eval(self, *args)

    def _eval(self, *args):
        msg = 'Needs to be implemented in subclass'
        assert NotImplementedError, msg


class GammaPrior(Prior):
    """GammaPrior

    prior = rate*precision - (shape - 1)*log(precision)

    Maximizing the gamma distribution of a precision parameter/hyperparameter. The class
    computes negative log prior for Gamma distribution

    Parameters
    ----------
    shape: shape parameter is defined for a gamma distribution
    rate: rate parameter is defined for a gamma distribution
    """

    # TODO: Is it a good idea to pass a similar argument in constructor and in
    #  the call method (in this case 'precision'), similarly a setter/getter
    #  method could be used maybe?
    def __init__(self, shape, rate):
        assert isinstance(shape, float)
        self._shape = shape

        assert isinstance(rate, float)
        self._rate = rate

    @property
    def shape(self):
        return self._shape

    @property
    def rate(self):
        return self._rate

    @shape.setter
    def shape(self, val):
        self._shape = float(val)

    @rate.setter
    def rate(self, val):
        self._rate = float(val)

    def _eval(self, precision):
        rate = self.rate
        shape = self.shape

        return rate * precision - (shape - 1) * np.log(precision)


class JeffreysPrior(Prior):
    """JeffreysPrior

    prior = log(precision)

    Maximizing Jeffreys prior distribution (inverse of precision of the likelihood
    distribution). The class computes negative log prior for Jeffreys prior.

    Parameters
    ----------
    precision: precision of the likelihood distribution or regularizer
    """

    def __init__(self, precision):
        assert isinstance(precision, float)
        self._precision = precision

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, val):
        self._precision = float(val)

    def _eval(self):
        precision = self._precision

        return np.log(precision)

