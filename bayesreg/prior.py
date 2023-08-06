"""
Collection of prior function classes that is to be maximized along with likelihood
function i.e., minimizing the total cost
"""
import numpy as np
from .model import LinearModel


class Prior:
    """Prior

    An abstract class that adds a prior probability distribution

    Parameters
    ----------
    model: model called from the model class
    """

    def __init__(self, model):
        assert isinstance(model, LinearModel)
        self.model = model

    def __call__(self, params=None):
        if params is not None:
            self.model.params = params

        return self._eval(params)

    def _eval(self, params):
        msg = "Needs to be implemented in subclass"
        assert NotImplementedError, msg


class HyperPrior:
    """HyperPrior

    An abstract class that evaluates prior of the prior distribution (hyperprior) for
    precision parameters/hyperparameters.
    """

    # TODO: Is this a good way to create an abstract class where the arguments are not
    #  given in constructor?
    def __init__(self, *args):
        self.args = args

    def __call__(self, precision_param):
        return self._eval(precision_param)

    def _eval(self, precision_param):
        msg = "Needs to be implemented in subclass"
        assert NotImplementedError, msg


class JeffreysPrior(HyperPrior):
    """JeffreysPrior

    prior = log(precision)

    Maximizing Jeffreys prior distribution (inverse of precision of the likelihood
    distribution). The class computes negative log prior for Jeffreys prior.

    Parameters
    ----------
    precision_param: precision value of the likelihood distribution or regularizer (alpha/beta)
    """

    def _eval(self, precision_param):
        precision_param = self.precision_param

        return np.log(precision_param)


class GammaPrior(HyperPrior):
    """GammaPrior

    prior = rate*precision - (shape - 1)*log(precision)

    Maximizing the gamma distribution of a precision parameter/hyperparameter. The class
    computes negative log prior for Gamma distribution

    Parameters
    ----------
    shape: shape parameter is defined for a gamma distribution
    rate: rate parameter is defined for a gamma distribution
    """

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

    def _eval(self, precision_param):
        rate = self.rate
        shape = self.shape

        return rate * precision_param - (shape - 1) * np.log(precision_param)
