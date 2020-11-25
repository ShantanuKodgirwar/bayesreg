"""
Collection of likelihood function classes that is to be maximized i.e., cost/error
is minimized
"""
import numpy as np
from .model import LinearModel
from .data import Data


class Likelihood:
    """Likelihood

    An abstract superclass for scoring model quality by introducing a likelihood
    function

    Parameters
    ----------
    model: A model called from the model class
    """

    def __init__(self, model, *args):
        assert isinstance(model, LinearModel)
        self.model = model

        self._precision = None
        self._hyperparameter = None

    @property
    def precision(self):
        return self._get_precision()

    def _get_precision(self):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)

    @precision.setter
    def precision(self, value):
        self._set_precision(value)

    def _set_precision(self, value):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)

    @property
    def hyperparameter(self):
        return self._get_hyperparameter()

    def _get_hyperparameter(self):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)

    @hyperparameter.setter
    def hyperparameter(self, value):
        self._set_hyperparameter(value)

    def _set_hyperparameter(self, value):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)

    def __call__(self, params):
        raise NotImplementedError
        # TODO: The following is removed: return self._eval(params)

        # TODO: Need to think whether this makes sense at all, since _eval operates only
        #       on the residuals which do not exist if we are dealing with a general cost (such
        #       e.g. the prior / regularizer)

    @property
    def has_gradient(self):
        return hasattr(self, 'gradient')

    def gradient(self, params):
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)


class LogLikelihood(Likelihood):
    """LogLikelihood

    Log of the likelihood by passing a function in it's subclass that decides a
    fit criterion that will be minimized to obtain the model that explains the
    data best.

    Parameters
    ----------
    model: A model called from the model class
    data: input variables (independent data); output/observed variables (dependent data)
    precision: Parameter that is the inverse of the variance (sigma^2); default=1.
    """

    def __init__(self, model, data, precision=1.):
        super().__init__(model)

        assert isinstance(data, Data)
        self.data = data

        self._precision = float(precision)

    def _get_precision(self):
        return self._precision

    def _set_precision(self, value):
        self._precision = value

    @property
    def residuals(self):
        return self.data.output - self.model(self.data.input)

    def __call__(self, params=None):
        if params is not None:
            self.model.params = params

        return self._eval(self.residuals)

    def _eval(self, residuals):
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)


class GaussianLikelihood(LogLikelihood):
    """GaussianLikelihood

    cost = 0.5 * beta * ||t - Xw||^2 - 0.5 * N * log(beta)

    A negative log likelihood is computed for a Gaussian distribution that results
    into least squares error function; N is the length of input data and beta is the
    precision of the distribution.

    Parameters
    ----------
    residuals: Deviation of the output/observed variables from the model that is fitted
    """

    def _eval(self, residuals):
        precision = self._precision

        return 0.5 * precision * residuals.dot(residuals) - 0.5 * len(self.data.input) \
               * np.log(precision)

    def gradient(self, params=None):
        if params is not None:
            self.model.params = params

        X = self.model.compute_design_matrix(self.data.input)
        return -self._precision * X.T.dot(self.residuals)


class Regularizer(Likelihood):
    """Regularizer

    cost = 0.5 * alpha * ||w||**2 - 0.5 * M * log(alpha)

    A negative log likelihood of the model coefficients/weights that has a gaussian
    distribution. This results into a penalizing function called as
    Regularizer that compensates for overfitting.

    Parameters
    ----------
    model: A model called from the model class
    hyperparameter: precision parameter for the model distribution (inverse of variance)
    A: General parameter based on a model that is chosen (default: A=I)
    """

    def __init__(self, model, hyperparameter=1., A=None):
        super().__init__(model)

        if hyperparameter is not None:
            self._hyperparameter = float(hyperparameter)

        if A is None:
            A = np.eye(len(model))

        else:
            msg = 'A must be a symmetric matrix'
            assert np.allclose(A, A.T, rtol=1e-05, atol=1e-08), msg

            msg = 'A must be positive semi-definite'
            assert np.linalg.eigvalsh(A) >= 0, msg

        self.A = A

    def _get_hyperparameter(self):
        return self._hyperparameter

    def _set_hyperparameter(self, value):
        self._hyperparameter = value

    def __call__(self, params):
        params = self.model.params
        hyperparameter = self._hyperparameter

        return 0.5 * hyperparameter * params.dot(self.A.dot(params)) \
               - 0.5 * len(params) * np.log(hyperparameter)


class SumOfCosts(Likelihood):
    """SumOfCosts

    total_cost = 0.5 * (beta * ||t - Xw||^2 - N * log(beta) +
                        alpha * ||w||^2 - M * log(alpha))

    Summation of costs from regression analysis
    (Ex: Gaussian log-likelihood (Least squares error) and Regularizer results to Ridge
    /Tikhonov regularization that is used for compensating overfitting)
    """

    def __init__(self, model, *costs):
        for cost in costs:
            msg = "{0} should be subclass of Cost".format(cost)
            assert isinstance(cost, Likelihood), msg
            assert cost.model is model

        super().__init__(model)
        self._costs = costs

    def __call__(self, params):
        return np.sum([cost(params) for cost in self._costs])

    @property
    def has_gradient(self):
        return all([cost.has_gradient for cost in self])

    def __iter__(self):
        return iter(self._costs)
