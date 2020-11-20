"""
Collection of classes used for evaluation of cost/error
"""
import numpy as np
from .model import LinearModel
from .data import Data


class Cost:
    """Cost

    Scoring model quality
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


class GoodnessOfFit(Cost):
    """GoodnessOfFit

    Fit criterion that will be minimized to obtain the model 
    that explains the data best.
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


class LeastSquares(GoodnessOfFit):
    """LeastSquares

    cost = 0.5 * beta * ||t - Xw||**2

    Sum of squares error term as a cost function (corresponding noise
    model is a Gaussian)
    """

    def _eval(self, residuals):
        precision = self._precision

        return 0.5 * precision * residuals.dot(residuals) - 0.5 * len(self.data.input) * np.log(precision)

    def gradient(self, params=None):
        if params is not None:
            self.model.params = params

        X = self.model.compute_design_matrix(self.data.input)
        return -self._precision * X.T.dot(self.residuals)


class RidgeRegularizer(Cost):
    """RidgeRegularizer

    cost = 0.5 * alpha * ||w||**2

    Implements the general ridge regularization term consisting of a 
    penalizing term 'hyperparameter (alpha)' and general regulazer term 'A'
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


class SumOfCosts(Cost):
    """SumOfCosts

    total_cost = 0.5 * (beta * ||t - Xw||**2 + alpha * ||w||**2)

    Summation of costs from regression analysis
    (Ex: Ordinary Least squares and Ridge Regularizer)
    """

    def __init__(self, model, *costs):
        for cost in costs:
            msg = "{0} should be subclass of Cost".format(cost)
            assert isinstance(cost, Cost), msg
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
