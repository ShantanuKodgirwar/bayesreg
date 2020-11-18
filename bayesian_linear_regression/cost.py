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
        # self.residuals = None
        assert isinstance(model, LinearModel)
        self.model = model
        self.data = None

    def __call__(self, params):
        return self._eval(params)

    def _eval(self, params):
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)

    @property
    def has_gradient(self):
        return hasattr(self, 'gradient')

    def gradient(self, params):
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)


class GoodnessOfFit(Cost):
    """GoodnessOfFit

    Fit criterion that will be minimized to obtain the model 
    that explains the data best
    """

    def __init__(self, model, data, precision=1.):
        super().__init__(model)

        assert isinstance(data, Data)
        self.data = data

        self.precision = float(precision)

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

    Sum of squares error term as a cost function (corresponding noise
    model is a Gaussian)
    """

    def _eval(self, residuals):
        beta = self.precision
        N = len(self.data.input)
        # Eq. (12)
        return 0.5 * beta * residuals.dot(residuals) - 0.5 * N * np.log(beta)

    def gradient(self, params=None):
        if params is not None:
            self.model.params = params

        X = self.model.compute_design_matrix(self.data.input)
        return -self.precision * X.T.dot(self.residuals)


class RidgeRegularizer(Cost):
    """RidgeRegularizer

    Implements the general ridge regularization term consisting of a 
    penalizing term 'ridge_param' and general regulazer term 'A'
    """

    def __init__(self, model, ridge_param=None, A=None):
        super().__init__(model)

        if ridge_param is not None:
            self._ridge_param = ridge_param

        if A is None:
            A = np.eye(len(model))

        else:
            msg = 'A must be a symmetric matrix'
            assert np.allclose(A, A.T, rtol=1e-05, atol=1e-08), msg

            msg = 'A must be positive semi-definite'
            assert np.linalg.eigvalsh(A) >= 0, msg

        self.A = A

    @property
    def ridge_param(self):
        return self._ridge_param

    @ridge_param.setter
    def ridge_param(self, ridge_param):
        self._ridge_param = ridge_param

    def _eval(self, params):
        params = self.model.params
        return 0.5 * self._ridge_param * params.dot(self.A.dot(params))


class SumOfCosts(Cost):
    """SumOfCosts

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

    def _eval(self, params):
        return np.sum([cost._eval(params) for cost in self._costs])

    @property
    def has_gradient(self):
        return all([cost.has_gradient for cost in self])

    def __iter__(self):
        return iter(self._costs)
