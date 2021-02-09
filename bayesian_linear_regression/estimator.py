"""
Collection of classes used for parameter estimation.
"""
import numpy as np

from .likelihood import Likelihood, GaussianLikelihood, Regularizer, SumOfCosts
from .prior import GammaPrior


class Estimator:
    """Fitter

    Fits the data by computing the unknown weights/parameters
    """

    def __init__(self, cost):
        assert isinstance(cost, Likelihood)
        self.cost = cost

    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)


class LSQEstimator(Estimator):
    """LSQEstimator

    Ordinary least squares (OLS) estimator that minimizes sum-of-squared residuals
    and calculates regression parameters  
    """

    def __init__(self, cost):
        assert isinstance(cost, GaussianLikelihood)

        super().__init__(cost)

    def run(self, *args):
        model = self.cost.model
        data = self.cost.data

        X = model.compute_design_matrix(data.input)
        y = data.output

        return np.linalg.pinv(X).dot(y)


class SVDEstimator(LSQEstimator):

    def run(self, *args):
        model = self.cost.model
        data = self.cost.data

        X = model.compute_design_matrix(data.input)
        y = data.output

        U, L, V = np.linalg.svd(X, full_matrices=False)

        return V.T.dot(U.T.dot(y) / L)


class RidgeEstimator(Estimator):
    """RidgeEstimator

    W = (beta * X.T*X + alpha*A)^{-1}*X.T*t

    Generalized Ridge regularizer estimator (modified LSQEstimator) that 
    minimizes sum-of-squares residuals
    """

    def __init__(self, sum_of_costs):

        assert isinstance(sum_of_costs, SumOfCosts)

        for cost in sum_of_costs:
            isinstance(cost, Regularizer) or isinstance(cost, GaussianLikelihood)

        super().__init__(sum_of_costs)

    def run(self):

        a = 0.
        b = 0.

        for cost in self.cost:

            if isinstance(cost, Regularizer):
                a += cost.hyperparameter * cost.A

            else:
                X = cost.model.compute_design_matrix(cost.data.input)
                a += cost.precision * X.T.dot(X)
                b += cost.precision * X.T.dot(cost.data.output)

        return np.linalg.inv(a) @ b


class PrecisionEstimator(Estimator):
    """PrecisionEstimator

    beta  = (N-2) + 2*shape / || t - Xw ||**2 + 2*rate
    alpha = (M-2) + 2*shape / ||w||**2 + 2*rate

    Estimates precision parameter/hyperparameter based on gaussian likelihood distribution
    with Jeffreys prior and gamma hyperprior distribution (optional) given by shape and rate.
    'beta', a precision parameter for the gaussian distribution of the output data t;
    'alpha' a precision parameter/hyperparameter for the gaussian distribution of
    coefficients w
    """

    def __init__(self, cost, hyperprior=None):

        if hyperprior is not None:
            assert isinstance(hyperprior, GammaPrior)
        self.hyperprior = hyperprior

        assert isinstance(cost, GaussianLikelihood) or isinstance(cost, Regularizer)
        super().__init__(cost)

    def run(self):
        cost = self.cost

        if isinstance(cost, GaussianLikelihood):
            data = self.cost.data
            residuals = self.cost.residuals

            if self.hyperprior is not None:
                return (len(data.input) - 2 + 2 * self.hyperprior.shape)\
                       / (np.linalg.norm(residuals) ** 2 + 2 * self.hyperprior.rate)
            else:
                return (len(data.input) - 2) / (np.linalg.norm(residuals) ** 2)

        else:
            params = self.cost.model.params

            if self.hyperprior is not None:
                return (len(params) - 2 + 2 * self.hyperprior.shape)\
                       / (np.linalg.norm(params) ** 2 + 2 * self.hyperprior.rate)
            else:
                return (len(params) - 2) / (np.linalg.norm(params) ** 2)