"""
Collection of classes used for parameter estimation.
"""
import numpy as np

from .cost import Cost, GoodnessOfFit, LeastSquares, RidgeRegularizer, SumOfCosts


class Estimator:
    """Fitter

    Fits the data by computing the unknown weights/parameters
    """

    def __init__(self, cost):
        assert isinstance(cost, Cost)
        self.cost = cost

    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)


class LSQEstimator(Estimator):
    """LSQEstimator

    Ordinary least squared estimator that minimizes sum-of-squared residuals 
    and calculates regression parameters  
    """

    def __init__(self, cost):
        assert isinstance(cost, LeastSquares)

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
            isinstance(cost, RidgeRegularizer) or isinstance(cost, LeastSquares)

        super().__init__(sum_of_costs)

    def run(self):

        a = 0.
        b = 0.

        for cost in self.cost:

            if isinstance(cost, RidgeRegularizer):
                a += cost.hyperparameter * cost.A

            else:
                X = cost.model.compute_design_matrix(cost.data.input)
                a += cost.precision * X.T.dot(X)
                b += cost.precision * X.T.dot(cost.data.output)

        return np.linalg.inv(a) @ b


class PrecisionEstimator(Estimator):
    """PrecisionEstimator

    beta  = (N-2) / || t - Xw ||**2

    By Maximum a posteriori (MAP) estimation under the assumption of a jeffreys prior,
    precision parameter "beta" (inverse of variance) is defined based on gaussian model.
    """

    def __init__(self, cost):
        assert isinstance(cost, LeastSquares)

        super().__init__(cost)

    def run(self):
        data = self.cost.data
        residuals = self.cost.residuals
        # coming from Gamma prior on alpha assuming identical shape and rate
        # if set to zero, we are back to Jeffreys' prior
        eps = 1e-3
        return (len(data.input) - 2 + 2*eps) / (np.linalg.norm(residuals)**2 + 2*eps)


class HyperparameterEstimator(Estimator):
    """HyperparameterEstimator

    alpha = (M-2) / ||w||**2

    By Maximum a posteriori (MAP) estimation under the assumption of a jeffreys prior,
    hyperparameter "alpha" is defined based on gaussian model.
    """
    def __init__(self, cost):
        assert isinstance(cost, RidgeRegularizer)

        super().__init__(cost)

    def run(self):
        params = self.cost.model.params
        # coming from Gamma prior on alpha assuming identical shape and rate
        # if set to zero, we are back to Jeffreys' prior
        eps = 1e-3
        return (len(params) - 2 + 2 * eps) / (np.linalg.norm(params) ** 2 + 2 * eps)
