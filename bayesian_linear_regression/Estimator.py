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

    W = (x.T*X + lambda*I)^{-1}*X.T*t

    Generalized Ridge regularizer estimator (modified LSQEstimator) that 
    minimizes sum-of-squares residuals
    """

    def __init__(self, sum_of_costs):

        assert isinstance(sum_of_costs, SumOfCosts)

        for cost in sum_of_costs:
            isinstance(cost, RidgeRegularizer) or isinstance(cost, LeastSquares)

        super().__init__(sum_of_costs)

    def run(self, ridge_param=None):

        A = 0.
        b = 0.

        for cost in self.cost:

            if ridge_param is not None:
                if isinstance(cost, RidgeRegularizer):
                    cost.ridge_param = ridge_param  # alpha later...

            if isinstance(cost, RidgeRegularizer):
                A += cost.ridge_param * cost.A

            else:
                X = cost.model.compute_design_matrix(cost.data.input)
                A += cost.precision * X.T.dot(X)
                b += cost.precision * X.T.dot(cost.data.output)

        return np.linalg.inv(A) @ b


class PrecisionEstimator(Estimator):
    """PrecisionEstimator

    beta  = (N-2) / || t - Xw ||**2

    By Maximum a posteriori (MAP) estimation under the assumption of a jeffreys prior,
    precision parameter "beta" (inverse of variance) is defined based on gaussian model.
    """

    def __init__(self, cost):
        assert isinstance(cost, GoodnessOfFit)

        super().__init__(cost)

    def run(self):
        data = self.cost.data
        cost = self.cost

        return (len(data.input) - 2)/np.linalg.norm(cost.residuals)**2


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
        return (len(params) - 2) / np.linalg.norm(params) ** 2
