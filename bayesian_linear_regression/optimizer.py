"""
Collection of optimizers
"""
import numpy as np

from .cost import Cost, LeastSquares, RidgeRegularizer, SumOfCosts
from .utils import calc_gradient

class Optimizer:
    """Optimizer
    
    Evaluates different optimization algorithms
    """

    def __init__(self, cost):
        assert isinstance(cost, Cost)
        self.cost = cost

    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)


class GradientDescent(Optimizer):
    """GradientDescent
    
    Implements the batch gradient descent algorithm for parameter estimation
    """

    def __init__(self, cost, learn_rate, num_iter):
        """

        Parameters
        ----------
        num_iter : int
                Iterations for the gradient descent
        learn_rate : float
                Learning rate for the algorithm

        """
        assert isinstance(cost, LeastSquares)
        super().__init__(cost)

        self.learn_rate = learn_rate
        self.num_iter = num_iter

    def feature_normalize(self, X):
        pass

    def run(self):
        cost = self.cost
        model = cost.model

        X = model.compute_design_matrix(cost.x)
        y = cost.y
        params = model.params
        for i in range(self.num_iter):
            for j in range(len(model.params)):
                params += params - self.learn_rate * calc_gradient(cost, params, eps)



