"""
Collection of optimizers
"""
import numpy as np

from .cost import Cost, LeastSquares, RidgeRegularizer, SumOfCosts


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
    
    Implements the gradient descent algorithm for parameter estimation
    """

    def __init__(self, cost, learning_rate):
        super().__init__(cost)

        self.learning_rate = learning_rate

    def run(self):

        model = self.cost.model
