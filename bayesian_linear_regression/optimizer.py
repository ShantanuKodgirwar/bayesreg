"""
Collection of optimizers
"""
import numpy as np
from .cost import Cost, LeastSquares
from .model import Polynomial


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
        assert cost.has_gradient
        assert isinstance(cost, LeastSquares)

        super().__init__(cost)

        self.learn_rate = learn_rate
        self.num_iter = num_iter

    def run(self, cost_expected=None):

        params = self.cost.model.params
        if cost_expected is not None:
            data = np.transpose([self.cost.x, self.cost.y])

            cost_iter = []
            for i in range(self.num_iter):
                params = params - self.learn_rate * self.cost.gradient(params)
                poly = Polynomial(params)
                lsq = LeastSquares(data, poly)
                cost = lsq._eval(lsq.residuals)
                cost_iter.append(cost)
                if cost <= cost_expected:
                    print('Iterations for gradient descent are: ', i+1)
                    return params, cost_iter

        for i in range(self.num_iter):
            params = params - self.learn_rate * self.cost.gradient(params)

        return params


class BarzilaiBorwein(GradientDescent):
    """BarzilaiBorwein

    Gradient descent method by determining learn rate using Barzilai Borwein method
    """

    def __init__(self, cost, learn_rate, num_iter):
        """

        Parameters
        ----------
        cost: Cost of the function passed
        learn_rate: learn_rate or step size of the gradient descent algorithm
        num_iter: Number of iterations used for convergence
        """

        assert cost.has_gradient
        assert isinstance(cost, LeastSquares)

        super().__init__(cost, learn_rate, num_iter)

    def run(self):

        params = self.cost.model.params
        for i in range(self.num_iter):
            curr_params = params.copy()
            curr_grad = self.cost.gradient(curr_params)
            if i > 0:
                diff_params = curr_params - prev_params
                diff_grad = curr_grad - prev_grad
                self.learn_rate = np.linalg.norm(diff_params) ** 2 / np.dot(diff_params, diff_grad)
            prev_params = curr_params
            prev_grad = curr_grad

            params -= self.learn_rate * self.cost.gradient(curr_params)
        print('Iterations for barzilai-borwein are: ', self.num_iter)
        return params


