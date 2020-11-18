"""
Collection of optimizers
"""
import numpy as np
from .cost import Cost, LeastSquares


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

    Parameters
    ----------
    num_iter : Iterations for the gradient descent
    learn_rate : Learning rate for the algorithm
    cost_expected: Expected value of cost based (default:None)
    """

    def __init__(self, cost, learn_rate, num_iter):
        assert cost.has_gradient
        assert isinstance(cost, LeastSquares)

        super().__init__(cost)

        self.learn_rate = learn_rate
        self.num_iter = num_iter

    def run(self, cost_expected=None):

        params = self.cost.model.params
        if cost_expected is not None:

            cost_iter = []
            for i in range(self.num_iter):
                params = params - self.learn_rate * self.cost.gradient(params)
                cost = self.cost(params)
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

    def run(self, cost_expected=None):
        num_iter = self.num_iter
        learn_rate = self.learn_rate

        params = self.cost.model.params
        prev_params = None
        prev_grad = None

        cost_iter = []
        for i in range(num_iter):
            curr_params = params.copy()
            curr_grad = self.cost.gradient(curr_params)

            if i > 0:
                diff_params = curr_params - prev_params
                diff_grad = curr_grad - prev_grad
                learn_rate = np.linalg.norm(diff_params) ** 2 / np.dot(diff_params, diff_grad)

            prev_params = curr_params
            prev_grad = curr_grad

            params -= learn_rate * self.cost.gradient(curr_params)
            cost_val = self.cost(params)
            cost_iter.append(cost_val)

        print('Iterations for barzilai-borwein are: ', num_iter)
        return params, cost_iter


