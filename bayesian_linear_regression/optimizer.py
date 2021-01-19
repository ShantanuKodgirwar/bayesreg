"""
Collection of optimizers
"""
import numpy as np
from .likelihood import Likelihood, GaussianLikelihood, LaplaceLikelihood
from scipy.optimize import minimize

class Optimizer:
    """Optimizer
    
    Evaluates different optimization algorithms
    """

    def __init__(self, cost):
        assert isinstance(cost, Likelihood)
        self.cost = cost

    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)


class ScipyOptimizer(Optimizer):
    """ScipyOptimizer

    Importing SciPy optimizers
    """

    def __init__(self, cost):
        super().__init__(cost)

    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)


class BFGS(ScipyOptimizer):
    """BFGS

    Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS') is used for solving
    unconstrained, nonlinear optimization problems by evaluating the gradient.
    """

    def __init__(self, cost):
        assert isinstance(cost, LaplaceLikelihood)
        super().__init__(cost)

    def run(self):
        cost = self.cost
        params = cost.model.params

        return minimize(cost, params, method='BFGS', options={'gtol': 1e-6, 'disp': True})


class GradientDescent(Optimizer):
    """GradientDescent
    
    Implements the batch gradient descent algorithm for parameter estimation

    Parameters
    ----------
    num_iter : Iterations for the gradient descent
    learn_rate : Learning rate for the algorithm
    cost_expected: Expected value of cost based (default:None) to stop the loop
    """

    def __init__(self, cost, learn_rate, num_iter):
        assert cost.has_gradient
        assert isinstance(cost, GaussianLikelihood)

        super().__init__(cost)

        self.learn_rate = learn_rate
        self.num_iter = num_iter

    def run(self, cost_expected=None):
        params = self.cost.model.params
        num_iter = self.num_iter

        cost_iter = []

        if cost_expected is not None:
            for i in range(num_iter):
                params = params - self.learn_rate * self.cost.gradient(params)
                cost = self.cost(params)
                cost_iter.append(cost)
                if cost <= cost_expected:
                    print('Iterations for gradient descent are: ', i+1)
                    return params, cost_iter

        for i in range(num_iter):
            params = params - self.learn_rate * self.cost.gradient(params)
            cost = self.cost(params)
            cost_iter.append(cost)

        return params, cost_iter


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
        assert isinstance(cost, GaussianLikelihood)

        super().__init__(cost, learn_rate, num_iter)

    def run(self, cost_expected=None):
        num_iter = self.num_iter
        learn_rate = self.learn_rate
        cost = self.cost
        params = cost.model.params

        prev_params = None
        prev_grad = None
        cost_iter = []

        for i in range(num_iter):
            curr_params = params.copy()
            curr_grad = cost.gradient(curr_params)

            if i > 0:
                diff_params = curr_params - prev_params
                diff_grad = curr_grad - prev_grad
                learn_rate = np.linalg.norm(diff_params) ** 2 / np.dot(diff_params, diff_grad)

            prev_params = curr_params
            prev_grad = curr_grad

            params -= learn_rate * cost.gradient(curr_params)
            cost_val = cost(params)
            cost_iter.append(cost_val)

        print('Iterations for barzilai-borwein are: ', num_iter)
        return params, cost_iter


