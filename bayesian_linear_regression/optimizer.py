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

        self._learn_rate = learn_rate
        self.num_iter = num_iter

        @property
        def learn_rate(self):
            return self._learn_rate

        @learn_rate.setter
        def learn_rate(self, learn_rate):
            self._learn_rate = learn_rate

    def run(self):
        """

        Returns
        -------
        params: ndarray
            returns the evaluated parameters
        params_hist: list
            returns the parameters for every iteration
        """

        params = self.cost.model.params
        params_iter = []
        for i in range(self.num_iter):
            params = params - self._learn_rate * self.cost.gradient(params)
            params_iter.append(params)
        return params_iter

    def barzilai_borwein(self, max_epoch):

        learn_rate = self._learn_rate
        params = self.cost.model.params

        learn_rate_iter = []
        for i in range(max_epoch):
            curr_params = params.copy()
            curr_grad = self.cost.gradient(curr_params)

            if i > 0:
                diff_params = curr_params - prev_params
                diff_grad = curr_grad - prev_grad
                learn_rate = np.linalg.norm(diff_params) ** 2 / np.dot(diff_params, diff_grad)
                learn_rate_iter.append(learn_rate)

            prev_params = curr_params
            prev_grad = curr_grad

            for j in range(self.num_iter):
                params -= learn_rate * self.cost.gradient(curr_params)

        return params


