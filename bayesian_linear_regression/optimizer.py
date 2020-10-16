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
        cost = self.cost
        model = cost.model

        params = model.params

        params_iter = []
        for i in range(self.num_iter):
            params = params - self._learn_rate * cost.gradient(params)
            params_iter.append(params)
        return params_iter

    def barzilai_borwein(self, params_iter):
        learn_rate = self._learn_rate

        cost = self.cost
        model = cost.model
        params = model.params

        params_iter_new = []
        for i in range(self.num_iter):
            curr_params = params_iter[i]
            curr_calc_grad = self.cost.gradient(curr_params)

            if i > 0:
                prev_params = params_iter[i-1]
                prev_calc_grad = self.cost.gradient(params_iter[i-1])
                diff_params = curr_params - prev_params
                diff_grad = curr_calc_grad - prev_calc_grad
                learn_rate = np.abs(diff_params.T.dot(diff_grad)) / \
                                   np.linalg.norm(diff_grad) ** 2

            params = params - learn_rate * self.cost.gradient(params)
            params_iter_new.append(params)

            return params_iter_new, learn_rate


