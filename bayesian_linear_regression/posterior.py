"""
Estimating the results of maximum posterior under bayesian inference.
"""
import numpy as np
from .cost import Cost, PrecisionParameter, HyperParameter
from .fitting import Fitter, LSQEstimator, RidgeEstimator


class MaximumPosterior:
    """MaximumPosterior

    Estimates various parameters based on maximum a posteriori(MAP)
    estimation
    """

    def __init__(self, fitter):
        assert isinstance(fitter, Fitter)

        self.fitter = fitter

    def run(self, *args):
        msg = 'Needs to be implemented by a subclass'
        raise NotImplementedError(msg)


class MAPJeffreysPrior(MaximumPosterior):
    """MAPJeffreysPrior

    Estimates MAP results for coefficients "w", precision parameter "beta" and
    hyperparameter "alpha" under the assumption of a jeffreys prior.

    Parameters
    ----------
    ridge_estimator: Estimated ridge regression
    """

    def __init__(self, ridge_estimator, hyper_parameter, precision_parameter):
        assert isinstance(ridge_estimator, RidgeEstimator)

        assert isinstance(hyper_parameter, HyperParameter)
        self.hyper_parameter = hyper_parameter

        assert isinstance(precision_parameter, PrecisionParameter)
        self.precision_parameter = precision_parameter

        super().__init__(ridge_estimator)

    def run(self, num_iter):
        params = self.hyper_parameter.model.params

        log_posterior_list = []
        for i in range(num_iter):
            curr_params = params.copy()
            alpha = self.hyper_parameter(curr_params)
            beta = self.precision_parameter(curr_params)
            params = self.fitter.run()
            log_posterior = np.log(alpha)+np.log(beta)
            log_posterior_list.append(log_posterior)

        return alpha, beta, params, log_posterior_list




