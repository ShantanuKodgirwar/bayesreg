"""
Estimating the results of maximum posterior under bayesian inference.
"""
import numpy as np
from .estimator import Estimator, RidgeEstimator, JeffreysPrecisionEstimator, \
    JeffreysHyperparameterEstimator


class Posterior:
    """MaximumPosterior

    Estimates various parameters based on maximum a posteriori(MAP)
    estimation
    """

    def __init__(self, estimator):
        assert isinstance(estimator, Estimator)
        self.estimator = estimator

    def run(self, *args):
        msg = 'Needs to be implemented by a subclass'
        raise NotImplementedError(msg)


class JeffreysGammasPosterior(Posterior):
    """JeffreysGammasPosterior

    Estimates MAP results for coefficients "w", precision parameter "beta" and
    hyperparameter "alpha" for a regularized gaussian likelihood (Ridge regression)
    with Jeffreys prior and Gammas hyperprior.

    Parameters
    ----------
    ridge_estimator: Estimated ridge regression that uses total sum (LSQ+Ridge)
    alpha_estimator: hyperparameter estimated as an analytical result under Jeffreys Prior
    beta_estimator: Precision parameter estimated as an analytical result under Jeffreys Prior
    """

    def __init__(self, ridge_estimator, alpha_estimator, beta_estimator):
        assert isinstance(ridge_estimator, RidgeEstimator)

        assert isinstance(alpha_estimator, JeffreysHyperparameterEstimator)
        self.alpha_estimator = alpha_estimator

        assert isinstance(beta_estimator, JeffreysPrecisionEstimator)
        self.beta_estimator = beta_estimator

        super().__init__(ridge_estimator)
        # TODO: A cleaner way for a constructor?!
        
    def run(self, num_iter):

        log_posterior_list = []
        states = []

        for i in range(num_iter):
            params = self.estimator.run()
            # TODO: Enabling self.alpha_estimator externally?!
            alpha = self.alpha_estimator.run()
            beta = self.beta_estimator.run()
            self.estimator.cost.model.params = params
            self.alpha_estimator.cost.hyperparameter = alpha
            self.beta_estimator.cost.precision = beta

            states.append((params.copy(), beta, alpha))

            # TODO: Gamma Prior class to separate out 'eps', setting a value of mode
            #  and a better structure
            eps = 1e-3  # needs to be the same as eps used in estimators
            a_alpha, b_alpha = eps, eps
            a_beta, b_beta = eps, eps
            log_likelihood_prior = - self.estimator.cost(params)
            log_hyperpriors = (a_alpha - 1) * np.log(alpha) - b_alpha * alpha
            log_hyperpriors += (a_beta - 1) * np.log(beta) - b_beta * beta
            log_posterior = log_likelihood_prior + 0 * log_hyperpriors
            log_posterior_list.append(log_posterior)

        return log_posterior_list, states




