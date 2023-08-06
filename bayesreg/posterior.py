"""
Estimating the results of maximum posterior under bayesian inference.
"""
from .estimator import Estimator, RidgeEstimator, PrecisionEstimator
from .prior import GammaPrior


class Posterior:
    """MaximumPosterior

    Estimates various parameters based on maximum a posteriori(MAP)
    estimation
    """

    def __init__(self, estimator):
        assert isinstance(estimator, Estimator)
        self.estimator = estimator

    def run(self, *args):
        msg = "Needs to be implemented by a subclass"
        raise NotImplementedError(msg)


class JeffreysPosterior(Posterior):
    """JeffreysPosterior

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
        super().__init__(ridge_estimator)

        assert isinstance(alpha_estimator, PrecisionEstimator)
        self.alpha_estimator = alpha_estimator

        assert isinstance(beta_estimator, PrecisionEstimator)
        self.beta_estimator = beta_estimator

    def run(self, num_iter, gamma_prior_alpha=None, gamma_prior_beta=None):
        """

        Parameters
        ----------
        num_iter: Iterations to converge the posterior solution
        gamma_prior_alpha: negative log of gamma priors for hyperparameter
        gamma_prior_beta: negative log of gamma priors for precision parameter

        Returns
        -------
        states: output parameters such log posterior, alpha, beta, params
        """

        states = []
        for i in range(num_iter):
            params = self.estimator.run()
            alpha = self.alpha_estimator.run()
            beta = self.beta_estimator.run()

            self.estimator.cost.model.params = params
            self.alpha_estimator.cost.hyperparameter = alpha
            self.beta_estimator.cost.precision = beta

            log_likelihood_prior = -self.estimator.cost(params)

            if gamma_prior_alpha is not None:
                assert isinstance(gamma_prior_alpha, GammaPrior)

            log_hyperpriors = -gamma_prior_alpha(alpha)

            if gamma_prior_beta is not None:
                assert isinstance(gamma_prior_beta, GammaPrior)

            log_hyperpriors += -gamma_prior_beta(beta)
            log_posterior = log_likelihood_prior + log_hyperpriors
            states.append((params.copy(), beta, alpha, log_posterior))

        return states
