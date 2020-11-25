"""
Bayesian linear regression package
"""
from .data import Data
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .estimator import Estimator, LSQEstimator, SVDEstimator, RidgeEstimator, PrecisionEstimator, \
     HyperparameterEstimator
from .likelihood import Likelihood, LogLikelihood, GaussianLikelihood, Regularizer, \
     SumOfCosts
from .optimizer import Optimizer, GradientDescent, BarzilaiBorwein
from .posterior import Posterior, MAPJeffreysPrior
from .utils import rmse, calc_gradient, calc_hessian
from .noise import NoiseModel

