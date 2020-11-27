"""
Bayesian linear regression package
"""
from .data import Data
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .prior import GammaPrior
from .likelihood import Likelihood, LogLikelihood, GaussianLikelihood, Regularizer, \
     SumOfCosts
from .estimator import Estimator, LSQEstimator, SVDEstimator, RidgeEstimator, JeffreysPrecisionEstimator, \
     JeffreysHyperparameterEstimator
from .optimizer import Optimizer, GradientDescent, BarzilaiBorwein
from .posterior import Posterior, JeffreysGammasPosterior
from .utils import rmse, calc_gradient, calc_hessian
from .noise import NoiseModel

