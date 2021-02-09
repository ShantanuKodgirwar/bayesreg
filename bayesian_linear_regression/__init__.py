"""
Bayesian linear regression package
"""
from .data import Data
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .prior import GammaPrior
from .likelihood import Likelihood, LogLikelihood, GaussianLikelihood, LaplaceLikelihood,\
     Regularizer, SumOfCosts
from .estimator import Estimator, LSQEstimator, SVDEstimator, RidgeEstimator, PrecisionEstimator
from .optimizer import Optimizer, GradientDescent, BarzilaiBorwein, ScipyOptimizer, BFGS
from .posterior import Posterior, JeffreysPosterior
from .utils import rmse, calc_gradient, calc_hessian
from .noise import NoiseModel

