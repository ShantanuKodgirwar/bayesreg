"""
Bayesian linear regression package
"""
from .data import Data
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .estimator import Estimator, LSQEstimator, SVDEstimator, RidgeEstimator, PrecisionEstimator, \
     HyperparameterEstimator
from .cost import Cost, GoodnessOfFit, LeastSquares, RidgeRegularizer, \
     SumOfCosts
from .optimizer import Optimizer, GradientDescent, BarzilaiBorwein
from .posterior import MaximumPosterior, MAPJeffreysPrior
from .utils import rmse, calc_gradient, calc_hessian
from .noise import NoiseModel

