"""
Bayesian linear regression package
"""
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .fitting import Fitter, LSQEstimator, SVDEstimator, RidgeEstimator
from .cost import Cost, GoodnessOfFit, LeastSquares, RidgeRegularizer, \
     SumOfCosts
from .optimizer import Optimizer, GradientDescent, BarzilaiBorwein
from .utils import rmse, calc_gradient, calc_hessian

