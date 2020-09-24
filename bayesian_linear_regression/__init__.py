"""
Bayesian linear regression package
"""
from .model import LinearModel, StraightLine, Polynomial, Sinusoid
from .fitting import Fitter, LSQEstimator, SVDEstimator, RidgeEstimator
from .cost import Cost, GoodnessOfFit, LeastSquares, RidgeRegularizer, \
     SumOfCosts
from .utils import rmse

