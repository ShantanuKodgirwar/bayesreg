"""
Bayesian linear regression package
"""
from .model import LinearModel, StraightLine, Polynomial, Sinusoid, Sinc
from .fitting import Fitter, LSQEstimator, SVDEstimator, RidgeEstimator
from .cost import Cost, GoodnessOfFit, LeastSquares, RidgeRegularizer, \
     SumOfCosts
from .optimizer import Optimizer, GradientDescent
from .utils import rmse

