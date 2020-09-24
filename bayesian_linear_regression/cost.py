"""
Cost functions for parameter fitting
"""
import numpy as np

from .model import LinearModel


class Cost:
    """Cost

    Scoring model quality
    """    
    def __init__(self, model, **kwargs):
        
        assert isinstance(model, LinearModel)
        
        self.model = model
        
        for A, ridge_param in kwargs.items():
                        
            self.A = A
            self.ridge_param = ridge_param

    def _eval(self, params):
        
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)

    
class GoodnessOfFit(Cost):
    """GoodnessOfFit

    Fit criterion that will be minimized to obtain the model 
    that explains the data best
    """
    def __init__(self, data, model):
        super().__init__(model)        
        self.data = np.array(data)

    @property
    def x(self):
        return self.data[:,0] 

    @property
    def y(self):
        return self.data[:,1]  

    @property
    def residuals(self):        
        return self.y - self.model(self.x)

    def __call__(self, params=None):
        
        if params is not None:
            self.model.params = params
            
        return self._eval(self.residuals)

    def _eval(self, residuals):
        
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)

    
class LeastSquares(GoodnessOfFit):
    """LeastSquares

    Sum of squares error term as a cost function (corresponding noise
    model is a Gaussian)
    """
    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals)

    
class Gradient(Cost):
    """Gradient

    Compute the gradient of the cost function and calculate the norm of it
    """    
    pass


class RidgeRegularizer(Cost):
    """RidgeRegularizer

    Implements the general ridge regularization term consisting of a 
    penalizing term 'ridge_param' and general regulazer term 'A'
    """
    def __init__(self, model, ridge_param, A=None):

        super().__init__(model)
        
        self._ridge_param = ridge_param 
        
        if A is None:
            A = np.eye(len(model))
        
        else:
            
            # test if the matrix is symmetric
            if A != np.transpose(A):
                raise ValueError('A must be a symmetric matrix')
            
            # eigen values of real symmetric/complex Hermitian matrix
            eig_vals, _ = np.linalg.eigh(A)
            
            # eigen values non-negative to test semi-definite condition
            if eig_vals < 0:
                raise ValueError('A is not a semi-definite matrix')

        self.A = A
    
    @property
    def ridge_param(self):
        return self._ridge_param
        
    def _eval(self, residuals):
        
        params = self.model.params
        
        return 0.5 * self._ridge_param * params.dot(self.A.dot(params))

    
class SumOfCosts(Cost):
    """SumOfCosts

    Summation of costs from regression analysis
    (Ex: Ordinary Least squares and Ridge Regularizer)
    """
    def __init__(self, model, *costs):

        for cost in costs:
            msg = "{0} should be subclass of Cost".format(cost)
            assert isinstance(cost, Cost), msg            
            assert cost.model is model
    
        super().__init__(model)
        
        self._costs = costs
    
    def _eval(self, params):
        return np.sum([cost._eval(params) for cost in self._costs])
        
    def __iter__(self):
        return iter(self._costs)

