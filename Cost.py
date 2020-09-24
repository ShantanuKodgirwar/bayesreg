#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:40:25 2020

@author: shantanu
"""

from Linear_Model import LinearModel
import numpy as np

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
    Fit criterion that will be minimized to obtain the model that explains
    the data best
    """
    def __init__(self, data, model):
        
        assert isinstance(model, LinearModel)
        
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
    sum-of-squares error term as a cost function
    """
    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals)

class RidgeRegularizer(Cost):
    """RidgeRegularizer
    Implements the general ridge regularization term consisting of a 
    penalizing term 'Ridge_param' and general regulazer term 'A'
    """
    def __init__(self, model, ridge_param, A=None):

        assert isinstance(model, LinearModel)
        
        super().__init__(model)
        
        self._ridge_param = ridge_param 
        
        if A is None:
            
            A = np.eye(len(model))
        
        else:
            
            msg = 'A must be a symmetric matrix'
            assert np.allclose(A, A.T), msg
            
            eig_vals = np.linalg.eigvalsh(A)
        
            msg = 'A must be a semi-definite matrix'
            assert np.greater_equal(eig_vals, 0), msg

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
        
        vals = [cost._eval(params) for cost in self._costs]
        
        return np.sum(vals)
    
    def __iter__(self):
        return iter(self._costs)
