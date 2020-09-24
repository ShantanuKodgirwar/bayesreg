#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:47:25 2020

@author: shantanu
"""
import numpy as np
import Cost

class Fitter:
    """Fitter
    Fits the data by computing the unknown weights/parameters
    """        
    def __init__(self, cost):
        
        assert isinstance(cost, Cost.Cost)
        self.cost = cost
        
    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)
        
class LSQEstimator(Fitter):
    """LSQEstimator
    Ordinary least squared estimator that minimizes sum-of-squared residuals 
    and calculates regression parameters  
    """
    def __init__(self, cost):
        
        assert isinstance(cost, Cost.LeastSquares)
        
        super().__init__(cost)
        
    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
         
        return np.linalg.pinv(X).dot(y)
    
class SVDEstimator(LSQEstimator):

    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
    
        U, L, V = np.linalg.svd(X, full_matrices=False)

        return V.T.dot(U.T.dot(y) / L)
    
class RidgeEstimator(Fitter):
    """RidgeEstimator
    Generalized Ridge regularizer estimator (modified LSQEstimator) that 
    minimizes sum-of-squares residuals
    """        
    def __init__(self, sum_of_costs):
        
        assert isinstance(sum_of_costs, Cost.SumOfCosts)
        
        for cost in sum_of_costs:
            
            isinstance(cost, Cost.RidgeRegularizer) or isinstance(cost, Cost.LeastSquares)
                   
        super().__init__(sum_of_costs)
        
    def run(self, *args):
        
        A = 0.
        b = 0.
        
        for cost in self.cost:
            
            if isinstance(cost, Cost.RidgeRegularizer):
                A += cost.ridge_param * cost.A
            else:     
                X = cost.model.compute_design_matrix(cost.x)
                A += X.T.dot(X)
                b += X.T.dot(cost.y)
      
        return np.linalg.inv(A)@b

class NoiseModel:    
    """NoiseModel
    Class that generates different types of noise
    """
                      
    def __init__(self, N):
        
        self._N = int(N) # number of dependent variables used
        
    def gaussian_noise(self, sigma):
        
        return np.random.standard_normal(self._N) * sigma
