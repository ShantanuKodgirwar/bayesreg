#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:48:35 2020

@author: shantanu
"""
import linear_regression as reg
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest

def ridge_sklearn(x_train, y_train, n_degree, alpha):
    
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    X = x_train[:, np.newaxis]
    y = y_train
    
    model_ridge = make_pipeline(PolynomialFeatures(n_degree-1), Ridge(alpha))
    model_ridge.fit(X, y)
    
    return model_ridge.predict(X) # returns best fit

class Test(unittest.TestCase):
    
    def test_ridgefit(self):
        # calling the linear model
        poly = reg.PolyCurve(np.ones(n_degree))
        
        # fit the data using ridge estimator
        fitter_ridge = reg.RidgeEstimator(reg.Ridge(data, poly, ridge_param))
        poly.params = fitter_ridge.run()
        
        [self.assertAlmostEqual(ridgefit, ridgefit_sklearn) for ridgefit, ridgefit_sklearn 
         in zip(poly(x_train), ridge_sklearn(x_train, y_train, n_degree, ridge_param))]

if __name__ == '__main__':
        
    true_model = reg.Sinusoid()
            
    n_degree = 10 #degree of the polynomial
    n_samples = 10 #number of dependent variables
    
    # assign the ridge parameter value    
    ridge_param = np.exp(-18)
   
    # define training data input vector x 
    x_train = np.linspace(0., 1., n_samples)
    
    # predefined noise with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861, -0.01635218, 
                              -0.08768144, 0.24820263, -0.08946657, 0.36653148, 0.13669558]) 
    
    # training response vector with noise
    y_train = true_model(x_train) + noise_train
     
    dx = np.diff(x_train)[0]
    
    # test data
    x_test = 0.5 * dx + x_train[:-1]
    
    # predefined noise with sigma = 0.2
    noise_test = np.asarray([-0.08638868, 0.02850903, -0.67500835, 0.01389309, -0.2408333, 
                             0.05583381, -0.1050192, -0.10009032, 0.08420656])
    # test response vector with noise
    y_test = true_model(x_test) + noise_test
    
    # concatenate the x and response vector y mixed with noise
    # will be used for 'training' the model
    data = np.transpose([x_train, y_train])
    
    # ridge param values
    lambda_start = int(-35.0)
    lambda_end = int(-2.0)
    lambda_vals = int(200.0)
    x_axis = np.linspace(start = lambda_start, stop = lambda_end, num = lambda_vals)
    lambda_vals = np.logspace(lambda_start, lambda_end, lambda_vals, base = math.e)
    
    # rmse values over varying lambda (ridge parameter) values
    training_error =  []
    test_error = []
    
    for i in lambda_vals:
        
        poly = reg.PolyCurve(np.ones(n_degree))
        fitter_ridge = reg.RidgeEstimator(reg.Ridge(data, poly, i))
        poly.params = fitter_ridge.run()
        training_error.append(reg.rmse(y_train, poly(x_train)))
        test_error.append(reg.rmse(y_test, poly(x_test)))
       
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)   
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.set_title('RMS error vs Ridge Parameter $\lambda$')
    ax.plot(x_axis, training_error, label='training error')
    ax.plot(x_axis, test_error, label='test error', color='r')
    ax.set_xlabel('ln $\lambda$')
    ax.set_ylabel('$E_{RMS}$')
    ax.legend(prop={'size': 12})
    
    # calling the linear model
    poly2 = reg.PolyCurve(np.ones(n_degree))
    
    # fit the data using ridge estimator
    fitter_ridge = reg.RidgeEstimator(reg.Ridge(data, poly2, ridge_param))
    poly_fit_ridge = fitter_ridge.run()
    
    # fit the data using ordinary least-squares estimator
    fitter_OLS = reg.LSQEstimator(reg.LeastSquares(data, poly2))
    poly_fit_OLS = fitter_OLS.run()
    
    # fit polynomial with Ridge Regression
    poly2.params = poly_fit_ridge
    
    # poly2.params = poly_fit_OLS
    
    # show true model and best fit 
    ax = axes[1]    
    ax.set_title('Ridge Regression')
    ax.scatter(*data.T, s=100, color='k', alpha=0.7)
    ax.plot(x_train, poly2(x_train), color='r', label='Best fit')
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()
    
    unittest.main()