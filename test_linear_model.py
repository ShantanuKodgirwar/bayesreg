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

class Test(unittest.TestCase):
    
    """ Class that calls the unittest module for writing test cases """
    
    def test_OLS_fit(self):
        
        """ Cross-verifies the fit from LSQEstimator class with the fit from sklearn Linear Regression """

        [self.assertAlmostEqual(OLS_fit, OLS_fit_sklearn, places = 5) for OLS_fit, OLS_fit_sklearn 
         in zip(OLS_fit(x_train, y_train, n_degree), 
                OLS_fit_sklearn(x_train, y_train, n_degree))]

    def test_ridge_fit(self):
        
        """ Cross-verifies the fit from ridge class with the fit of ridge sklearn """
                
        [self.assertAlmostEqual(ridgefit, ridge_fit_sklearn, places = 5) for ridgefit, ridge_fit_sklearn 
         in zip(ridge_fit(x_train, y_train, n_degree, ridge_param), 
                ridge_fit_sklearn(x_train, y_train, n_degree, ridge_param))]

def OLS_fit(x_train, y_train, n_degree):
    
    """ Implements the classes that calls the Ordinary Least-Square estimator """
    
    data = np.transpose([x_train, y_train])
    
    poly = reg.PolyCurve(np.ones(n_degree))

    poly.params = reg.LSQEstimator(reg.LeastSquares(data, poly)).run()
        
    return poly(x_train)

def OLS_fit_sklearn(x_train, y_train, n_degree):
    
    """ Implements OLS using scikit-learn """
    
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    X = x_train[:, np.newaxis]
    y = y_train
    
    model_OLS = make_pipeline(PolynomialFeatures(n_degree-1),LinearRegression())
    model_OLS.fit(X,y)
    
    return model_OLS.predict(X)

def rmse_OLS(x_train, y_train, x_test, y_test, n_degree):

    """ Training data and test data rms error by varying n_degree """
    
    data = np.transpose([x_train, y_train])
    
    train_error = []
    test_error = []

    for n_degree in range(1, len(x_train)):
        
        poly = reg.PolyCurve(np.ones(n_degree))
        
        poly.params = reg.LSQEstimator(reg.LeastSquares(data, poly)).run()
        
        train_error.append(reg.rmse(y_train, poly(x_train)))
        
        test_error.append(reg.rmse(y_test, poly(x_test)))
        
    return train_error, test_error

def ridge_fit(x_train, y_train, n_degree, ridge_param):
    
    """Implements the necessary classes to predict values of response vector"""
    
    data = np.transpose([x_train, y_train])

    poly = reg.PolyCurve(np.ones(n_degree))
    
    poly.params = reg.RidgeEstimator(reg.Ridge(data, poly, ridge_param)).run()
        
    return poly(x_train)

def ridge_fit_sklearn(x_train, y_train, n_degree, ridge_param):
    
    """Ridge regression using scikit-learn"""
    
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    X = x_train[:, np.newaxis]
    y = y_train
    
    model_ridge = make_pipeline(PolynomialFeatures(n_degree-1), Ridge(ridge_param))
    model_ridge.fit(X, y)
    
    return model_ridge.predict(X) # returns best fit

def rmse_ridge(x_train, y_train, x_test, y_test, lambda_vals):   
    
    """training data and test data rmse error by varying ridge parameter"""
    
    train_error = []
    test_error = []

    data = np.transpose([x_train, y_train])

    poly = reg.PolyCurve(np.ones(n_degree))
    
    for ridge_param in lambda_vals:       
            
        poly.params = reg.RidgeEstimator(reg.Ridge(data, poly, ridge_param)).run()
        
        # poly.params = fitter_ridge.run()
                
        train_error.append(reg.rmse(y_train, poly(x_train)))
        
        test_error.append(reg.rmse(y_test, poly(x_test)))
    
    return train_error, test_error
    
if __name__ == '__main__':
        
    true_model = reg.Sinusoid()
            
    n_degree = 10 # degree of the polynomial

    n_samples = 10 # number of dependent variables

    sigma = 0.2 # Gaussian noise parameter
           
    ridge_param = np.exp(-9) # assign the ridge parameter value 
        
    x_train = np.linspace(0., 1., n_samples) # define training data input vector x
    
    # predefined noise with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861, -0.01635218, 
                              -0.08768144, 0.24820263, -0.08946657, 0.36653148, 0.13669558])
    
    # noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma)
     
    y_train = true_model(x_train) + noise_train # training response vector with noise
     
    dx = np.diff(x_train)[0]
    
    x_test = 0.5 * dx + x_train[:-1] # test data input vector x
    
    # predefined noise with sigma = 0.2
    noise_test = np.asarray([-0.08638868, 0.02850903, -0.67500835, 0.01389309, -0.2408333, 
                              0.05583381, -0.1050192, -0.10009032, 0.08420656])

    # noise_test = reg.NoiseModel(len(x_test)).gaussian_noise(sigma)
            
    y_test = true_model(x_test) + noise_test # test response vector with noise
        
    # ridge param values
    lambda_start = int(-35.0)
    lambda_end = int(-2.0)
    lambda_vals = int(200.0)
    x_axis = np.linspace(start = lambda_start, stop = lambda_end, num = lambda_vals)
    lambda_vals = np.logspace(lambda_start, lambda_end, lambda_vals, base = math.e)
    
    train_error, test_error = rmse_ridge(x_train, y_train, x_test, y_test, lambda_vals)
    
    train_error_OLS, test_error_OLS = rmse_OLS(x_train, y_train, x_test, y_test, n_degree)

    # plot results
            
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)   
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ax = axes[0,0]    
    ax.set_title('Ordinary Least Squares')
    ax.scatter(x_train, y_train, s=100, color='k', alpha=0.7)
    ax.plot(x_train, OLS_fit(x_train, y_train, n_degree), color='r', label='Best fit')
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()

    ax = axes[0,1]
    ax.set_title('RMS error vs n_degree')
    ax.plot(train_error_OLS, label='training error')
    ax.plot(test_error_OLS, label='test error', color='r')
    ax.set_xlabel('n_degree')
    ax.set_ylabel('$E_{RMS}$')
    ax.legend(prop={'size': 12})

    ax = axes[1,0]    
    ax.set_title('Ridge Regression')
    ax.scatter(x_train, y_train, s=100, color='k', alpha=0.7)
    ax.plot(x_train, ridge_fit(x_train, y_train, n_degree, ridge_param), color='r', label='Best fit')
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()

    ax = axes[1,1]
    ax.set_title('RMS error vs ln $\lambda$')
    ax.plot(x_axis, train_error, label='training error')
    ax.plot(x_axis, test_error, label='test error', color='r')
    ax.set_xlabel('ln $\lambda$')
    ax.set_ylabel('$E_{RMS}$')
    ax.legend(prop={'size': 12})

    unittest.main()