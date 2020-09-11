#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:11:35 2020

@author: shantanu
"""

import linear_regression as reg
import numpy as np
import matplotlib.pyplot as plt

    
true_model = reg.Sinusoid()
        
M = 9 #degree of the polynomial
N = 15 #number of dependent variables
sigma = 0.2
    
# define input vector x and response vector y 
x = np.linspace(0., 1., N)
y = true_model(x)
 
dx = np.diff(x)[0]

# test data
x_test = 0.5 * dx + x[:-1]
noise = reg.NoiseModel(len(x_test))
y_test = true_model(x_test) + noise.gaussian_noise(sigma)

# Introduce some noise for training data
noise = reg.NoiseModel(N)    
gauss_noise = noise.gaussian_noise(sigma)

# concatenate the x and response vector y mixed with noise
# will be used for 'training' the model
data = np.transpose([x,y+gauss_noise])
   
# calling the linear model
poly = reg.PolyCurve(np.ones(M))

lsq_line = reg.LeastSquares(data, reg.StraightLine())
    
fit_poly = reg.LSQEstimator(reg.LeastSquares(data, poly))

poly.params = fit_poly.run()
 
print(f'RMSE: {reg.rmse(y,poly(x))}')

training_error =  []
test_error = []
test_error2 = []
training_error2 = []

for M in range(1, N):
    
    poly2 = reg.PolyCurve(np.ones(M))
    
    fitter = reg.LSQEstimator(reg.LeastSquares(data, poly2))
    poly2.params = fitter.run()
    training_error.append(reg.rmse(y,poly2(x)))
    test_error.append(reg.rmse(y_test, poly2(x_test)))
    
    fitter2 = reg.SVDFitter(reg.LeastSquares(data, poly2))
    poly2.params = fitter2.run()
    training_error2.append(reg.rmse(y,poly2(x)))
    test_error2.append(reg.rmse(y_test, poly2(x_test)))

fig, axes = plt.subplots(1, 3, figsize=(12,4))
ax = axes[0]
ax.set_title('generalized inverse')
ax.plot(training_error, label='training error')
ax.plot(test_error, label='test error')
ax.set_xlabel('M')
ax.set_ylabel('$E_{RMS}$')
ax.legend(prop={'size': 12})

ax = axes[1]
ax.set_title('SVD')
ax.plot(training_error2, label='training error')
ax.plot(test_error2, label='test error')
ax.set_xlabel('M')
ax.set_ylabel(r'$E_{RMS}$')
ax.legend(prop={'size': 12})
 
ax = axes[2]
ax.scatter(*data.T, label='training data')
ax.scatter(x_test, y_test, label='test data')
ax.plot(x, true_model(x), label='true fit', color='g')
ax.plot(x, poly(x), label='best fit', color='r')
ax.set_xlabel(r'$x_n$')
ax.set_ylabel(r'$y_n$')
ax.legend(loc = 3, prop={'size': 10})
fig.tight_layout()


if False:

    ## show true model and best fit
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)   
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    ax = axes[0]    
    ax.scatter(*data.T, s=100, color='k', alpha=0.7)
    ax.plot(x, poly(x), color='r', label='Best fit')
    ax.plot(x, true_model(x), color='g', label='true model')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()
    
    
    # grid for cost function    
    n_grid = int(1e2)
    intercept_axis = np.linspace(-1., 1., n_grid)
    slope_axis = np.linspace(-1., 3., n_grid)
    grid = np.meshgrid(intercept_axis, slope_axis)
    grid = np.reshape(grid, (2, -1)).T
    costs = np.array([lsq_line(params) for params in grid])
    
    ax = axes[1]
    ax.contour(intercept_axis, slope_axis, np.exp(-0.5*costs.reshape(n_grid, -1)))
    ax.scatter(*poly.params[:2], s=100, color='k', alpha=0.5, label='best')
    # ax.scatter(*params_true, s=100, color='r', marker='x', alpha=0.5, label='truth')    
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    ax.legend()
    fig.tight_layout()
