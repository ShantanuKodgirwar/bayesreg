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
        
M = 4 #degree of the polynomial
N = 15 #number of dependent variables
sigma = 0.2
    
# define training input vector
x_train = np.linspace(0., 1., N)

# define training response vector and add noise
noise = reg.NoiseModel(len(x_train))   
y_train = true_model(x_train) + noise.gaussian_noise(sigma)
 
# test data
dx = np.diff(x_train)[0]
x_test = 0.5 * dx + x_train[:-1]

# define training response vector and add noise
noise = reg.NoiseModel(len(x_test))
y_test = true_model(x_test) + noise.gaussian_noise(sigma)

# concatenate the x_train and response vector y_train 
data = np.transpose([x_train, y_train])
   
# calling the linear model and fitting data
poly = reg.PolyCurve(np.ones(M))
lsq_line = reg.LeastSquares(data, reg.StraightLine())   
fit_poly = reg.LSQEstimator(reg.LeastSquares(data, poly))
poly.params = fit_poly.run()
 
print(f'RMSE: {reg.rmse(y_train,poly(x_train))}')

training_error =  []
test_error = []

for M in range(1, N):
    
    poly2 = reg.PolyCurve(np.ones(M))    
    fitter = reg.LSQEstimator(reg.LeastSquares(data, poly2))   
    poly2.params = fitter.run()
    
    # append the rmse error
    training_error.append(reg.rmse(y_train,poly2(x_train)))   
    test_error.append(reg.rmse(y_test, poly2(x_test)))
    
fig, axes = plt.subplots(1, 2, figsize=(10,5))
ax = axes[0]
ax.set_title('generalized inverse')
ax.plot(training_error, label='training error')
ax.plot(test_error, label='test error')
ax.set_xlabel('M')
ax.set_ylabel('$E_{RMS}$')
ax.legend(prop={'size': 12})
 
ax = axes[1]
ax.scatter(*data.T, label='training data')
ax.scatter(x_test, y_test, label='test data')
ax.plot(x_train, true_model(x_train), label='true fit', color='g')
ax.plot(x_train, poly(x_train), label='best fit', color='r')
ax.set_xlabel(r'$x_n$')
ax.set_ylabel(r'$y_n$')
ax.legend(loc = 3, prop={'size': 10})
fig.tight_layout()
            