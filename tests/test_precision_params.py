"""
Testing the precision parameter beta and alpha (hyperparameter)
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
import bayesian_linear_regression as reg


def calc_ridge_jeffreys(x, y, n_degree, num_iter):
    """
    Implements the necessary classes to predict values of response vector
    """
    data = np.transpose([x, y])  # define the data that is passed
    poly = reg.Polynomial(np.ones(n_degree))  # model is defined that is used
    hyper_param = reg.HyperParameter(poly)  # hyperparameter based on Jeffreys prior
    prec_param = reg.PrecisionParameter(data, poly)  # precision parameter based on Jeffreys prior
    lsq = reg.LeastSquares(data, poly)  # least-squares as cost function
    ridge = reg.RidgeRegularizer(poly, 1)  # Regularizer term; pass ridge_param = 0
    total_cost = reg.SumOfCosts(poly, lsq, ridge)  # lsq+ridge to give the modified cost function
    ridge_estimator = reg.RidgeEstimator(total_cost)  # estimate the modified error function
    alpha, beta, params, log_posterior = reg.MAPJeffreysPrior(ridge_estimator, hyper_param, prec_param).run(num_iter)
    poly.params = params

    return poly(x), log_posterior


if __name__ == '__main__':

    true_model = reg.Sinusoid()
    n_degree, n_samples = 9, 10  # degree of the polynomial and input samples
    sigma = 0.2  # Gaussian noise parameter

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # predefined Gaussian noise for training data with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861,
                              -0.01635218, -0.08768144, 0.24820263, -0.08946657,
                              0.36653148, 0.13669558])
    # Gaussian noise
    # noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma)

    y_train = true_model(x_train) + noise_train  # training response vector with noise
    dx = np.diff(x_train)[0]
    x_test = 0.5 * dx + x_train[:-1]  # test data input vector x

    # predefined Gaussian noise for testing data with sigma = 0.2
    noise_test = np.asarray([-0.08638868, 0.02850903, -0.67500835, 0.01389309,
                             -0.2408333, 0.05583381, -0.1050192, -0.10009032,
                             0.08420656])
    # Gaussian noise
    # noise_test = reg.NoiseModel(len(x_test)).gaussian_noise(sigma)

    y_test = true_model(x_test) + noise_test  # test response vector with noise

    # number of iterations
    num_iter = 10

    # plot results
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(hspace=0.3)

    ridge_training, log_posterior_train = calc_ridge_jeffreys(x_train, y_train, n_degree, num_iter)
    ridge_testing, log_posterior_test = calc_ridge_jeffreys(x_test, y_test, n_degree, num_iter)

    ax = axes[0]
    ax.set_title('Ridge Regression')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.scatter(x_test, y_test, s=100, color='r', alpha=0.7)
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.plot(x_train, ridge_training, label='training fit')
    ax.plot(x_test, ridge_testing, color='r', label='testing fit')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.grid(linestyle='--')
    ax.legend()

    num_iter = np.linspace(1, num_iter, num_iter)

    ax = axes[1]
    ax.set_title(r'Log Posterior vs iterations')
    ax.plot(num_iter, log_posterior_train, label='training error')
    ax.plot(num_iter, log_posterior_test, label='testing error', color='r')
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$ln({w}, \alpha, \beta)$')
    ax.set_ylim(0., 1.)
    ax.grid(linestyle='--')
    ax.legend(prop={'size': 12})

    plt.show()

