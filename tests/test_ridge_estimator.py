"""
Testing Ridge Regression
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
import bayesian_linear_regression as reg

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class Test(unittest.TestCase):
    """
    Class that calls the unittest module for writing test cases
    """

    def test_ridge_fit(self):
        """
        Cross-verifies the fit from ridge class with the fit of sklearn
        class Ridge
        """

        np.testing.assert_almost_equal(calc_ridge_estimator(x_train, y_train, n_degree, ridge_param),
                                       calc_ridge_estimator_sklearn(x_train, y_train, n_degree,
                                                                    ridge_param), decimal=3)


def calc_ridge_estimator(x, y, n_degree, ridge_param):
    """
    Implements the necessary classes to predict values of response vector
    """
    data = reg.Data(np.transpose([x, y]))
    poly = reg.Polynomial(np.ones(n_degree))
    lsq = reg.LeastSquares(poly, data)
    ridge = reg.RidgeRegularizer(poly, ridge_param)
    total_cost = reg.SumOfCosts(poly, lsq, ridge)
    estimator = reg.RidgeEstimator(total_cost)
    poly.params = estimator.run()

    return poly(x)


def calc_ridge_estimator_sklearn(x, y, n_degree, ridge_param):
    """
    Ridge regression using scikit-learn
    """
    x = x[:, np.newaxis]

    model_ridge = make_pipeline(PolynomialFeatures(n_degree - 1), Ridge(ridge_param))
    model_ridge.fit(x, y)

    return model_ridge.predict(x)  # returns best fit


def calc_rmse_ridge(x_train, y_train, x_test, y_test, n_degree, lambda_logspace):
    """
    Training data and test data rmse error by varying ridge parameter
    """
    train_error = []
    test_error = []

    data = reg.Data(np.transpose([x_train, y_train]))
    poly = reg.Polynomial(np.ones(n_degree))

    for ridge_param in lambda_logspace:
        lsq = reg.LeastSquares(poly, data)
        ridge = reg.RidgeRegularizer(poly, ridge_param)
        total_cost = reg.SumOfCosts(poly, lsq, ridge)
        estimator = reg.RidgeEstimator(total_cost)
        poly.params = estimator.run()
        train_error.append(reg.rmse(y_train, poly(x_train)))
        test_error.append(reg.rmse(y_test, poly(x_test)))

    return train_error, test_error


if __name__ == '__main__':
    true_model = reg.Sinusoid()

    n_degree = 9  # degree of the polynomial

    n_samples = 10  # number of dependent variables

    sigma = 0.2  # Gaussian noise parameter

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # predefined Gaussian noise with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861,
                              -0.01635218, -0.08768144, 0.24820263, -0.08946657,
                              0.36653148, 0.13669558])

    # Gaussian noise
    # noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma)

    y_train = true_model(x_train) + noise_train  # training response vector with noise

    dx = np.diff(x_train)[0]

    x_test = 0.5 * dx + x_train[:-1]  # test data input vector x

    # predefined Gaussian noise with sigma = 0.2
    noise_test = np.asarray([-0.08638868, 0.02850903, -0.67500835, 0.01389309,
                             -0.2408333, 0.05583381, -0.1050192, -0.10009032,
                             0.08420656])

    # Gaussian noise
    # noise_test = reg.NoiseModel(len(x_test)).gaussian_noise(sigma)

    y_test = true_model(x_test) + noise_test  # test response vector with noise

    # ridge param values
    lambda_start = int(-35.0)
    lambda_end = int(3.0)
    lambda_val = int(200.0)

    lambda_logspace = np.logspace(lambda_start, lambda_end, lambda_val, base=math.e)

    train_error_ridge, test_error_ridge = calc_rmse_ridge(x_train, y_train, x_test, y_test,
                                                          n_degree, lambda_logspace)

    # plot results
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(hspace=0.3)

    lambda_vals = np.linspace(lambda_start, lambda_end, lambda_val)

    # select the ridge parameter value
    ridge_param = np.exp(lambda_vals[np.argmin(test_error_ridge)])

    ax = axes[0]
    ax.set_title('Ridge Regression')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.scatter(x_test, y_test, s=100, color='r', alpha=0.7)
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.plot(x_train, calc_ridge_estimator(x_train, y_train, n_degree, ridge_param),
            label='training fit')
    ax.plot(x_test, calc_ridge_estimator(x_test, y_test, n_degree, ridge_param), color='r',
            label='testing fit')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.grid(linestyle='--')
    ax.legend()

    ax = axes[1]
    ax.set_title('RMS error vs ln $\lambda$')
    ax.plot(lambda_vals, train_error_ridge, label='training error')
    ax.plot(lambda_vals, test_error_ridge, label='testing error', color='r')
    ax.set_xlabel('ln $\lambda$')
    ax.set_ylabel('$E_{RMS}$')
    ax.set_ylim(0., 1.)
    ax.grid(linestyle='--')
    ax.legend(prop={'size': 12})

    plt.show()
    unittest.main()
