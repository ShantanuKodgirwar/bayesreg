"""
Testing Gradient Descent algorithm
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import time
import unittest


def calc_gradient_descent(x_train, y_train, alpha, num_iter):
    """calc_gradient_descent

    Calculates the parameters by gradient descent and passes to the polynomial
    class for fitting.

    Parameters
    ----------
    x_train : input training/testing set
    y_train : output training/testing set
    alpha : Learning rate
    num_iter : No. of iterations

    Returns
    -------
    poly(x): A fitted polynomial with calculated parameters

    """
    data = np.transpose([x_train, y_train])

    poly = reg.Polynomial(np.ones(n_degree))

    lsq = reg.LeastSquares(data, poly)

    optimize = reg.GradientDescent(lsq, alpha, num_iter)

    params, params_iter = optimize.run()

    poly.params = params

    return poly(x_train), params_iter


def calc_cost_iter(x_train, y_train, alpha, num_iter):
    """

    Parameters
    ----------
    x_train : input training/testing set
    y_train : output training/testing set
    alpha : Learning rate
    num_iter : No. of iterations

    Returns
    -------
    cost_iter: returns the cost for parameters evaluated in every iteration
    with gradient descent
    """

    data = np.transpose([x_train, y_train])
    _, params_hist = calc_gradient_descent(x_train, y_train, alpha, num_iter)

    cost_iter = []
    for params in params_hist:
        poly = reg.Polynomial(params)

        lsq = reg.LeastSquares(data, poly)
        cost_val = lsq._eval(lsq.residuals)
        cost_iter.append(cost_val)

    return cost_iter


if __name__ == '__main__':
    t = time.process_time()

    true_model = reg.Sinusoid()

    n_degree = 10  # degree of the polynomial

    n_samples = 10  # number of dependent variables

    sigma = 0.2  # Gaussian noise parameter

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # predefined Gaussian noise with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861,
                              -0.01635218, -0.08768144, 0.24820263, -0.08946657,
                              0.36653148, 0.13669558])

    # Gaussian noise
    # noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma)

    # training response vector with noise
    y_train = true_model(x_train) + noise_train

    alpha = 0.09  # Learning rate

    num_iter = 5000  # No. of iterations

    # predicted value after implementing gradient descent
    y_pred, _ = calc_gradient_descent(x_train, y_train, alpha, num_iter)

    cost_iter = calc_cost_iter(x_train, y_train, alpha, num_iter)

    #%%
    # plot
    plt.figure()
    plt.scatter(x_train, y_train, s=100, alpha=0.7)
    plt.plot(x_train, y_pred, label='Gradient Descent')
    plt.plot(x_train, true_model(x_train), label='true model')
    plt.title('Fitting by Gradient Descent')
    plt.xlabel(r'$x_n$')
    plt.ylabel(r'$y_n$')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

    t = time.process_time()-t
    print('computation time: ', t)
