"""
Testing Gradient Descent algorithm
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import unittest


def calc_gradient_descent(x_train, y_train, alpha, num_iter):
    """calc_gradient_descent

    Calculates the parameters by gradient descent and passes to the polynomial
    class for fitting.

    Parameters
    ----------
    x_train : Training or testing set
    y_train :
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

    poly.params = optimize.run()

    return poly(x_train)


if __name__ == '__main__':
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

    y_train = true_model(x_train) + noise_train  # training response vector with noise

    alpha = 0.01  # Learning rate

    num_iter = 2000  # No. of iterations

    calc_gradient_descent(x_train, y_train, alpha, num_iter)

    plt.figure()
    plt.scatter(x_train, y_train, s=100, alpha=0.7)
    plt.plot(x_train, calc_gradient_descent(x_train, y_train, alpha, num_iter), label='Gradient Descent fit')
    plt.plot(x_train, true_model(x_train), label='true model')
    plt.xlabel(r'$x_n$')
    plt.ylabel(r'$y_n$')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()
