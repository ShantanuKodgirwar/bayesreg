"""
Testing Gradient Descent algorithm
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import time


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


def calc_lsq_estimator(x, y, n_degree):
    """
    Implements the classes that calls the Ordinary Least-Square estimator
    """
    data = np.transpose([x, y])
    poly = reg.Polynomial(np.ones(n_degree))
    lsq = reg.LeastSquares(data, poly)
    estimator = reg.LSQEstimator(lsq)
    poly.params = estimator.run()

    return poly(x)


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
    y_pred, params_iter = calc_gradient_descent(x_train, y_train, alpha, num_iter)

    cost_iter = []
    for params in params_iter:
        poly = reg.Polynomial(params)
        lsq = reg.LeastSquares(data, poly)
        cost_val = lsq._eval(lsq.residuals)
        cost_iter.append(cost_val)

    return y_pred, cost_iter


if __name__ == '__main__':
    t = time.process_time()

    true_model = reg.Sinusoid()

    n_degree = 10  # degree of the polynomial

    n_samples = 10  # number of dependent variables

    sigma = 0.2  # Gaussian noise parameter

    cost_expected = n_samples * sigma ** 2 / 2.  # expected chi-squared n_samples

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # predefined Gaussian noise with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861,
                              -0.01635218, -0.08768144, 0.24820263, -0.08946657,
                              0.36653148, 0.13669558])

    # training response vector with noise
    y_train = true_model(x_train) + noise_train

    alpha = 0.09  # Learning rate
    num_iter = int(0.4e4)  # No. of iterations
    num_start = int(num_iter * 1.0)

    y_grad_desc, cost_iter = calc_cost_iter(x_train, y_train, alpha, num_iter)

    y_lsq = calc_lsq_estimator(x_train, y_train, n_degree)

    # %%
    # plot
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(hspace=0.3)

    start = int(100)
    stop = num_iter
    x_num_iter = np.linspace(start, stop, stop - start)

    ax = axes[0]
    ax.set_title('Cost with iterations')
    ax.plot(x_num_iter, cost_iter[start::])
    ax.axhline(cost_expected, ls='--', color='r')
    ax.set_xlabel('num_iter', fontweight='bold')
    ax.set_ylabel('cost', fontweight='bold')
    ax.grid(linestyle='--')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax = axes[1]
    ax.set_title('Fitting by Gradient Descent')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.plot(x_train, y_grad_desc, label='Gradient Descent')
    ax.plot(x_train, y_lsq, label='LSQ estimator')
    ax.plot(x_train, true_model(x_train), label='true model')
    ax.set_xlabel(r'$x_n$', fontweight='bold')
    ax.set_ylabel(r'$y_n$', fontweight='bold')
    ax.grid(linestyle='--')
    ax.legend()

    plt.show()

    t = time.process_time() - t
    print('computation time: ', t)
