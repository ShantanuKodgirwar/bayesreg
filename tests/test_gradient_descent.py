"""
Testing Gradient Descent algorithm
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import time


def calc_grad_desc(x, y, alpha, num_iter):
    """calc_grad_desc

    Solves the least-squares by gradient descent algorithm

    Parameters
    ----------
    x : input training/testing set
    y : output training/testing set
    alpha : Learning rate
    num_iter : No. of iterations

    Returns
    -------
    poly(x): A fitted polynomial with calculated parameters

    """
    data = np.transpose([x, y])
    poly = reg.Polynomial(np.ones(n_degree))
    lsq = reg.LeastSquares(data, poly)
    optimize = reg.GradientDescent(lsq, alpha, num_iter)
    params_iter = np.array(optimize.run())
    poly.params = params_iter[-1]

    return params_iter, poly(x)


def calc_barzilai_borwein(x, y, alpha, num_iter):
    """calc_barzilai_borwein

    Solves the least-squares by gradient descent algorithm

    Parameters
    ----------
    x : input training/testing set
    y : output training/testing set
    alpha : Learning rate
    num_iter : No. of iterations

    Returns
    -------
    poly(x): A fitted polynomial with calculated parameters

    """
    data = np.transpose([x, y])
    poly = reg.Polynomial(np.zeros(n_degree))
    lsq = reg.LeastSquares(data, poly)
    optimize = reg.BarzilaiBorwein(lsq, alpha, num_iter)
    params = optimize.run()
    poly.params = params

    return poly(x)


def calc_cost_params(x, y, params_iter):
    """calc_cost_params

    calculates cost from the parameters found by gradient descent and passes the
    parameters for fitting

    Parameters
    ----------
    x: input training/testing set
    y: output training/testing set
    params_iter: parameters evaluated by calculating the gradient

    Returns
    -------
    cost_iter: returns the cost for parameters evaluated in every iteration
    with gradient descent

    """
    data = np.transpose([x, y])
    cost_iter = []
    for params in params_iter:
        poly = reg.Polynomial(params)
        lsq = reg.LeastSquares(data, poly)
        cost_val = lsq._eval(lsq.residuals)
        cost_iter.append(cost_val)

    return cost_iter


def calc_lsq_estimator(x, y, n_degree):
    """calc_lsq_estimator

    Solves the least-squares equation method by the analytical solution and passes to
    the polynomial class for fitting

    Parameters
    ----------
    x: input training/testing set
    y: output training/testing set
    n_degree: no. of regression coefficients (parameter vector)

    Returns
    -------
    poly(x): A fitted polynomial with calculated parameters
    """

    data = np.transpose([x, y])
    poly = reg.Polynomial(np.ones(n_degree))
    lsq = reg.LeastSquares(data, poly)
    estimator = reg.LSQEstimator(lsq)
    poly.params = estimator.run()

    return poly(x)


def main():

    t = time.process_time()
    params_iter, y_grad_desc = calc_grad_desc(x_train, y_train, alpha, num_iter)
    cost_iter = calc_cost_params(x_train, y_train, params_iter)
    t = time.process_time() - t
    print('Gradient descent time:', t)

    # run LSQ estimator
    t2 = time.process_time()
    y_lsq = calc_lsq_estimator(x_train, y_train, n_degree)
    t2 = time.process_time() - t2
    print('LSQ estimator time:', t2)

    t3 = time.process_time()
    y_grad_bb = calc_barzilai_borwein(x_train, y_train, alpha_init, num_iter_bb)
    t3 = time.process_time() - t3
    print('Barzilai-Borwein time :', t3)

    return cost_iter, y_grad_desc, y_lsq, y_grad_bb

if __name__ == '__main__':
    t = time.process_time()

    true_model = reg.Sinusoid()

    n_degree = 8  # degree of the polynomial

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

    # parameters for gradient descent method
    alpha = 0.07  # Learning rate
    num_iter = int(0.7e4)  # No. of iterations

    # parameters for gradient descent with barzilai borwein method
    num_iter_bb = 50
    alpha_init = 1e-3

    cost_iter, y_grad_desc, y_lsq, y_grad_bb = main()

    # %%
    # plot
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(hspace=0.3)

    start = int(100)
    stop = num_iter
    x_num_iter = np.linspace(start, stop, stop - start)

    ax = axes[0]
    ax.set_title('Cost for Gradient Descent')
    ax.plot(x_num_iter, cost_iter[start::])
    ax.axhline(cost_expected, ls='--', color='r')
    ax.set_xlabel('num_iter', fontweight='bold')
    ax.set_ylabel('cost', fontweight='bold')
    ax.grid(linestyle='--')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax = axes[1]
    ax.set_title('Gradient Descent')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.plot(x_train, y_grad_desc, label='Gradient Descent', linewidth=4.0)
    ax.plot(x_train, y_lsq, label='LSQ estimator')
    ax.plot(x_train, true_model(x_train), label='true model', linestyle='--')
    ax.set_xlabel(r'$x_n$', fontweight='bold')
    ax.set_ylabel(r'$y_n$', fontweight='bold')
    ax.grid(linestyle='--')
    ax.legend()

    ax = axes[2]
    ax.set_title('Barzilai Borwein')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.plot(x_train, y_grad_bb, label='Barzilai Borwein', linewidth=4.0)
    ax.plot(x_train, y_lsq, label='LSQ estimator')
    ax.plot(x_train, true_model(x_train), label='true model', linestyle='--')
    ax.set_xlabel(r'$x_n$', fontweight='bold')
    ax.set_ylabel(r'$y_n$', fontweight='bold')
    ax.grid(linestyle='--')
    ax.legend()

    plt.show()
