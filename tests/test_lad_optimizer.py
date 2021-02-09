"""
Testing Least absolute deviations with scipy optimizers
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import time


def calc_lad_bfgs(x, y, n_degree):
    """calc_lad_bfgs

    Evaluates the cost function least absolute deviation (LAD) with an iterative gradient
    based optimizer BFGS.

    Parameters
    ----------
    x : input training/testing set
    y : output training/testing set
    n_degree: degree of the fitting polynomial

    Returns
    -------
    poly(x): A fitted polynomial with calculated parameters

    """
    data = reg.Data(np.transpose([x, y]))
    poly = reg.Polynomial(np.ones(n_degree))
    lad = reg.LaplaceLikelihood(poly, data)
    optimize = reg.BFGS(lad)
    new_params, cost_iter = optimize.run()
    poly.params = new_params

    return poly(x), cost_iter


if __name__ == '__main__':

    # %% input parameters
    t = time.process_time()

    true_model = reg.Sinusoid()

    n_degree = 8  # degree of the polynomial

    n_samples = 50  # number of dependent variables

    sigma = 0.2  # Gaussian noise parameter

    cost_expected = n_samples * sigma ** 2 / 2.  # expected chi-squared n_samples

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # Gaussian training noise with a fixed seed value.
    noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma, seed=10)

    # training response vector with noise
    y_train = true_model(x_train) + noise_train

    # calculate the least-absolute deviation (LAD)
    y_lad, cost_iter = calc_lad_bfgs(x_train, y_train, n_degree)

    # %% plot results
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3)

    start = int(0)
    stop = len(cost_iter)
    x_num_iter = np.linspace(start, stop, stop - start)

    ax = axes[0]
    ax.set_title('Cost LAD (BFGS)')
    ax.plot(x_num_iter, cost_iter[start::])
    if cost_expected is not None:
        ax.axhline(cost_expected, ls='--', color='r')
    ax.set_xlabel('num_iter', fontweight='bold')
    ax.set_ylabel('cost', fontweight='bold')
    ax.grid(linestyle='--')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax = axes[1]
    ax.set_title('Least Absolute Deviations')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.plot(x_train, y_lad, label='LAD')
    ax.plot(x_train, true_model(x_train), label='true model', linestyle='--')
    ax.set_xlabel(r'$x_n$', fontweight='bold')
    ax.set_ylabel(r'$y_n$', fontweight='bold')
    ax.grid(linestyle='--')
    ax.legend()

    plt.show()
