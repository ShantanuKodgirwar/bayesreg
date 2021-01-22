"""
Testing Least absolute deviations with scipy optimizers
"""
import numpy as np
import bayesian_linear_regression as reg
import matplotlib.pyplot as plt
import time


def calc_lad_bfgs(x, y, n_degree):
    """calc_lad_bfgs

    Evaluates the cost function least absolute deviation (LAD) with a scipy optimizer BFGS.

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
    new_params = optimize.run()
    poly.params = new_params

    return poly(x)


if __name__ == '__main__':

    # %% input parameters
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

    # %% plot results
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(1, 2, figsize=(5, 10))
    plt.subplots_adjust(hspace=0.3)






