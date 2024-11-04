"""
Testing the precision parameter beta and alpha (hyperparameter)
"""
import matplotlib.pyplot as plt
import numpy as np

import bayesreg as reg


def calc_map_jeffreys(input_vec, output_vec, n_degree, num_iter):
    """
    Implements the necessary classes to predict values of response vector
    """
    # Instantiate data and model
    data = reg.Data(
        np.transpose([input_vec, output_vec])
    )  # define the data that is passed
    poly = reg.Polynomial(np.ones(n_degree))  # model is defined that is used

    # Instantiate likelihood/cost based on Gaussian distribution
    lsq = reg.GaussianLikelihood(poly, data)  # least-squares as cost function
    regularizer = reg.Regularizer(poly)  # regularizer
    total_cost = reg.SumOfCosts(
        poly, lsq, regularizer
    )  # lsq+regularizer gives ridge regularizer

    # Instantiate priors
    eps = float(1e-3)
    gamma_prior_alpha = reg.GammaPrior(shape=eps, rate=eps)
    gamma_prior_beta = reg.GammaPrior(shape=eps, rate=eps)

    # Instantiate estimators
    alpha_estimator = reg.PrecisionEstimator(
        regularizer, gamma_prior_alpha
    )  # hyperparameter estimator
    beta_estimator = reg.PrecisionEstimator(
        lsq, gamma_prior_beta
    )  # precision parameter estimator
    ridge_estimator = reg.RidgeEstimator(
        total_cost
    )  # estimate the ridge error function

    # Maximum posterior with Jeffreys prior
    max_posterior = reg.JeffreysPosterior(
        ridge_estimator, alpha_estimator, beta_estimator
    )
    states = max_posterior.run(num_iter, gamma_prior_alpha, gamma_prior_beta)

    # evaluated parameter values
    params, beta, alpha, log_posterior = list(map(np.array, zip(*states)))
    poly.params = params[-1, :]
    # TODO: Making this more scalable to not use so many commands (A task under design pattern)

    return poly(input_vec), log_posterior, alpha, beta, params


if __name__ == "__main__":
    true_model = reg.Sinusoid()
    n_degree, n_samples = 9, 100  # degree of the polynomial and input samples
    sigma = 0.2  # Gaussian noise parameter

    x_train = np.linspace(0.0, 1.0, n_samples)  # define training data input vector x

    # Gaussian training noise with a fixed seed value.
    noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma, seed=10)

    y_train = true_model(x_train) + noise_train  # training response vector with noise
    dx = np.diff(x_train)[0]
    x_test = 0.5 * dx + x_train[:-1]  # test data input vector x

    # Gaussian testing noise with a fixed seed value
    noise_test = reg.NoiseModel(len(x_test)).gaussian_noise(sigma, seed=42)

    y_test = true_model(x_test) + noise_test  # test response vector with noise

    # number of iterations
    num_iter = 100
    num_iter_axis = np.linspace(1, num_iter, num_iter)

    # call the function
    fit_train, log_posterior_train, alpha, beta, params = calc_map_jeffreys(
        x_train, y_train, n_degree, num_iter
    )
    print(f"alpha={alpha[-1]}, beta={beta[-1]}")
    print(f"Best fit ridge parameter (log scale) is {np.log(alpha[-1]/beta[-1])}")

    # %% plot results
    plt.rc("lines", lw=3)
    plt.rc("font", weight="bold", size=12)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)

    ax = axes[0, 0]
    ax.set_title("Ridge Regression")
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.scatter(x_test, y_test, s=100, color="r", alpha=0.7)
    ax.plot(x_train, true_model(x_train), color="g", label="true model")
    ax.plot(x_train, fit_train, label="training fit")
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$y_n$")
    ax.grid(linestyle="--")
    ax.legend(prop={"size": 12})

    ax = axes[0, 1]
    ax.set_title(r"Log Posterior")
    ax.plot(num_iter_axis, log_posterior_train, label="Log Posterior")
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$ln({w}, \alpha, \beta)$")
    ax.grid(linestyle="--")

    ax = axes[1, 0]
    ax.set_title(r"Noise precision $\beta$")
    ax.plot(num_iter_axis, beta, label="beta")
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\beta$")
    ax.grid(linestyle="--")

    ax = axes[1, 1]
    ax.set_title(r"Prior precision $\alpha$")
    ax.plot(num_iter_axis, alpha, label="alpha")
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(linestyle="--")

    plt.show()
