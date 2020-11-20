"""
Testing the precision parameter beta and alpha (hyperparameter)
"""
import numpy as np
import matplotlib.pyplot as plt
import bayesian_linear_regression as reg


def calc_map_jeffreys(input_vec, output_vec, n_degree, num_iter):
    """
    Implements the necessary classes to predict values of response vector
    """
    data = reg.Data(np.transpose([input_vec, output_vec]))  # define the data that is passed
    poly = reg.Polynomial(np.ones(n_degree))  # model is defined that is used
    lsq = reg.LeastSquares(poly, data)  # least-squares as cost function
    regularizer = reg.RidgeRegularizer(poly)  # Regularizer term; pass ridge_param = 0
    precision_estimator = reg.PrecisionEstimator(lsq)  # precision parameter based on Jeffreys prior
    hyperparameter_estimator = reg.HyperparameterEstimator(regularizer)  # hyperparameter based on Jeffreys prior
    total_cost = reg.SumOfCosts(poly, lsq, regularizer)  # lsq+regularizer to give the modified cost function
    ridge_estimator = reg.RidgeEstimator(total_cost)  # estimate the modified error function
    max_posterior = reg.MAPJeffreysPrior(ridge_estimator, hyperparameter_estimator, precision_estimator)
    log_posterior, states = max_posterior.run(num_iter)
    params, beta, alpha = list(map(np.array, zip(*states)))
    poly.params = params[-1, :]

    return poly(input_vec), log_posterior, alpha, beta, params


if __name__ == '__main__':

    true_model = reg.Sinusoid()
    n_degree, n_samples = 9, 100  # degree of the polynomial and input samples
    sigma = 0.2  # Gaussian noise parameter

    x_train = np.linspace(0., 1., n_samples)  # define training data input vector x

    # predefined Gaussian noise for training data with sigma = 0.2
    noise_train = np.asarray([0.02333039, 0.05829248, -0.13038691, -0.29317861,
                              -0.01635218, -0.08768144, 0.24820263, -0.08946657,
                              0.36653148, 0.13669558])
    # Gaussian noise
    noise_train = reg.NoiseModel(len(x_train)).gaussian_noise(sigma)

    y_train = true_model(x_train) + noise_train  # training response vector with noise
    dx = np.diff(x_train)[0]
    x_test = 0.5 * dx + x_train[:-1]  # test data input vector x

    # predefined Gaussian noise for testing data with sigma = 0.2
    noise_test = np.asarray([-0.08638868, 0.02850903, -0.67500835, 0.01389309,
                             -0.2408333, 0.05583381, -0.1050192, -0.10009032,
                             0.08420656])
    # Gaussian noise
    noise_test = reg.NoiseModel(len(x_test)).gaussian_noise(sigma)

    y_test = true_model(x_test) + noise_test  # test response vector with noise

    # number of iterations
    num_iter = 100
    num_iter_axis = np.linspace(1, num_iter, num_iter)

    # call the function
    fit_train, log_posterior_train, alpha, beta, params = calc_map_jeffreys(x_train, y_train, n_degree, num_iter)
    print(f'alpha={alpha[-1]}, beta={beta[-1]}')
    print(f'Best fit ridge parameter (log scale) is {np.log(alpha[-1]/beta[-1])}')

    # %% plot results
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)

    ax = axes[0, 0]
    ax.set_title('Ridge Regression')
    ax.scatter(x_train, y_train, s=100, alpha=0.7)
    ax.scatter(x_test, y_test, s=100, color='r', alpha=0.7)
    ax.plot(x_train, true_model(x_train), color='g', label='true model')
    ax.plot(x_train, fit_train, label='training fit')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.grid(linestyle='--')
    ax.legend(prop={'size': 12})

    ax = axes[0, 1]
    ax.set_title(r'Log Posterior')
    ax.plot(num_iter_axis, log_posterior_train, label='Log Posterior')
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$ln({w}, \alpha, \beta)$')
    ax.grid(linestyle='--')

    ax = axes[1, 0]
    ax.set_title(r'Noise precision $\beta$')
    ax.plot(num_iter_axis, beta, label='beta')
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\beta$')
    ax.grid(linestyle='--')

    ax = axes[1, 1]
    ax.set_title(r'Prior precision $\alpha$')
    ax.plot(num_iter_axis, alpha, label='alpha')
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\alpha$')
    ax.grid(linestyle='--')

    plt.show()

