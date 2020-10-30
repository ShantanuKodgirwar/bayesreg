"""
Collection of useful functions
"""
import numpy as np
from scipy.optimize import approx_fprime


def rmse(y_data, y_model):
    return np.linalg.norm(y_data - y_model) / np.sqrt(len(y_data))


def calc_gradient(f, x, eps=1e-7):
    """
    Estimate gradient of scalar valued function by computing
    finite differences

    Parameters
    ----------
    f : callable
        DESCRIPTION.
    x : numpy.array
        DESCRIPTION.
    eps : positive float
        Increment for computing finite differences. The default is 1e-7.

    Returns
    -------
    numpy.array
        Estimated gradient.

    """
    return approx_fprime(x, f, eps)


def calc_hessian(f, x, eps=1e-7):
    """
    Estimate Hessian of scalar valued function by computing
    2nd order finite differences

    Parameters
    ----------
    f : callable
        DESCRIPTION.
    x : numpy.array
        DESCRIPTION.
    eps : positive float
        Increment for computing finite differences. The default is 1e-7.

    Returns
    -------
    numpy array
        Estimated Hessian matrix.
    """
    H = []
    for i in range(len(x)):
        partial_derivative = lambda x: calc_gradient(f, x, eps)[i]
        H.append(calc_gradient(partial_derivative, x, eps))
    H = np.transpose(H)

    return H
