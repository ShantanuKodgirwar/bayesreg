"""
Testing polynomial model
"""

import bayesian_linear_regression as reg
import numpy as np

n_data = 20
n_deg  = 2   ## straight line

intercept = 0.
slope = 1.

params = np.array([intercept, slope])
poly = reg.PolyCurve(params, len(params)-1)
line = reg.StraightLine(params)

## straight line and polynomial of degree one should agree

x = np.linspace(0., 1, n_data)
max_diff = np.fabs(poly(x) < line(x)).max()
tolerance = 1e-10
print('line and poly agree? - {0}'.format(max_diff < tolerance))

## test design matrix

n_deg = 5
params = np.random.standard_normal(n_deg + 1)
poly = reg.PolyCurve(params, len(params) - 1)

design_matrix = reg.ComputeParameters(n_data, n_deg, sigma=0.)

X = design_matrix.get_input()
assert np.shape(X) == (n_data, n_deg+1)

## compute design matrix by using PolyCurve

params_orig = poly.params.copy()

x = X[:,1].copy()
Y = []

## switch on only a single expansion coefficient to compute
## design matrix with instance of PolyCurve

for params in np.eye(n_deg+1):
    poly.params = params
    Y.append(poly(x))

Y = np.transpose(Y)

max_diff = np.fabs(X - Y).max()

print('design matrices agree? - {0}'.format(max_diff < tolerance))

## set original model parameters

poly.params = params_orig

## is output correct?

y = design_matrix.get_output(poly.params) 
y2 = poly(x)

max_diff = np.fabs(y - y2).max()

## remember that sigma=0. in design_matrix, therefore 'y' and 'y2' should
## agree

print('outputs agree? - {0}'.format(max_diff < tolerance))

## trying to correct computation of output

class ComputeParameters(reg.ComputeParameters):
    """
    Correct version ... but functionality should be split into separate
    objects nevertheless
    """
    def get_output(self, params):
        """
        Output is X*w + n where 'X' is the design matrix, 'w' is the
        weight vector and 'n' is the noise
        """
        noise = np.random.standard_normal(self._N) * self._sigma
        X = self.get_input()
        y = X.dot(params)

        return y + noise

design_matrix = ComputeParameters(n_data, n_deg, sigma=0.)

y = design_matrix.get_output(poly.params)

max_diff = np.fabs(y - y2).max()

print('outputs agree? - {0}'.format(max_diff < tolerance))

