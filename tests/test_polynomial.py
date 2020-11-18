"""
Some basic tests of the functionality of the Polynomial class
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
import bayesian_linear_regression as reg

n_params = 20
n_data = 1000

params = np.random.standard_normal(n_params) / np.arange(1, n_params + 1) ** 0.5

model = reg.Polynomial(params)

print(f'#params={len(model)}')

x = np.linspace(0., 1., n_data)

# specialized version
t = time.process_time()
X = model.compute_design_matrix(x)
t = time.process_time() - t

# generic version
t2 = time.process_time()
X2 = super(model.__class__, model).compute_design_matrix(x)
t2 = time.process_time() - t2

print('difference in design matrices:', np.fabs(X - X2).max())
print('computation times: ', t, t2)


class PolyFitter(reg.LSQEstimator):

    def run(self, *args):
        params = np.polyfit(self.cost.data.input, self.cost.data.output,
                            len(self.cost.model) - 1)

        warnings.simplefilter("ignore", np.RankWarning)

        return params[::-1]


class PolyFitter2(reg.LSQEstimator):
    eps = 1e-10

    def run(self, *args):
        model = self.cost.model
        X = model.compute_design_matrix(self.cost.data.input)
        y = self.cost.data.output

        inv = np.linalg.inv(X.T.dot(X) + self.eps * np.eye(len(model)))

        return inv.dot(X.T.dot(y))


def fit_models(fitter, n_params, test_set=None):
    cost = fitter.cost

    original_model = cost.model
    Model = original_model.__class__

    train_error = []
    test_error = []

    for n in n_params:
        model = Model(np.ones(n))
        cost.model = model
        params = fitter.run()
        model.params = params
        train_error.append(reg.rmse(cost.data.output, model(cost.data.input)))
        if test_set is not None:
            test_error.append(reg.rmse(test_set[:, 1],
                                       model(test_set[:, 0])))

    cost.model = original_model

    train_error = np.array(train_error)
    test_error = np.array(test_error)

    if test_set is not None:
        return train_error, test_error
    else:
        return train_error


# true_model = lambda x: np.sinc(x)
true_model = reg.Sinc()

n_params = 20
n_data = 20
x_range = (-1., 1.)

# generate a random model
params = np.random.standard_normal(n_params) / np.arange(1, n_params + 1) ** 0.5
model = reg.Polynomial(params)

x = np.linspace(*(x_range + (n_data,))) * 10.
y = true_model(x)

dx = x[1] - x[0]
x_test = 0.5 * dx + x[:-1]
y_test = true_model(x_test)

test_set = np.transpose([x_test, y_test])

# evaluate models on finer grid

X = np.linspace(x.min(), x.max(), 5 * len(x))
Y = true_model(X)

params_true = np.polyfit(x, y, n_params - 1)[::-1]
model.params = params_true

data = reg.Data(np.transpose([x, y]))
lsq = reg.LeastSquares(model, data)
fitter = reg.LSQEstimator(lsq)
fitter = PolyFitter2(lsq)
fitter2 = PolyFitter(lsq)
theta = fitter.run()
theta2 = fitter2.run()

n_params_fitter = np.arange(1, 31, dtype='i')
train_error, test_error = fit_models(fitter, n_params_fitter, test_set)
train_error2, test_error2 = fit_models(fitter2, n_params_fitter, test_set)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].set_title(fitter.__class__.__name__)
ax[0].plot(n_params_fitter, train_error, label='train')
ax[0].plot(n_params_fitter, test_error, label='test')

ax[1].set_title(fitter2.__class__.__name__)
ax[1].plot(n_params_fitter, train_error2, label='train')
ax[1].plot(n_params_fitter, test_error2, label='test')

for a in ax:
    a.set_ylim(-0.1, 1.)
    a.legend()

fig.tight_layout()

print('polyfit vs our:', reg.rmse(params_true, theta))
print('polyfit vs our:', reg.rmse(params_true, theta2))

plt.figure()
plt.scatter(x, y, s=100, color='r', alpha=0.5)
plt.plot(X, model(X), label='polynomial fit')
plt.plot(X, Y, label='true model')
plt.ylim(Y.min() * 1.1, Y.max() * 1.1)
plt.legend()

A = model.compute_design_matrix(x)
A2 = np.power.outer(x, np.arange(n_params))

print(np.fabs(A - A2).max())
print(np.linalg.cond(A))
