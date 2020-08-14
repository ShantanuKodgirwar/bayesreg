import numpy as np
import matplotlib.pyplot as plt

from linear_regression import StraightLine, LeastSquares

params_true = [0., 1.]

line = StraightLine(params_true)

## generate some noisy data

N = 10
x = np.linspace(0., 1., N)
sigma = 0.2
noise = np.random.standard_normal(N) * sigma

y = line(x)
data = np.transpose([x, y+noise])

## fit criterion

lsq = LeastSquares(data, line)

# =============================================================================
# ## find best fit by brute force using a grid search
# 
# n_grid = int(1e2)
# 
# intercept = np.linspace(-1., 1., n_grid)
# slope = np.linspace(0., 2., n_grid)
# grid = np.meshgrid(intercept, slope)
# grid = np.reshape(grid, (2, -1)).T
# 
# costs = np.array([lsq(params) for params in grid])
# params_best = grid[np.argmin(costs)]
# 
# line.params = params_best
# =============================================================================

## finding slope and intercept using algebra

slope_num = N*np.sum(x*(y+noise)) - np.sum(x)*np.sum(y+noise)
slope_den = N*np.sum(x*x) - np.sum(x)*np.sum(x)
slope = slope_num/slope_den
intercept = np.mean(y+noise) - slope*np.mean(x)
params_best = [intercept, slope]


# grid for cost function

n_grid = int(1e2)
intercept_axis = np.linspace(-1., 1., n_grid)
slope_axis = np.linspace(0., 2., n_grid)
grid = np.meshgrid(intercept_axis, slope_axis)
grid = np.reshape(grid, (2, -1)).T
costs = np.array([lsq(params) for params in grid])

line.params = params_best

## show true model and best fit
plt.rc('lines', lw=3)
plt.rc('font', weight='bold', size=12)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.scatter(*data.T, s=100, color='k', alpha=0.7)
ax.plot(x, y, color='r', label='true model')
ax.plot(x, line(x), color='k', label='best fit')
ax.set_xlabel(r'$x_n$')
ax.set_ylabel(r'$y_n$')
ax.legend()
ax = axes[1]
ax.contour(intercept_axis, slope_axis, np.exp(-0.5*costs.reshape(n_grid, -1)))
ax.scatter(*params_best, s=100, color='r', alpha=0.5, label='best')
ax.scatter(*params_true, s=100, color='k', marker='x', alpha=0.5, label='truth')


ax.set_xlabel('intercept')
ax.set_ylabel('slope')
ax.legend()
fig.tight_layout()
