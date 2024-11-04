"""
Testing the straight line fitting and the cost value
"""
import matplotlib.pyplot as plt
import numpy as np

import bayesreg as reg

params_true = [0.0, 1.0]

line = reg.StraightLine(params_true)

# generate some noisy data

N = 11
x = np.linspace(0.0, 1.0, N)
sigma = 0.2
noise = np.random.standard_normal(N) * sigma

y = line(x)
data = reg.Data(np.transpose([x, y + noise]))

# fit criterion

lsq = reg.GaussianLikelihood(line, data)

# finding slope and intercept using algebra
slope_num = N * np.sum(x * (y + noise)) - np.sum(x) * np.sum(y + noise)
slope_den = N * np.sum(x * x) - np.sum(x) * np.sum(x)
slope = slope_num / slope_den
intercept = np.mean(y + noise) - slope * np.mean(x)
params_best = [intercept, slope]

# grid for cost function
n_grid = int(1e2)
intercept_axis = np.linspace(-1.0, 1.0, n_grid)
slope_axis = np.linspace(0.0, 2.0, n_grid)
grid = np.meshgrid(intercept_axis, slope_axis)
grid = np.reshape(grid, (2, -1)).T
costs = np.array([lsq(params) for params in grid])

# params_best = grid[np.argmin(costs)] # Minimizing the value from the grid

# Best params value
line.params = params_best

# show true model and best fit
plt.rc("lines", lw=3)
plt.rc("font", weight="bold", size=12)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(hspace=0.3)

ax = axes[0]
ax.scatter(data.input, data.output, s=100, color="k", alpha=0.7)
ax.set_title(r"Simple Linear Regression")
ax.plot(x, y, color="r", label="true model")
ax.plot(x, line(x), color="k", label="best fit")
ax.set_xlabel(r"$x_n$")
ax.set_ylabel(r"$y_n$")
ax.grid(linestyle="--")
ax.legend(prop={"size": 12})

ax = axes[1]
ax.set_title(r"Contour Plot")
ax.contour(intercept_axis, slope_axis, np.exp(-0.5 * costs.reshape(n_grid, -1)))
ax.scatter(*params_best, s=100, color="r", alpha=0.5, label="best")
ax.scatter(*params_true, s=100, color="k", marker="x", alpha=0.5, label="truth")
ax.set_xlabel("intercept")
ax.set_ylabel("slope")
ax.grid(linestyle="--")
ax.legend(prop={"size": 12})

plt.show()
