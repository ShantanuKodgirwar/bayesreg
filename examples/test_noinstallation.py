import os
import sys
import time
import numpy as np

pypath = os.path.abspath('../')

if pypath not in sys.path:
    sys.path.insert(0, pypath)

import bayesian_linear_regression as reg


n_params = 20
n_data = 1000

params = np.random.standard_normal(n_params) / np.arange(1, n_params+1)**0.5

model = reg.Polynomial(params)

print(f'#params={len(model)}')

x = np.linspace(0., 1., n_data)

## specialized version
t = time.process_time()
X = model.compute_design_matrix(x)
t = time.process_time() - t

## generic version
t2 = time.process_time()
X2 = super(model.__class__, model).compute_design_matrix(x)
t2 = time.process_time() - t2

print('difference in design matrices:', np.fabs(X-X2).max())
print('computation times: ', t, t2)

