import os
import sys

pypath = os.path.abspath('../')

if pypath not in sys.path:
    sys.path.insert(0, pypath)

import bayesian_linear_regression as reg

