import numpy as np


class LinearModel:
    """LinearModel
    Abstract class that defines the interface for linear models
    """
    def __init__(self, params):
        self._params = np.array(params)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, values):
        self._params[...] = values

    def __len__(self):
        return len(self._params)

    def __call__(self, x):
        return self._eval(x)

    def _eval(self, x):
        msg = 'Needs to implemented by subclass'
        raise NotImplementedError(msg)

    
class StraightLine(LinearModel):

    def __init__(self, params):
        
        msg = 'A straight line has only two parameters: slope and intercept'
        assert len(params) == 2, msg
    
        super().__init__(params)

    def _eval(self, x):
        return self._params[0] + self._params[1] * x


class Cost:
    """Cost
    Fit criterion that will be minimized to obtain the model that explains
    the data best
    """
    def __init__(self, data, model):

        assert isinstance(model, LinearModel)

        self.data = np.array(data)
        self.model = model

    @property
    def x(self):
        return self.data[:,0]

    @property
    def y(self):
        return self.data[:,1]

    @property
    def residuals(self):
        return self.y - self.model(self.x)

    def __call__(self, params=None):
        if params is not None:
            self.model.params = params
        return self._eval(self.residuals)

    def _eval(self, residuals):
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)

    
class LeastSquares(Cost):

    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals)
