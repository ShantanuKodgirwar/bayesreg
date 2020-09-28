"""
Forward models
"""
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
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)

    def compute_design_matrix(self, x):
        
        current_params = self.params.copy()
        matrix = np.empty((len(x), len(self)))
        for i, params in enumerate(np.eye(len(self))):
            self.params = params
            matrix[:,i] = self(x)
        
        self.params = current_params
        
        return matrix

    
class StraightLine(LinearModel):

    def __init__(self, params=[0., 1.]):
        
        msg = 'A straight line has only two parameters: slope and intercept'
        assert len(params) == 2, msg
    
        super().__init__(params)

    def _eval(self, x):
        return self._params[0] + self._params[1] * x

    
class Polynomial(LinearModel):
    
    def _eval(self, x):
        return np.polyval(self._params[::-1], x)

    def compute_design_matrix(self, x):
        return np.power.outer(x, np.arange(len(self)))
    
    
class Sinusoid(LinearModel):
    """Sinusoid

    Mostly used for testing
    """
    def __init__(self):
        super().__init__([1.])
    
    def _eval(self, x):
        return self.params[0] * np.sin(2*np.pi*x)
        
class Sinc(LinearModel):
    """Sinc
    Used for testing
    """
    def __init__(self):
        super().__init__([1.])
        
    def _eval(self, x):
        return self.params[0] * np.sinc(x)
    

