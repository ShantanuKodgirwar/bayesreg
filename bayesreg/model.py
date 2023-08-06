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

    def __call__(self, input_val):
        return self._eval(input_val)

    def _eval(self, input_val):
        msg = "Needs to be implemented by subclass"
        raise NotImplementedError(msg)

    def compute_design_matrix(self, input_val):
        current_params = self.params.copy()
        design_matrix = np.empty((len(input_val), len(self)))
        for i, params in enumerate(np.eye(len(self))):
            self.params = params
            design_matrix[:, i] = self(input_val)

        self.params = current_params

        return design_matrix


class StraightLine(LinearModel):
    def __init__(self, params=[0.0, 1.0]):
        msg = "A straight line has only two parameters: slope and intercept"
        assert len(params) == 2, msg

        super().__init__(params)

    def _eval(self, input_val):
        return self._params[0] + self._params[1] * input_val


class Polynomial(LinearModel):
    def _eval(self, input_val):
        return np.polyval(self._params[::-1], input_val)

    def compute_design_matrix(self, input_val):
        return np.power.outer(input_val, np.arange(len(self)))


class Sinusoid(LinearModel):
    """Sinusoid

    Mostly used for testing
    """

    def __init__(self):
        super().__init__([1.0])

    def _eval(self, input_val):
        return self.params[0] * np.sin(2 * np.pi * input_val)


class Sinc(LinearModel):
    """Sinc
    Used for testing
    """

    def __init__(self):
        super().__init__([1.0])

    def _eval(self, input_val):
        return self.params[0] * np.sinc(input_val)
