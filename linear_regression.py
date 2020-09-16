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

class PolyCurve(LinearModel):
    
    def __init__(self, params):
        
        super().__init__(params)
        
    def _eval(self, x):
        
        # evaluate the polynomial        
        return np.polyval(self._params[::-1], x)
    
class Sinusoid(LinearModel):
    
    def __init__(self):
        super().__init__([1.])
    
    def _eval(self, x):
        return self.params[0] * np.sin(2*np.pi*x)
        

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
    """LeastSquares
    sum-of-squares error term as a cost function
    """
    
    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals)
    
class Ridge(Cost):
    """Ridge
    A modified cost function that adds regularization term (ridge regression) 
    with the sum-of-squares error term 
    """

    def __init__(self, data, model, ridge_param):

        assert isinstance(model, LinearModel)
        
        super().__init__(data, model)
        
        # Ridge parameter lambda determines regularization strength
        self._ridge_param = ridge_param 
    
    @property
    def ridge_param(self):
        return self._ridge_param
        
    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals) + 0.5 * self._ridge_param * np.sum(self.model.params[1:]**2)

class Fitter:
    """Fitter
    Fits the data by estimating the least square estimator or regression coefficients (weights) 
    """        
    def __init__(self, cost):
        
        assert isinstance(cost, Cost)
        self.cost = cost
        
    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)
        
class LSQEstimator(Fitter):
    
    def __init__(self, cost):
        
        assert isinstance(cost, LeastSquares)
        
        super().__init__(cost)
        
    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
         
        return np.linalg.pinv(X).dot(y)
    
class SVDFitter(LSQEstimator):

    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
    
        U, L, V = np.linalg.svd(X, full_matrices=False)

        return V.T.dot(U.T.dot(y) / L)
    
class RidgeEstimator(Fitter):
        
    def __init__(self, cost):
        
        assert isinstance(cost, Ridge)
        
        super().__init__(cost)
        
    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        X_trans = np.transpose(X)
        
        y = cost.y
        ridge_param = cost.ridge_param
        
        # basis vector
        e = np.eye(1,len(model.params),0)
        
        mat_inv = X_trans@X + ridge_param * (np.eye(len(model.params)) - np.transpose(e)@e)
        
        return np.linalg.inv(mat_inv)@X_trans@y

class NoiseModel:
    
    """NoiseModel
    Class that generates noise of different type
    """
                      
    def __init__(self, N):
        
        self._N = int(N) # number of dependent variables used
        
    def gaussian_noise(self, sigma):
        
        return np.random.standard_normal(self._N) * sigma

def rmse(y_data, y_model):
    return np.linalg.norm(y_data-y_model) / np.sqrt(len(y_data))

