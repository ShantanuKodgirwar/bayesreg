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
    Scoring model quality
    """    
    def __init__(self, model, **kwargs):
        
        assert isinstance(model, LinearModel)
        
        self.model = model
        
        for A, ridge_param in kwargs.items():
                        
            self.A = A
            self.ridge_param = ridge_param

    def _eval(self, params):
        
        msg = 'Needs to be implemented by subclass'
        return NotImplementedError(msg)

class GoodnessOfFit(Cost):
    """GoodnessOfFit
    Fit criterion that will be minimized to obtain the model that explains
    the data best
    """
    def __init__(self, data, model):
        
        assert isinstance(model, LinearModel)
        
        super().__init__(model)
        
        self.data = np.array(data)

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

class LeastSquares(GoodnessOfFit):
    """LeastSquares
    sum-of-squares error term as a cost function
    """
    def _eval(self, residuals):
        return 0.5 * residuals.dot(residuals)

class Gradient(Cost):
    """Gradient
    Compute the gradient of the cost function and calculate the norm of it
    """
    
    pass

class RidgeRegularizer(Cost):
    """RidgeRegularizer
    Implements the general ridge regularization term consisting of a 
    penalizing term 'Ridge_param' and general regulazer term 'A'
    """
    def __init__(self, model, ridge_param, A=None):

        assert isinstance(model, LinearModel)
        
        super().__init__(model)
        
        self._ridge_param = ridge_param 
        
        if A is None:
            A = np.eye(len(model))
        
        else:
            
            # test if the matrix is symmetric
            if A != np.transpose(A):
                raise ValueError('A must be a symmetric matrix')
            
            # eigen values of real symmetric/complex Hermitian matrix
            eig_vals, _ = np.linalg.eigh(A)
            
            # eigen values non-negative to test semi-definite condition
            if eig_vals < 0:
                raise ValueError('A is not a semi-definite matrix')

        self.A = A
    
    @property
    def ridge_param(self):
        return self._ridge_param
        
    def _eval(self, residuals):
        
        params = self.model.params
        
        return 0.5 * self._ridge_param * params.dot(self.A.dot(params))

class SumOfCosts(Cost):
    """SumOfCosts
    Summation of costs from regression analysis
    (Ex: Ordinary Least squares and Ridge Regularizer)
    """

    def __init__(self, model, *costs):

        for cost in costs:
            msg = "{0} should be subclass of Cost".format(cost)
            assert isinstance(cost, Cost), msg
            
            assert cost.model is model
    
        super().__init__(model)
        
        self._costs = costs
    
    def _eval(self, params):
        
        vals = [cost._eval(params) for cost in self._costs]
        
        return np.sum(vals)
    
    def __iter__(self):
        return iter(self._costs)

class Fitter:
    """Fitter
    Fits the data by computing the unknown weights/parameters
    """        
    def __init__(self, cost):
        
        assert isinstance(cost, Cost)
        self.cost = cost
        
    def run(self, *args):
        msg = 'Needs to be implemented by subclass'
        raise NotImplementedError(msg)
        
class LSQEstimator(Fitter):
    """LSQEstimator
    Ordinary least squared estimator that minimizes sum-of-squared residuals 
    and calculates regression parameters  
    """
    def __init__(self, cost):
        
        assert isinstance(cost, LeastSquares)
        
        super().__init__(cost)
        
    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
         
        return np.linalg.pinv(X).dot(y)
    
class SVDEstimator(LSQEstimator):

    def run(self, *args):
        
        cost = self.cost
        model = cost.model
        
        X = model.compute_design_matrix(cost.x)
        y = cost.y
    
        U, L, V = np.linalg.svd(X, full_matrices=False)

        return V.T.dot(U.T.dot(y) / L)
    
class RidgeEstimator(Fitter):
    """RidgeEstimator
    Generalized Ridge regularizer estimator (modified LSQEstimator) that 
    minimizes sum-of-squares residuals
    """        
    def __init__(self, sum_of_costs):
        
        assert isinstance(sum_of_costs, SumOfCosts)
        
        for cost in sum_of_costs:
            
            isinstance(cost, RidgeRegularizer) or isinstance(cost, LeastSquares)
                   
        super().__init__(sum_of_costs)
        
    def run(self, *args):
        
        A = 0.
        b = 0.
        
        for cost in self.cost:
            
            if isinstance(cost, RidgeRegularizer):
                A += cost.ridge_param * cost.A
            else:     
                X = cost.model.compute_design_matrix(cost.x)
                A += X.T.dot(X)
                b += X.T.dot(cost.y)
      
        return np.linalg.inv(A)@b

class NoiseModel:    
    """NoiseModel
    Class that generates different types of noise
    """
                      
    def __init__(self, N):
        
        self._N = int(N) # number of dependent variables used
        
    def gaussian_noise(self, sigma):
        
        return np.random.standard_normal(self._N) * sigma

def rmse(y_data, y_model):
    return np.linalg.norm(y_data-y_model) / np.sqrt(len(y_data))

