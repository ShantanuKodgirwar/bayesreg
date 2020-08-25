import numpy as np
import matplotlib.pyplot as plt

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

class PolyCurve(LinearModel):
    
    def __init__(self, params, M):
        
        super().__init__(params)
        
        self.M = M 
        
        assert len(params) == M+1
     
    def _eval(self, x):
        
        # evaluate the polynomial 
        
        return np.polyval(self._params[::-1], x)

class ComputeParameters:
    """
    Class that provides the input (design matrix), output
    (response vector) and estimated polynomial regression coefficients
    """
    def __init__(self, N, M, sigma):
        
        self._N = int(N) # number of dependent variable
        self._M = int(M) # degree of the polynomial
        self._sigma = sigma
    
    def get_input(self):
        
        # Forming the design matrix X
        
        return np.power.outer(np.linspace(0., 1., self._N), range(self._M+1))

    def get_output(self):
        
        # creating a response vector y
        
        noise = np.random.standard_normal(self._N) * self._sigma
        temp = np.linspace(0., 1., self._N)
        
        return np.add(noise, temp)
    

    def compute_params(self, x, y):
        
        # Implementing the normal equation for estimating regression coefficents
                            
        return np.linalg.pinv(x.T@x)@x.T@y 

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
    
    
if __name__ == '__main__':
    
    
    M = 4 #degree of the polynomial
    N = 20 #number of dependent variables
    sigma = 0.2
    
    par = ComputeParameters(N, M, sigma)
    
    x = par.get_input()
    y = par.get_output()
        
    params_poly = par.compute_params(x,y)
    print("The regression coefficients are",params_poly)
    
    data = np.concatenate([np.reshape(x[:,1],[len(x),-1]), np.reshape(y,[len(y),-1])], axis = 1)
        
    poly = PolyCurve(params_poly, M)
    
    poly.params = params_poly
    
    # true fit 
    params_true = [0., 1.]
    line = StraightLine(params_true)
        
    ## show true model and best fit
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)   
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]    
    ax.scatter(x[:,1], y, s=100, color='k', alpha=0.7)
    ax.plot(x[:,1], poly(x[:,1]),color='k', label='true model')
    ax.plot(x[:,1], line(x[:,1]),color='r', label='Best fit')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()
    
    
    # fit criteria
    
    lsq = LeastSquares(data, line)

    # grid for cost function
    
    n_grid = int(1e2)
    intercept_axis = np.linspace(-1., 1., n_grid)
    slope_axis = np.linspace(-1., 3., n_grid)
    grid = np.meshgrid(intercept_axis, slope_axis)
    grid = np.reshape(grid, (2, -1)).T
    costs = np.array([lsq(params) for params in grid])
    
    ax = axes[1]
    ax.contour(intercept_axis, slope_axis, np.exp(-0.5*costs.reshape(n_grid, -1)))
    ax.scatter(*params_poly[:2], s=100, color='r', alpha=0.5, label='best')
    ax.scatter(*params_true, s=100, color='k', marker='x', alpha=0.5, label='truth')    
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    ax.legend()
    fig.tight_layout()
