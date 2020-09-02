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
    
    # def compute_design_matrix(self, x):
        
    #     return np.power.outer(x,range(len(self)))
        
class NoiseModel:
    
    """NoiseModel
    Class that generates noise of different type
    """
                      
    def __init__(self, N):
        
        self._N = int(N) # number of dependent variables used
        
    def gaussian_noise(self, sigma):
        
        return np.random.standard_normal(self._N) * sigma


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
    
class Sinusoid(LinearModel):
    
    def __init__(self):
        super().__init__([1.])
    
    def _eval(self, x):
        return self.params[0] * np.sin(2*np.pi*x)
           
    
def rmse(y_data, y_model):
    return np.linalg.norm(y_data-y_model) / np.sqrt(len(y_data))
    
    
if __name__ == '__main__':
    
    true_model = Sinusoid()
    
    M = 9 #degree of the polynomial
    N = 10 #number of dependent variables
    sigma = 0.3
    
    # params_poly = np.linalg.inv(x)@y    
    # print(np.linalg.norm(y-x@params_poly)) 
    
    # define input vector x and response vector y 
    x = np.linspace(0., 1., N)
    y = true_model(x)
 
    dx = np.diff(x)[0]
    
    # test data
    x_test = 0.5 * dx + x[:-1]
    noise = NoiseModel(len(x_test))
    y_test = true_model(x_test) + noise.gaussian_noise(sigma)
    
    # Introduce some noise
    noise = NoiseModel(N)    
    gauss_noise = noise.gaussian_noise(sigma)
    
      
    # concatenate the x and response vector y mixed with noise
    # will be used for 'training' the model
    data = np.transpose([x,y+gauss_noise])
        
    # calling the linear model
    poly = PolyCurve(np.ones(M))
    
    lsq_line = LeastSquares(data, StraightLine())
    
    # des_mat = DesignMatrix(params_init)
    
    # poly.params = params_poly
    
    fit_poly = LSQEstimator(LeastSquares(data, poly))
    
    params_best = fit_poly.run()
    
    poly.params = params_best
 
 
    print(f'RMSE: {rmse(y,poly(x))}')
    
    training_error =  []
    test_error = []
    test_error2 = []
    training_error2 = []
    
    for M in range(3, N+5):
        
        poly = PolyCurve(np.ones(M))
        
        fitter = LSQEstimator(LeastSquares(data, poly))
        poly.params = fitter.run()
        training_error.append(rmse(y,poly(x)))
        test_error.append(rmse(y_test, poly(x_test)))
        
        fitter2 = SVDFitter(LeastSquares(data, poly))
        poly.params = fitter2.run()
        training_error2.append(rmse(y,poly(x)))
        test_error2.append(rmse(y_test, poly(x_test)))
   
    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    
    ax[0].set_title('generalized inverse')
    ax[0].plot(training_error, label='training error')
    ax[0].plot(test_error, label='test error')
    ax[0].set_xlabel('M')
    ax[0].set_ylabel('$E_{RMS}$')
    ax[0].legend(prop={'size': 12})
    
    ax[1].set_title('SVD')
    ax[1].plot(training_error2, label='training error')
    ax[1].plot(test_error2, label='test error')
    ax[1].set_xlabel('M')
    ax[1].set_ylabel(r'$E_{RMS}$')
    ax[1].legend(prop={'size': 12})
 
    ax[2].scatter(*data.T, label='training data')
    ax[2].scatter(x_test, y_test, label='test data')
    ax[2].plot(x, true_model(x), label='true fit', color='g')
    ax[2].plot(x, poly(x), label='best fit', color='r')
    ax[2].set_xlabel(r'$x_n$')
    ax[2].set_ylabel(r'$y_n$')
    ax[2].legend(loc = 3, prop={'size': 10})
    fig.tight_layout()
    
    
if False:
    
    ## show true model and best fit
    plt.rc('lines', lw=3)
    plt.rc('font', weight='bold', size=12)   
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]    
    ax.scatter(x, y, s=100, color='k', alpha=0.7)
    ax.plot(x, poly(x), color='r', label='Best fit')
    ax.plot(x, np.sin(2*np.pi*x), color='g', label='true model')
    ax.set_xlabel(r'$x_n$')
    ax.set_ylabel(r'$y_n$')
    ax.legend()
    

    # grid for cost function    
    n_grid = int(1e2)
    intercept_axis = np.linspace(-1., 1., n_grid)
    slope_axis = np.linspace(-1., 3., n_grid)
    grid = np.meshgrid(intercept_axis, slope_axis)
    grid = np.reshape(grid, (2, -1)).T
    costs = np.array([lsq_line(params) for params in grid])
    
    ax = axes[1]
    ax.contour(intercept_axis, slope_axis, np.exp(-0.5*costs.reshape(n_grid, -1)))
    ax.scatter(*params_best[:2], s=100, color='k', alpha=0.5, label='best')
    # ax.scatter(*params_true, s=100, color='r', marker='x', alpha=0.5, label='truth')    
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    ax.legend()
    fig.tight_layout()
