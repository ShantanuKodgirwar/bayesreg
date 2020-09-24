"""
Various noise models
"""
import numpy as np

class NoiseModel:    
    """NoiseModel

    Class that generates different types of noise
    """                      
    def __init__(self, N):
        self._N = int(N) # number of dependent variables used
        
    def gaussian_noise(self, sigma):
        
        return np.random.standard_normal(self._N) * sigma

