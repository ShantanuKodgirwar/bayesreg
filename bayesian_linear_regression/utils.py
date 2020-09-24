"""
Collection of useful functions
"""
import numpy as np

def rmse(y_data, y_model):
    return np.linalg.norm(y_data-y_model) / np.sqrt(len(y_data))

