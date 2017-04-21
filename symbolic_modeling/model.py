import pandas as pd
import numpy as np

from .expression import Expression, Mono, Interaction, Combination

class Model:
    def __init__(self):
        raise NotImplementedError()
        
    def fit(self, data):
        raise NotImplementedError()
        
    def predict(self, data):
        raise NotImplementedError()
    
    
class LinearModel(Model):
    def __init__(self, explanatory, response):
        self.ex = explanatory
        self.re = response
        self.bhat = None

    def fit(self, data):
        # Construct X matrix
        X = None
        # Construct Y vector
        y = None
        # Solve equation
        self.bhat = np.linalg.solve(X.T @ X, X.T @ y)
        return self.bhat
        
    def predict(self, data):
        # Construct the X matrix
        X = None
        # Multiply the weights to each column and sum across rows
        return (X @ self.bhat).sum(axis = 1)
        
    # static method
    def extract_columns(expr, data):
        if isinstance(expr, Combination):
            pass
        elif isinstance(expr, Interaction):
            pass
        elif isinstance(expr, Mono):
            pass
        else:
            raise Exception("LinearModel only suppoprts expressions consisting of Mono, Interaction, and Combination.")