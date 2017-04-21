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
        X = extract_columns(self.ex, data)
        # Construct Y vector
        y = extract_columns(self.re, data)
        # Ensure correct dimensionality of y
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise Exception("Response variable of linear model can only be a single term expression.")
        # Solve equation
        self.bhat = np.linalg.solve(X.T @ X, X.T @ y)
        return self.bhat
        
    def predict(self, data):
        # Construct the X matrix
        X = extract_columns(self.ex, data)
        # Multiply the weights to each column and sum across rows
        return (X @ self.bhat).sum(axis = 1)
        
    # static method
    def extract_columns(expr, data):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise Exception("Only DataFrames and Series are supported for LinearModel.")
    
        if isinstance(expr, Combination):
            columns = [extract_columns(e, data) for e in expr.flatten()]
            return pd.concat(columns, axis = 1)
            
        elif isinstance(expr, Interaction):
            columns = [extract_columns(e, data) for e in expr.flatten(True)]
            product = pd.concat(columns, axis = 1).prod(axis = 0)
            return transform(expr, product)
        
        elif isinstance(expr, Mono):
            if expr.name not in list(data):
                raise Exception("Variable { " + expr.name + " } not found within data.")
            return transform(expr, data[expr.name])
            
        else:
            raise Exception("LinearModel only suppoprts expressions consisting of Mono, Interaction, and Combination.")
            
    # static method
    def transform(expr, data):
        if not isinstance(expr, Mono):
            raise Exception("Transformation of data is only supported on singular terms of expressions.")
        
        return ((data + expr.shift) * expr.coefficient) ** expr.transformation

        