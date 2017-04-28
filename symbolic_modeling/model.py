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
        self.training_x = None
        self.trainign_y = None

    def fit(self, data):
        # Construct X matrix
        X = LinearModel.extract_columns(self.ex, data)
        X = pd.concat([LinearModel.ones_column(data), X], axis = 1)
        self.training_x = X
        # Construct Y vector
        y = LinearModel.extract_columns(self.re, data)
        self.training_y = y
        # Ensure correct dimensionality of y
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise Exception("Response variable of linear model can only be a single term expression.")
        # Solve equation
        self.bhat = pd.DataFrame(np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)), 
                                 index=X.columns, columns = ["Weights"])
        return self.bhat
        
    def predict(self, data):
        # Construct the X matrix
        X = LinearModel.extract_columns(self.ex, data)
        X = pd.concat([LinearModel.ones_column(data), X], axis = 1)
        # Multiply the weights to each column and sum across rows
        return pd.DataFrame({"Predicted " + str(self.re) : np.dot(X, self.bhat).sum(axis = 1)})
        
    # static method
    def ones_column(data):
        return pd.DataFrame({"Intercept" : np.repeat(1, data.shape[0])})
        
    # static method
    def extract_columns(expr, data):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise Exception("Only DataFrames and Series are supported for LinearModel.")
    
        if isinstance(expr, Combination):
            columns = [LinearModel.extract_columns(e, data) for e in expr.flatten()]
            return pd.concat(columns, axis = 1)
            
        elif isinstance(expr, Interaction):
            columns = [LinearModel.extract_columns(e, data) for e in expr.flatten(True)]
            product = pd.DataFrame({str(expr) : pd.concat(columns, axis = 1).prod(axis = 1)})
            return LinearModel.transform(expr, product)
        
        elif isinstance(expr, Mono):
            if expr.name not in list(data):
                raise Exception("Variable { " + expr.name + " } not found within data.")
            return LinearModel.transform(expr, pd.DataFrame({str(expr) : data[expr.name]}))
            
        else:
            raise Exception("LinearModel only suppoprts expressions consisting of Mono, Interaction, and Combination.")
            
    # static method
    def transform(expr, data):
        if not isinstance(expr, Mono):
            raise Exception("Transformation of data is only supported on singular terms of expressions.")
        
        return ((data + expr.shift) * expr.coefficient) ** expr.transformation

        