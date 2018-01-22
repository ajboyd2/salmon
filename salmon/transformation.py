import pandas as pd
import numpy as np


# This is a class to package together the logic of how to transform
# with how to display / print the transformation
class Transformation():
    def __init__(self, func, pattern):
        self.func = func
        self.pattern = pattern
        
    def __str__(self):
        return self.pattern
        
    def compose(self, inner):
        return self.pattern.format(inner)
        
    def transform(self, values, training = True):
        return self.func(values)
    
    def copy(self):
        return Transformation(self.func, self.pattern)

class Center(Transformation):
    def __init__(self):
        self.pattern = "{0}-E({0})"
        self.past_mean = 0
        
    def transform(self, values, training = True):
        if training:
            self.past_mean = values.mean()
        
        return values - self.past_mean
    
    def copy(self):
        ret_val = Center()
        ret_val.past_mean = self.past_mean
        return ret_val

class Standardize(Transformation):
    def __init__(self):
        self.pattern = "({0}-E({0}))/Std({0})"
        self.past_mean = 0
        self.past_std = 1
        
    def transform(self, values, training = True):
        if training:
            self.past_mean = values.mean()
            self.past_std = values.std()
            
        return (values - self.past_mean) / self.past_std    
    
    def copy(self):
        ret_val = Standardize()
        ret_val.past_mean, ret_val.past_std = self.past_mean, self.past_std
        return ret_val
    
# Aliases
Sin = lambda i: Transformation(np.sin, "sin({})")
Cos = lambda i: Transformation(np.cos, "cos({})")
Log = lambda i: Transformation(np.log, "log({})")
Log10 = lambda i: Transformation(np.log10, "log({})")
Exp = lambda i: Transformation(np.exp, "exp({})")
Z = lambda i: Standardize()
Cen = lambda i: Center()
Identity = lambda i: Transformation(lambda x: x, "{}")
Increment = lambda i: Transformation(lambda x: x + i, "{}+"+str(i) if i >= 0 else "{}-"+str(-i))
Multiply = lambda i: Transformation(lambda x: x * i, str(i) + "*{}")
Power = lambda i: Transformation(lambda x: x ** i, "{}^" + str(i))

_default_transformations = {
    "sin" : Sin,
    "cos" : Cos,
    "log" : Log,
    "log10" : Log10,
    "exp" : Exp,
    "standardize" : Z,
    "center": Cen,
    "identity": Identity
}