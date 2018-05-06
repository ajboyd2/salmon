import pandas as pd
import numpy as np


# This is a class to package together the logic of how to transform
# with how to display / print the transformation
class Transformation():
    def __init__(self, func, pattern, name, inverse = None):
        self.func = func
        self.pattern = pattern
        self.inverse = inverse
        self.name = name
        
    def __str__(self):
        return self.pattern
        
    def compose(self, inner):
        return self.pattern.format(inner)
        
    def transform(self, values, training = True):
        return self.func(values)
    
    def copy(self):
        return Transformation(self.func, self.pattern, self.name, self.inverse)

    def invert(self, data):
        if self.inverse is None:
            raise Exception("Inverse not defined for " + self.name + " transformation.")

        return self.inverse(data)

class Center(Transformation):
    def __init__(self):
        self.pattern = "{0}-E({0})"
        self.past_mean = 0
        self.name = "Center"
        
    def transform(self, values, training = True):
        if training:
            self.past_mean = values.mean()
        
        return values - self.past_mean
    
    def copy(self):
        ret_val = Center()
        ret_val.past_mean = self.past_mean
        return ret_val

    def invert(self, data):
        return data + self.past_mean

class Standardize(Transformation):
    def __init__(self):
        self.pattern = "({0}-E({0}))/Std({0})"
        self.past_mean = 0
        self.past_std = 1
        self.name = "Standardize"
        
    def transform(self, values, training = True):
        if training:
            self.past_mean = values.mean()
            self.past_std = values.std()
            
        return (values - self.past_mean) / self.past_std    
    
    def copy(self):
        ret_val = Standardize()
        ret_val.past_mean, ret_val.past_std = self.past_mean, self.past_std
        return ret_val

    def invert(self, data):
        return (data * self.past_std) + self.past_mean
    
# Aliases
Sin = lambda i: Transformation(np.sin, "sin({})", "Sine",  np.arcsin)
Cos = lambda i: Transformation(np.cos, "cos({})", "Cosine", np.arccos)
Log = lambda i: Transformation(np.log, "log({})", "Natural Log", np.exp)
Log10 = lambda i: Transformation(np.log10, "log({})", "Log Base 10", lambda x: 10 * x)
Exp = lambda i: Transformation(np.exp, "exp({})", "Exponential", np.log)
Z = lambda i: Standardize()
Cen = lambda i: Center()
Identity = lambda i: Transformation(lambda x: x, "{}", "Identity", lambda x: x)
Increment = lambda i: Transformation(lambda x: x + i, "{}+"+str(i) if i >= 0 else "{}-"+str(-i), "Increment", lambda x: x - i)
Multiply = lambda i: Transformation(lambda x: x * i, str(i) + "*{}", "Multiply", lambda x: x * (1/i))
Power = lambda i: Transformation(lambda x: x ** i, "{}^" + str(i), "Power", lambda x: x ** (1/i) if i % 2 == 1 else x.clip(0, None) ** (1/i))

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