import pandas as pd
import numpy as np


# This is a class to package together the logic of how to transform
# with how to display / print the transformation

class Transformation():
    ''' A Trasformation object holds the actual function to calculate transformations 
    on data as well as some helper information for printing and visualizing. 
    '''
    
    def __init__(self, func, pattern, name, inverse = None):
        ''' Creates a Transformation object.

        Arguments:
            func - A function to be applied to a column of data in a DataFrame.
            pattern - A str holding a template for printing.
            name - A str describing the transformation.
            inverse - An optional function that will undo the func operation.
        '''
        self.func = func
        self.pattern = pattern
        self.inverse = inverse
        self.name = name
        
    def __str__(self):
        ''' Returns the given pattern for debugging. '''
        return self.pattern

    def __eq__(self, other):
        ''' Check if two objects are equivalent. 

        Arguments:
            other - An object.

        Returns:
            A boolean, True if other is a Transformation and equivalent, False if else.
        '''
        if isinstance(other, Transformation):
            return self.pattern == other.pattern and self.name == other.name
        else:
            return False

    def __hash__(self):
        ''' Hash the transformation for the purposes of storing in sets and dictionaries.

        Returns:
            A real value representing the hash of the Transformation.
        '''

        return hash((self.pattern, self.name))
        
    def compose(self, inner):
        ''' Applies specific data to the pattern for printing. 
        
        Arguments:
            inner - An object to be injested by str.format()
        
        Returns:
            A processed and evaluated str representing the Trasnformation.
        '''
        return self.pattern.format(inner)
        
    def transform(self, values, training = True):
        ''' Apply the function to the data.

        Arguments:
            values - A Series object that is the data to trasnform.
            training - A flag to indicate is this transformation is during training or not. Default is True.

        Returns:
            A trasnformed Series. 
        '''
        return self.func(values)
    
    def copy(self):
        ''' Returns a deep copy of the Transformation. '''
        return Transformation(self.func, self.pattern, self.name, self.inverse)

    def invert(self, data):
        ''' If available, invert the data.

        Arguments:
            data - A Series object holding the data to invert.

        Returns:
            A Series object that has the inverted data.
        '''
        if self.inverse is None:
            raise Exception("Inverse not defined for " + self.name + " transformation.")

        return self.inverse(data)

class Center(Transformation):
    ''' A specific type of Trasnformation for centering data so that it has a mean of 0. '''

    def __init__(self):
        ''' Create a Center object. '''
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
    ''' A specific type of Transformation that standardizes the data so that it has a mean of 0 and standard deviation of 1. '''
    def __init__(self):
        ''' Create a Standardize object. '''
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
    
# Aliases for common Transformations
Sin = lambda i: Transformation(np.sin, "sin({})", "Sine")
Cos = lambda i: Transformation(np.cos, "cos({})", "Cosine")
Log = lambda i: Transformation(np.log, "log({})", "Natural Log", np.exp)
Log10 = lambda i: Transformation(np.log10, "log({})", "Log Base 10", lambda x: 10 * x)
Exp = lambda i: Transformation(np.exp, "exp({})", "Exponential", np.log)
Std = lambda i: Standardize()
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
    "standardize" : Std,
    "center": Cen,
    "identity": Identity
}