import pandas as pd
import numpy as np

class Var:

    def __init__(self, name):
        self.name = name
        
class Combination(Var):

    def __init__(self, vars, weights = None):
        # By default produces a linear combination of the variables | names
        
        if type(vars) is not list or type(vars) is not Var:
            raise Exception("Combination takes a list of Var objects or strings when initializing.")
        
        if type(vars) is list and type(vars) is not str:
            for var in vars:
                if type(var) is not Var or type(var) is not str:
                    raise Exception("Combination takes a list of Var objects or strings when initializing.")
        
        self.vars = [var if type(var) is Var else Var(var) for var in vars] # Convert all strings in list to vars
        
        if weights is None:
            self.weights = [1 for _ in vars]
        else:
            self.weights = weights
           

class Poly(Combination):

