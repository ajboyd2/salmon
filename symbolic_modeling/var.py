import pandas as pd
import numpy as np

class Expression:
    def __init__(self):
        raise Exception("Cannot instantiate a Expression.")

    def __add__(self, other):
        return Combination(self, other)

    def __mult__(self, other):
        # This is to prevent having an interaction of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 
    
    def __pow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 
    

class Mono(Expression):

    def __init__(self, name, transformation = None, coefficient = 1, shift = 0):
        self.name = name
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
        
class Interaction(Expression):

    def __init__(self, e1, e2):
        if type(e1) is not Expression or type(e2) is not Expression:
            raise Exception("Interaction takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2
       
class Combination(Expression):

    def __init__(self, e1, e2):
        if type(e1) is not Expression or type(e2) is not Expression:
            raise Exception("Combination takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2       

class Poly(Combination):
    
    def __init__(self, var, power):
        if type(var) is str:
            base = Mono(var)
        elif type(var) is Mono:
            base = var
        else:
            raise Exception("Poly takes a (str or Mono) and an int.")
            
        if type(power) is not int or type(power) is not long:
            raise Exception("Poly takes a (str or Mono) and an int.")
        
        