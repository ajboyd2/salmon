import pandas as pd
import numpy as np

class Expression:
    def __init__(self):
        raise Exception("Cannot instantiate a Expression.")

    def __add__(self, other):
        return Combination(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return Combination(self, other)
        
    def __mul__(self, other):
        # This is to prevent having an interaction of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        # This is to prevent having an interaction of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 
    
    def __pow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 

    def __rpow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Mono)
        return self.__pow__(other)
        
    def __ipow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Mono)
        raise NotImplementedError() 
        
class Mono(Expression):

    def __init__(self, name, transformation = 1, coefficient = 1, shift = 0):
        self.name = name
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
    
    def __mul__(self, other):
        if isinstance(other, Mono):
            return(Interaction(self, other))
        elif isinstance(other, (int, long, float)):
            return(Mono(self.name, self.transformation, self.coefficient * other, self.shift))
        else:
            raise Exception("Multiplication of expressions must involve a (term * number) or a (term * term).")
    
    def __imul__(self, other):
        if isinstance(other, Mono):
            return(Interaction(self, other))
        elif isinstance(other, (int, long, float)):
            self.coefficient *= other
            return(self)
        else:
            raise Exception("Inplace multiplication of expressions must involve a (term *= number) or a (term *= term).")
        
    def __pow__(self, other):
        if isinstance(other, (int, long, float)):
            if isinstance(self, Mono):
                return(Mono(self.name, self.transformation + other, self.coefficient, self.shift))
            elif isinstance(self, Interaction):
                return(Interaction(self.e1, self.e2, self.transformation + other, self.coefficient, self.shift)
        else:
            raise Exception("Transforming a term must involve a number like so: (term ** number).")

    def __ipow__(self, other):
        if isinstance(other, (int, long, float)):
            self.transformation += other
            return self
        else:
            raise Exception("Transforming a term must involve a number like so: (term ** number).")
            
class Interaction(Mono):

    def __init__(self, e1, e2, transformation = 1, coefficient = 1, shift = 0):
        if type(e1) is not Expression or type(e2) is not Expression:
            raise Exception("Interaction takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
       
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
        
        