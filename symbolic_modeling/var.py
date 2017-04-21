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
        
    def __str__(self):
        # Every inherited class of term must have a way
        # representing itself as a string
        raise NotImplementedError()
    
    def flatten(self):
        # For taking the paired terms (Interaction and Combination)
        # and returning a list of the variables being multiplied or
        # added separately
        raise NotImplementedError()
        
class Mono(Expression):

    def __init__(self, name, transformation = 1, coefficient = 1, shift = 0):
        self.name = name
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
    
    def __mul__(self, other):
        if isinstance(other, Mono):
            return Interaction(self, other) 
        elif isinstance(other, (int, float)):
            return Mono(self.name, self.transformation, self.coefficient * other, self.shift)
        else:
            raise Exception("Multiplication of expressions must involve a (term * number) or a (term * term).")
    
    def __imul__(self, other):
        if isinstance(other, Mono):
            return Interaction(self, other) 
        elif isinstance(other, (int, float)):
            self.coefficient *= other
            return self 
        else:
            raise Exception("Inplace multiplication of expressions must involve a (term *= number) or a (term *= term).")
        
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self, Mono):
                return Mono(self.name, self.transformation * other, self.coefficient, self.shift)
            elif isinstance(self, Interaction):
                return Interaction(self.e1, self.e2, self.transformation * other, self.coefficient, self.shift)
        else:
            raise Exception("Transforming a term must involve a number like so: (term ** number).")

    def __ipow__(self, other):
        if isinstance(other, (int, float)):
            self.transformation += other
            return self
        else:
            raise Exception("Transforming a term must involve a number like so: (term ** number).")
    
    def __str__(self):
        base = "%s"
        params = [self.name]
        
        shift_flag = False
        
        if self.shift > 0:
            base = "(" + base + "+%s)"
            params.append(self.shift)
            shift_flag = True
        elif self.shift < 0:
            base = "(" + base + "-%s)"
            params.append(str(self.shift * -1))
            shift_flag = True
            
        if self.transformation != 1:
            if shift_flag:
                base = base + "^%s"
            else:
                base = "(" + base + ")^%s"
            params.append(str(self.transformation))
            
        if self.coefficient != 1:
            base = "%s*" + base
            params.insert(0, str(self.coefficient))

        return base % tuple(params)

    def flatten(self):
        return [self]
        
class Interaction(Mono):

    def __init__(self, e1, e2, transformation = 1, coefficient = 1, shift = 0):
        if not (isinstance(e1, Expression) and isinstance(e2, Expression)):
            raise Exception("Interaction takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
        
    def __str__(self):
        vars = self.flatten()
        output = "{"
        for var in vars:
            output += str(var) + "}{"
        return output[:-1]
        
    def flatten(self):
        return self.e1.flatten() + self.e2.flatten()
        
       
class Combination(Expression):

    def __init__(self, e1, e2):
        if not (isinstance(e1, Expression) and isinstance(e2, Expression)):
            raise Exception("Combination takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2   

    def __str__(self):
        return str(self.e1) + " + " + str(self.e2)
        
    def flatten(self):
        return self.e1.flatten() + self.e2.flatten()        

def Poly(var, power):
    if isinstance(var, str):
        base = Mono(var)
    elif isinstance(var, Mono):
        base = var
    else:
        raise Exception("Poly takes a (str or Mono) and a positive int.")
        
    if not (isinstance(power, int) and power > 0):
        raise Exception("Poly takes a (str or Mono) and a positive int.")
    
    if power == 1:
        return base
    else:
        return base ** power + Poly(base, power - 1)
        