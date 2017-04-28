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
        # Expressions other than single variables (Var)
        raise NotImplementedError() 
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        # This is to prevent having an interaction of 
        # Expressions other than single variables (Var)
        raise NotImplementedError() 
    
    def __pow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Var)
        raise NotImplementedError() 

    def __rpow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Var)
        return self.__pow__(other)
        
    def __ipow__(self, other):
        # This is to prevent having a transformation of 
        # Expressions other than single variables (Var)
        raise NotImplementedError() 
        
    def __str__(self):
        # Every inherited class of term must have a way
        # representing itself as a string
        raise NotImplementedError()
    
    def flatten(self, separate_interactions = False):
        # For taking the paired terms (Interaction and Combination)
        # and returning a list of the variables being multiplied or
        # added separately
        raise NotImplementedError()
        
class Var(Expression):

    def __init__(self, name, transformation = 1, coefficient = 1, shift = 0):
        self.name = name
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Var(self.name, self.transformation, self.coefficient, self.shift + other)
        else:
            return super().__add__(other)
            
    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self.shift += other
            return self
        else:
            return super().__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, Var):
            return Interaction(self, other) 
        elif isinstance(other, (int, float)):
            return Var(self.name, self.transformation, self.coefficient * other, self.shift)
        else:
            raise Exception("Multiplication of expressions must involve a (term * number) or a (term * term).")
    
    def __imul__(self, other):
        if isinstance(other, Var):
            return Interaction(self, other) 
        elif isinstance(other, (int, float)):
            self.coefficient *= other
            return self 
        else:
            raise Exception("Inplace multiplication of expressions must involve a (term *= number) or a (term *= term).")
        
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self, Var):
                return Var(self.name, self.transformation * other, self.coefficient, self.shift)
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

    def flatten(self, separate_interactions = False):
        return [self]
        
class Interaction(Var):

    def __init__(self, e1, e2, transformation = 1, coefficient = 1, shift = 0):
        if not (isinstance(e1, Expression) and isinstance(e2, Expression)):
            raise Exception("Interaction takes two Expressions for initialization.")

        self.e1 = e1
        self.e2 = e2
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
        
    def __str__(self):
        vars = self.flatten(True)
        output = "{"
        for var in vars:
            output += str(var) + "}{"
        return output[:-1]
        
    def flatten(self, separate_interactions = False):
        if separate_interactions:
            return self.e1.flatten() + self.e2.flatten()
        else:
            return [self]
       
class Combination(Expression):

    def __init__(self, e1, e2 = None):
        if e2 is None:
            es = e1
            if not isinstance(es, list) or len(es) < 2:
                raise Exception("Combination takes either two Expressions, or a list of Expressions / Strings for initialization.")

            self.e1 = es[0] if isinstance(es[0], Expression) else Var(es[0])
            if len(es) == 2:
                self.e2 = es[1] if isinstance(es[1], Expression) else Var(es[1])
            else:
                self.e2 = Combination(es[1:])
        else:
            if not (isinstance(e1, Expression) and isinstance(e2, Expression)):
                raise Exception("Combination takes either two Expressions, or a list of Expressions / Strings for initialization.")

            self.e1 = e1
            self.e2 = e2   

    def __mul__(self, other):
        if isinstance(other, (int, float, Var)):
            e1 = self.e1 * other
            e2 = self.e2 * other
            return Combination(e1, e2)
        else:
            raise Exception("Multiplication of Combination is only supported for numbers and single term expressions (Var, Interaction).")
            
    def __imul__(self, other):
        if isinstance(other, (int, float, Var)):
            self.e1 *= other
            self.e2 *= other
            return self
        else:
            raise Exception("Multiplication of Combination is only supported for numbers and single term expressions (Var, Interaction).")
        
    def __str__(self):
        return str(self.e1) + " + " + str(self.e2)
        
    def flatten(self):
        return self.e1.flatten() + self.e2.flatten()        

def Poly(var, power):
    if isinstance(var, str):
        base = Var(var)
    elif isinstance(var, Var):
        base = var
    else:
        raise Exception("Poly takes a (str or Var) and a positive int.")
        
    if not (isinstance(power, int) and power > 0):
        raise Exception("Poly takes a (str or Var) and a positive int.")
    
    if power == 1:
        return base
    else:
        return Poly(base, power - 1) + base ** power
        
V = Var
        