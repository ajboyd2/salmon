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
        
    def copy(self):
        # Creates a deep copy of an expression
        raise NotImplementedError()    
        
    def interpret(self, data):
        # Cast a Var to either a Quantitative or Categorical
        raise NotImplementedError()
        
class Var(Expression):

    def __init__(self, name):
        self.name = name
  
    def __mul__(self, other):
        if isinstance(other, Var):
            return Interaction(self, other) 
        elif isinstance(other, Combination):
            return other.__mul__(self)
        else:
            raise Exception("Multiplication of expressions must involve a (Quantitative * number) or a (Expression * Expression).")
    
    def __imul__(self, other):
        if isinstance(other, Var):
            return Interaction(self, other) 
        elif isinstance(other, Combination):
            return other.__mul__(self)
        else:
            raise Exception("Inplace multiplication of expressions must involve a (term *= number) or a (term *= term).")
 
    def __str__(self):
        return self.name

    def flatten(self, separate_interactions = False):
        return [self]

    def copy(self):
        return Var(self.name)
        
    def interpret(self, data):
        if 'float' in data[self.name].dtype.name or 'int' in data[self.name].dtype.name:
            return Quantitative(self.name)
        else:
            return Categorical(self.name)
        
class Quantitative(Var):
    
    def __init__(self, name, transformation = 1, coefficient = 1, shift = 0):
        self.name = name
        self.transformation = transformation
        self.coefficient = coefficient
        self.shift = shift
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Quantitative(self.name, self.transformation, self.coefficient, self.shift + other)
        else:
            return super().__add__(other)
            
    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self.shift += other
            return self
        else:
            return super().__add__(other)
    
    def __mul__(self, other):
        if isinstance(self, Interaction) or isinstance(other, Interaction):
            return super().__mul__(other)
        elif isinstance(other, Quantitative) and self.name == other.name and self.shift == other.shift and isinstance(self.transformation, (int, float)) and isinstance(other.transformation, (int, float)): # Combine variables of same base
            return Quantitative(self.name, self.transformation + other.transformation, self.coefficient * other.coefficient, self.shift)
        elif isinstance(other, (int, float)):
            return Quantitative(self.name, self.transformation, self.coefficient * other, self.shift)
        else:
            return super().__mul__(other)
            
    def __imul__(self, other):
        if isinstance(self, Interaction) or isinstance(other, Interaction):
            return super().__imul__(other)
        elif isinstance(other, Quantitative) and self.name == other.name and self.shift == other.shift and isinstance(self.transformation, (int, float)) and isinstance(other.transformation, (int, float)): # Combine variables of same base
            self.transformation += other.transformation
            self.coefficient *= other.coefficient
            return self
        elif isinstance(other, (int, float)):
            self.coefficient *= other
            return self 
        else:
            return super().__imul__(other)
            
    def __pow__(self, other):
        if isinstance(other, (int, float)) and isinstance(self.transformation, (int, float)):
            return Quantitative(self.name, self.transformation * other, self.coefficient, self.shift)
        else:
            raise Exception("Transforming a term must involve a number like (term ** number) and not be already non-linearly transformed.")
            
    def __ipow__(self, other):
        if isinstance(other, (int, float)) and isinstance(self.transformation, (int, float)):
            self.transformation += other
            return self
        else:
            raise Exception("Transforming a term must involve a number like (term ** number) and not be already non-linearly transformed.")
    
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
        
        if isinstance(self.transformation, (int, float)):
            if self.transformation != 1:
                if shift_flag:
                    base = base + "^%s"
                else:
                    base = "(" + base + ")^%s"
                params.append(str(self.transformation))
        else:
            base = "%s(" + base + ")"
            params.insert(0, self.transformation)
            
        if self.coefficient != 1:
            base = "%s*" + base
            params.insert(0, str(self.coefficient))

        return base % tuple(params)
    
    def transform(self, func_name):
        if self.transformation != 1:
            raise Exception("Function transformations only valid on untransformed variables.")
        if isinstance(func_name, str):
            self.transformation = func_name
        else:
            raise Exception("Only function names (strings) are accepted for transformations.")

    def copy(self):
        return Quantitative(self.name, self.transformation, self.coefficient, self.shift)
        
    def interpret(self, data):
        return self
            
class Categorical(Var):
    
    def __init__(self, name, method = 'one-hot', levels = None):
        self.name = name
        supported_methods = ['one-hot']
        if method not in supported_methods:
            raise Exception("Method " + str(method) + " not supported for Categorical variables.")
        self.method = method
        self.levels = levels
    
    def __str__(self):
        return self.name
        
    def copy(self):
        return Categorical(self.name, self.method, self.levels)
                
    def interpret(self, data):
        return self

        
class Interaction(Quantitative):

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
 
    def __pow__(self, other):
        if isinstance(other, (int, float)) and isinstance(self.transformation, (int, float)):
            return Interaction(self.e1, self.e2, self.transformation * other, self.coefficient, self.shift)
        else:
            raise Exception("Transforming a term must involve a number like (term ** number) and not be already non-linearly transformed.") 
 
    def flatten(self, separate_interactions = False):
        if separate_interactions:
            return self.e1.flatten(separate_interactions) + self.e2.flatten(separate_interactions)
        else:
            return [self]

    def copy(self):
        return Interaction(self.e1.copy(), self.e2.copy(), self.transformation, self.coefficient, self.shift)
        
    def interpret(self, data):
        self.e1 = self.e1.interpret(data)
        self.e2 = self.e2.interpret(data)
        return self
    
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
        if isinstance(other, (int, float, Var, Combination)):
            e1 = self.e1 * other
            e2 = self.e2 * other
            return Combination(e1, e2)
        else:
            raise Exception("Multiplication of Combination is only supported for numbers, single term expressions, and other Combinations.")
            
    def __imul__(self, other):
        if isinstance(other, (int, float, Var, Combination)):
            self.e1 *= other
            self.e2 *= other
            return self
        else:
            raise Exception("Multiplication of Combination is only supported for numbers and single term expressions (Var, Interaction).")
    
    def __pow__(self, other):
        if isinstance(other, int):
            n = other
            if n > 1:
                if n % 2 == 0:
                    return (self * self) ** (n // 2)
                else:
                    return ((self * self) ** ((n - 1) // 2)) * self
            elif n == 1:
                return self
        else:
            raise Exception("Exponentiation of Combinations only supported with powers of positive integers.")
    
    def __str__(self):
        return str(self.e1) + " + " + str(self.e2)
        
    def flatten(self, separate_interactions = False):
        return self.e1.flatten(separate_interactions) + self.e2.flatten(separate_interactions)        

    def copy(self):
        return Combination(self.e1.copy(), self.e2.copy())
        
    def interpret(self, data):
        self.e1 = self.e1.interpret(data)
        self.e2 = self.e2.interpret(data)
        return self
        
def Poly(var, power):
    if isinstance(var, str):
        base = Quantitative(var)
    elif isinstance(var, Quantitative):
        base = var
    else:
        raise Exception("Poly takes a (str or Quantitative) and a positive int.")
        
    if not (isinstance(power, int) and power > 0):
        raise Exception("Poly takes a (str or Quantitative) and a positive int.")
    
    if power == 1:
        return base
    else:
        return Poly(base, power - 1) + base ** power
        
# Transformations 
def Log(var):
    if isinstance(var, Quantitative):
        var.transform("log")
        return var
    else:
        raise Exception("Can only take the log of quantitative variables.")
        
def Log10(var):
    if isinstance(var, Quantitative):
        var.transform("log10")
        return var
    else:
        raise Exception("Can only take the log of quantitative variables.")
        
def Sin(var):
    if isinstance(var, Quantitative):
        var.transform("sin")
        return var
    else:
        raise Exception("Can only take the sin of quantitative variables.")        
        
# Aliases
V = Var
Q = Quantitative
Quant = Quantitative
C = Categorical
Cat = Categorical
Nominal = Categorical
Nom = Categorical
N = Categorical