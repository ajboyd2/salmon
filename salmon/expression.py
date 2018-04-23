import collections
import pandas as pd
import numpy as np
from functools import reduce
from itertools import product
from abc import ABC, abstractmethod
from scipy.special import binom

from . import transformation as _t 

_supported_encodings = ['one-hot']


# abstract
class Expression(ABC):
    
    def __init__(self, scale = 1):
        self.scale = scale
        
    @abstractmethod
    def __str__(self):
        pass
        
    def __eq__(self, other):
        if isinstance(other, Expression):
            return self.scale == other.scale
        return False
    
    def __sim__(self, other):
        if isinstance(other, Expression):
            self_copy = self.copy()
            self_copy.scale = 1
            other_copy = other.copy()
            other_copy.scale = 1
            return self_copy == other_copy
        return False
        
    def __hash__(self):
        return hash(str(self))

    @abstractmethod
    def copy(self):
        # Creates a deep copy of an expression
        pass    
        
    @abstractmethod
    def interpret(self, data):
        # Cast a Var to either a Quantitative or Categorical
        pass

    def transform(self, transformation):
        if isinstance(transformation, str):
            if transformation in _t._default_transformations:
                transformation = _t._default_transformations[transformation](None)
            else:
                raise Exception("Transformation specified is not a default function.")
        elif not isinstance(transformation, _t.Transformation):
            raise Exception("A Transformation or String are needed when transforming a variable.")
            
        self_copy = self.copy()
        return TransVar(self_copy, transformation)
        
    def __add__(self, other):
        if other == 0:
            return self
        elif isinstance(other, (int, float)):
            ret_exp = self.copy()
            shift = Constant(other)
            return Combination((ret_exp, shift))
        elif self.__sim__(other):
            ret_exp = self.copy()
            ret_exp.scale += other.scale
            if ret_exp.scale == 0:
                return 0
            else:
                return ret_exp
        elif isinstance(other, Combination):
            return other.__add__(self)
        elif isinstance(other, Var) or isinstance(other, TransVar):
            return Combination((self, other))
        else:
            raise Exception("Expressions do not support addition with the given arguments.")
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + -1 * other
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            ret_exp = self.copy()
            ret_exp.scale *= other
            return ret_exp
        elif isinstance(other, Constant):
            ret_exp = self.copy()
            ret_exp.scale *= other.scale
            return ret_exp
        elif isinstance(other, Interaction):
            return other.__mul__(self)
        elif isinstance(other, Expression):
            if self.__sim__(other): # Consolidate
                return self.copy() ** 2
            else: # Explicit interaction
                return Interaction((self, other))
        else:
            raise Exception("Expressions do not support multiplication with the given arguments.")
            
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            self_copy = self.copy()
            self_copy.scale *= other
            return self_copy
        elif isinstance(other, Constant):
            self_copy = self.copy()
            self_copy.scale *= other.scale
            return self_copy
        else:
            return self.__mul__(other)
    
    def __pow__(self, other): # Assumes a single term variable, Combinations and Interactions are overloaded
        if other == 1:
            return self
        elif other == 0:
            return 1
        elif isinstance(other, int) and other > 0: # Proper power
            return PowerVar(self, other)
        elif isinstance(other, (int, float)): # Improper, will reduce upon evaluation
            return TransVar(self, _t.Power(other))
        
    def descale(self):
        ret_exp = self.copy()
        ret_exp._descale()
        return ret_exp
        
    def _descale(self): # Overwrite on expression containers such as Interactions and Combinations
        self.scale = 1
        
    @abstractmethod
    def evaluate(self, data, fit = True):
        # Transform the incoming data depending on what type of variable the data is represented by
        pass
    
    def reduce(self):
        return self._reduce({"Q":set(), "C":set(), "V":set(), "Constant": None})
    
    @abstractmethod
    def _reduce(self, ret_dict):
        # Reduce an expression to a dictionary containing lists of unique Quantitative and Categorical Variables
        pass
    
    @abstractmethod
    def get_terms(self):
        # Return a list of top level terms
        pass
    
    @abstractmethod
    def get_dof(self):
        # Retrun the degrees of freedom for an expression (not including constants)
        pass

class Var(Expression):

    def __init__(self, name, scale = 1):
        super().__init__(scale = scale)
        self.name = name
        
    def __eq__(self, other):
        if isinstance(other, Var):
            if self.name == other.name:
                return super().__eq__(other)
        return False
    
    def __hash__(self):
        return hash((self.name, self.scale))
        
    def __str__(self):
        if self.scale != 1:
            return "{1}*{0}".format(self.name, self.scale) 
        else:
            return self.name
        #return reduce(lambda x, y: y.compose(x), [self.name] + self.transformations)
  
    def copy(self):
        return Var(self.name, self.scale)
        
    def interpret(self, data):
        if 'float' in data[self.name].dtype.name or 'int' in data[self.name].dtype.name:
            return Quantitative(self.name, self.transformations)
        else:
            return Categorical(self.name)
        
    def evaluate(self, data, fit = True):
        raise UnsupportedMethodException("Must call interpret prior to evaluating data for variables.")
        
    def _reduce(self, ret_dict):
        ret_dict["V"].add(self)
        return ret_dict
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return 1

class TransVar(Expression):
    
    def __init__(self, var, transformation, scale = 1):
        self.scale = scale
        self.var = var
        self.transformation = transformation
        
    def __str__(self):
        base = self.transformation.compose(str(self.var))
        if self.scale == 1:
            return base
        else:
            return "{}*{}".format(self.scale, base)
        
    def copy(self):
        return TransVar(self.var.copy(), self.transformation.copy(), self.scale)
    
    def __eq__(self, other):
        if isinstance(other, TransVar):
            return self.var == other.var and \
                   self.transformation == other.transformation and \
                   self.scale == other.scale
                    
        return False
    
    def __hash__(self):
        return hash((self.var, self.scale, self.transformation))
    
    def __add__(self, other):
        if isinstance(other, TransVar) and self.var == other.var and self.transformation == other.transformation:
            return TransVar(self.var.copy(), self.transformation, self.scale + other.scale)
        else:
            return Combination((self, other))
        
    def interpret(self, data):
        self.var = self.var.interpret(data)
        return self
    
    def _descale(self):
        self.scale = 1
        self.var._descale()
        
    def evaluate(self, data, fit = True):
        base_data = self.var.evaluate(data, fit)
        base_data = base_data.sum(axis = 1)
        transformed_data = self.scale * self.transformation.transform(values = base_data, training = fit)
        transformed_data.name = str(self)
        return pd.DataFrame(transformed_data)
    
    def _reduce(self, ret_dict):
        return self.var._reduce(ret_dict)
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return 1
    
class PowerVar(TransVar):
    
    def __init__(self, var, power, scale = 1):
        self.scale = scale * (var.scale ** power)
        self.var = var.copy()
        self.var.scale = 1
        self.transformation = _t.Power(power)
        self.power = power
        
    def __eq__(self, other):
        if isinstance(other, PowerVar):
            return self.var == other.var and \
                   self.power == other.power and \
                   self.scale == other.scale

        return False
    
    def copy(self):
        return PowerVar(self.var, self.power, self.scale)
    
    def __hash__(self):
        return hash((self.var, self.scale, self.transformation, self.power))
        
    def __add__(self, other):
        if self.__sim__(other):
            return PowerVar(self.var.copy(), self.power, self.scale + other.scale)
        else:
            return Combination((self, other))
        
    def __mul__(self, other):
        if isinstance(other, PowerVar):
            if self.var.__sim__(other.var):
                base = self.var.copy()
                base.scale = 1
                scalar = self.scale * (self.var.scale ** self.power) * (other.var.scale ** other.power)
                power = self.power + other.power
                return PowerVar(self.var.copy(), power = power, scale = scalar)
        elif isinstance(other, Expression):
            if self.var.__sim__(other):
                base = self.var.copy()
                scalar = self.scale * (self.var.scale ** self.power) * (other.scale)
                power = self.power + 1
                return PowerVar(self.var.copy(), power = power, scale = scalar)
            
        return super().__mul__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if isinstance(other, int):
            return PowerVar(self.var.copy(), self.power * other)
        return super().__pow__(other)
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return 1
    
class Quantitative(Var):
    
    def __init__(self, name, scale = 1):
        super().__init__(name = name, scale = scale)
        self.name = name
    
    def copy(self):
        return Quantitative(self.name, self.scale)
        
    def interpret(self, data):
        return self
    
    def evaluate(self, data, fit = True):
        transformed_data = self.scale * data[self.name]
        transformed_data.name = str(self)
        return pd.DataFrame(transformed_data)
    
    def _reduce(self, ret_dict):
        ret_dict["Q"].add(self)
        return ret_dict
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return 1
        
class Constant(Expression):
    
    def __init__(self, scale = 1):
        self.scale = scale
        
    def __str__(self):
        return str(self.scale)
    
    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.scale == other.scale
        return False
    
    def __sim__(self, other):
        return isinstance(other, Constant)
    
    def __hash__(self):
        return hash((self.scale))
    
    def copy(self):
        return Constant(self.scale)
    
    def interpret(self, data):
        return self
    
    def __pow__(self, other):
        return Constant(self.scale ** other)
    
    def __mul__(self, other):
        if isinstance(other, Expression):
            ret_exp = other.copy()
            ret_exp.scale *= self.scale
            return ret_exp
        elif isinstance(other, (int, float)):
            return Constant(self.scale * other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def evaluate(self, data, fit = True):
        transformed_data = pd.DataFrame(pd.Series(self.scale, data.index, name = str(self)))
        return transformed_data
        
    def _reduce(self, ret_dict):
        ret_dict['Constant'] = self.scale
        return ret_dict
    
    def get_terms(self):
        return list()
    
    def get_dof(self):
        return 0
    
class Categorical(Var):
    
    def __init__(self, name, encoding = 'one-hot', levels = None, baseline = None):
        self.scale = 1
        self.name = name
        if encoding not in _supported_encodings:
            raise Exception("Method " + str(method) + " not supported for Categorical variables.")
        self.encoding = encoding
        self.levels = levels
        self.baseline = baseline
        
    def __str__(self):
        return self.name
        
    def copy(self):
        return Categorical(self.name, self.encoding, None if self.levels is None else self.levels[:], self.baseline)
                
    def interpret(self, data):
        return self
    
    def transform(self, transformation):
        raise Exception("Categorical variables cannot be transformed.")
        
    def set_baseline(self, value):
        if isinstance(value, collections.Iterable):
            self.baseline = value
        else:
            self.baseline = [value]
        
    def _set_levels(self, data, override_baseline = True):
        unique_values = data[self.name].unique()
        unique_values.sort()
        self.levels = unique_values[:]
        if override_baseline:
            self.set_baseline(unique_values[0])
        
    def _one_hot_encode(self, data):
        return pd.DataFrame({self.name + "{" + str(level) + "}" : (data[self.name] == level) * 1 for level in self.levels if level not in self.baseline})
        
    def evaluate(self, data, fit = True):
        if self.levels is None:
            self._set_levels(data)
        
        if self.encoding == 'one-hot':
            return self._one_hot_encode(data)
        else:
            raise NotImplementedException()
        
    def _reduce(self, ret_dict):
        ret_dict["C"].add(self)
        return ret_dict
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return len(self.levels) - len(self.baseline)
        
class Interaction(Expression):
    def __init__(self, terms, scale = 1):
        self.scale = scale

        if any(not isinstance(t, Expression) for t in terms):
            raise Exception("Interaction takes only Expressions for initialization.")
        
        self.terms = set()
        for term in terms:
            self._add_term(term)
        
    def __eq__(self, other):
        if isinstance(other, Interaction):
            if len(self.terms.difference(other.terms)) == 0 and len(other.terms.difference(slef.terms)) == 0:
                return super().__eq__(other)
        return False
    
    def __hash__(self):
        return hash((frozenset(self.terms), self.scale))
        
    def __str__(self):
        base = "(" + ")(".join(sorted(str(term) for term in self.terms)) + ")" 
        if self.scale == 1:
            return base
        else:
            return "{}*{}".format(self.scale, base)
        
    def _add_term(self, other_term):
        if isinstance(other_term, PowerVar):
            base_term = other_term.var
        else:
            base_term = other_term
        similar_term = None
        for term in self.terms:
            if term.__sim__(base_term) or (isinstance(term, PowerVar) and term.var.__sim__(base_term)):
                similar_term = term
                break

        if similar_term is not None:
            self.terms.remove(similar_term)
            self.terms.add(similar_term * other_term)
        else:
            self.terms.add(other_term)               
   
    def _add_terms(self, other_terms):
        for term in other_terms:
            self._add_term(term)


    def copy(self):
        return Interaction({term.copy() for term in self.terms}, self.scale)
        
    def interpret(self, data):
        self.terms = [term.interpret(data) for term in self.terms]
        return self
    
    def __mul__(self, other):
        if isinstance(other, Interaction):
            self_copy = self.copy()
            self_copy._add_terms(other.terms)
            return self_copy
        elif isinstance(other, Constant):
            self_copy = self.copy()
            self_copy.scale *= other.scale
            return self_copy
        elif isinstance(other, Expression):
            self_copy = self.copy()
            self_copy._add_term(other)
            return self_copy
        else:
            return super().__mul__(other)
        
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            ret_int = Interaction(())
            for term in self.terms:
                ret_int._add_term(term ** other)
            return ret_int
        return super().__pow__(other)
    
    def _descale(self):
        self.scale = 1
        for term in self.terms:
            term._descale()
            
    def evaluate(self, data, fit = True):
        transformed_data_sets = [var.evaluate(data, fit) for var in self.terms]
        # rename columns in sets
        for data_set in transformed_data_sets:
            data_set.columns = ["({})".format(col) for col in data_set.columns]
            
        base_set = transformed_data_sets[0]
        for data_set in transformed_data_sets[1:]:
            new_set = pd.DataFrame()
            for base_column in base_set:
                for new_column in data_set:
                    new_set[base_column + new_column] = base_set[base_column] * data_set[new_column]
            
            base_set = new_set
            
        return base_set
    
    def _reduce(self, ret_dict):
        for term in self.terms:
            ret_dict = term._reduce(ret_dict)
            
        return ret_dict
    
    def get_terms(self):
        return [self]
    
    def get_dof(self):
        return reduce(lambda x,y: x*y, (term.get_dof() for term in self.terms))
    
class Combination(Expression):

    def __init__(self, terms, scale = 1):
        self.scale = scale
        
        if any(not isinstance(t, Expression) for t in terms):
            raise Exception("Combination takes only Expressions for initialization.")
                        
        self.terms = set()
        for term in terms:
            self._add_term(term)
                    
    def __eq__(self, other):
        if isinstance(other, Combination):
            if len(self.terms.difference(other.terms)) == 0 and len(other.terms.difference(slef.terms)) == 0:
                return super().__eq__(other)
        return False

    def __hash__(self):
        return hash((frozenset(self.terms), self.scale))
                
    def __str__(self):
        base = "+".join(sorted(str(term) for term in self.terms)) 
        if self.scale == 1:
            return base
        else:
            return "{}*({})".format(self.scale, base)
        
    def _add_term(self, other_term):
        if isinstance(other_term, (int, float)):
            other_term = Constant(other_term)
        
        similar_term = None
        for term in self.terms:
            if term.__sim__(other_term):
                similar_term = term
                break
                
        if similar_term is not None:
            self.terms.remove(similar_term)
            self.terms.add(similar_term + other_term)
        else:
            self.terms.add(other_term)       
            
    def _add_terms(self, other_terms):
        for term in other_terms:
            self._add_term(term)
        
    def copy(self):
        return Combination({term.copy() for term in self.terms}, self.scale)
        
    def interpret(self, data):
        self.terms = [term.interpret(data) for term in self.terms]
        return self
    
    def __add__(self, other):
        if isinstance(other, Combination):
            ret_comb = self.copy()
            ret_comb._add_terms(other.terms)
            return ret_comb
        elif isinstance(other, Expression):
            ret_comb = self.copy()
            ret_comb._add_term(other)
            return ret_comb
        else:
            return super().__add__(other)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, Combination):
            ret_comb = Combination(())
            for other_term in other.terms:
                ret_comb._add_terms((self * other_term).terms)
            return ret_comb
        elif isinstance(other, Expression):
            ret_comb = Combination(())
            for term in self.terms:
                ret_comb._add_term(term * other)
            return ret_comb
        else:
            return super().__mul__(other)
            
    def __pow__(self, other):
        if isinstance(other, int) and other > 0:
            terms = [self.scale * term for term in self.terms]
            return MultinomialExpansion(terms, other)
        else:
            return super().__pow__(other)
        
    def _descale(self):
        self.scale = 1
        for term in self.terms:
            term._descale()
            
    def evaluate(self, data, fit = True):
        return pd.concat([term.evaluate(data, fit) for term in self.terms], axis = 1)
    
    def _reduce(self, ret_dict):
        for term in self.terms:
            ret_dict = term._reduce(ret_dict)
            
        return ret_dict
    
    def get_terms(self):
        return self.terms
    
    def get_dof(self):
        return reduce(lambda x,y: x+y, (term.get_dof() for term in self.terms))
    
def MultinomialCoef(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * MultinomialCoef(params[:-1]) 

def MultinomialExpansion(terms, power):
    combination_terms = []
    generators = [range(power+1) for _ in range(len(terms))]
    for powers in product(*generators): # Inefficient, optimize later
        if sum(powers) == power:
            interaction_terms = [int(MultinomialCoef(powers))]
            for term, term_power in zip(terms, powers):
                if term_power == 0:
                    continue
                interaction_terms.append(term ** term_power)
            combination_terms.append(reduce(lambda x,y: x * y, interaction_terms))
    return reduce(lambda x,y: x + y, combination_terms)
    
def Poly(var, power):
    if isinstance(var, str):
        var = Q(var)
    
    if not isinstance(power, int) or power < 0:
        raise Exception("Must raise to a non-negative integer power.")
    elif power == 0:
        return Constant(1)
    else:
        terms = [var ** i for i in range(1, power+1)]
        return reduce(lambda x,y: x + y, terms)
           
        
# Transformations 
Log = lambda var: var.transform("log")
Log10 = lambda var: var.transform("log10")
Sin = lambda var: var.transform("sin")
Cos = lambda var: var.transform("cos")
Exp = lambda var: var.transform("exp")
Standardize = lambda var: var.transform("standardize")
Z = lambda var: var.transform("standardize")
Cen = lambda var: var.transform("center")
Center = lambda var: var.transform("center")
Identity = lambda var: var.transform("identity")

# Aliases
V = Var
Q = Quantitative
Quant = Quantitative
C = Categorical
Cat = Categorical
Nominal = Categorical
Nom = Categorical
N = Categorical