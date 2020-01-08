import collections
import pandas as pd
import numpy as np
from functools import reduce
from itertools import product
from abc import ABC, abstractmethod
from scipy.special import binom

from . import transformation as _t 

_supported_encodings = ['one-hot']


# ABC is a parent object that allows for Abstract methods
class Expression(ABC):
    ''' The parent abstract class that all subsequent representations of coefficients stem from and Model objects utilize. 
    '''
    
    def __init__(self, scale = 1):
        ''' Create an expression. Cannot be done directly, only through inheritance.

        Arguments:
            scale - A real value, default is 1. This represents a constant scaling that is applied to the Expression.

        Returns:
            An Expression object.
        '''

        self.scale = scale
        
    @abstractmethod
    def __str__(self):
        ''' Represent an Expression object as a String utilizing standard algebraic notation.
        '''
        pass
        
    def __eq__(self, other):
        ''' Check if an Expression is equivalent to another object.

        Arguments:
            other - An object to compare self to.

        Returns:
            A boolean that is True if they are equivalent, False if otherwise.
        '''

        if isinstance(other, Expression):
            return self.scale == other.scale
        return False
    
    def __sim__(self, other):
        ''' Check if an Expression is similar to another object. This is done for the purposes of algebraic simplification.

        Arguments:
            other - An object to compare self to.

        Returns:
            A boolean that is True if they are similar, False if otherwise.
        '''

        if isinstance(other, Expression):
            self_copy = self.copy()
            self_copy.scale = 1
            other_copy = other.copy()
            other_copy.scale = 1
            return self_copy == other_copy
        return False
        
    def __hash__(self):
        ''' Hash an Expression for the purposes of storing Expressions in sets and dictionaries.

        Returns:
            A real value that represents the hash of the object.
        '''
        return hash(str(self))

    @abstractmethod
    def copy(self):
        '''Creates a deep copy of an expression'''
        pass    
        
    @abstractmethod
    def interpret(self, data):
        ''' Cast a Var object or any nested Var objects in a functional manner to 
        be either a Quantitative or Categorical variable.

        Arguments:
            data - A DataFrame object that contains a column == self.name

        Returns:
            A similar object that is either Quantitative or Categorical object.
        '''        
        pass

    def transform(self, transformation):
        '''Apply a transformation to an Expression object.

        Arguments:
            transformation - Either a transformation object of a str representation of common Transformations.

        Returns:
            A transformed Expression in the form of a TransVar object.
        '''
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
        ''' Form a combination of two Expressions by the addition operator (+).

        Arguments:
            other - An Expresssion object.

        Returns:
            Another Expression object that models the two operands added together. 
            In most cases this is with a Combination object. Other cases it may be a simplified 
            object of the same types as the original operands (i.e. X + X = 2*X).
        '''
        if other == 0:
            return self
        elif isinstance(other, (int, float)):
            ret_exp = self.copy()
            shift = Constant(other)
            return ret_exp + shift
        elif self.__sim__(other):
            ret_exp = self.copy()
            ret_exp.scale += other.scale
            if ret_exp.scale == 0:
                return 0
            else:
                return ret_exp
        elif isinstance(other, Combination):
            return other.__add__(self)
        elif isinstance(other, (Var, TransVar, Constant, Interaction)):  # single term expressions
            return Combination((self, other))
        else:
            raise Exception("Expressions do not support addition with the given arguments.")
        
    def __radd__(self, other):
        ''' See __add__. '''
        return self.__add__(other)
    
    def __sub__(self, other):
        ''' Subtract one Expression object from another. This is a special case of addition. See __add__. '''
        return self + -1 * other
    
    def __mul__(self, other):
        ''' Multiply two Expression objects together. When possible, simplification will be done. 
        This is called with the multiplication operator (*).

        Arguments:
            other - An Expression object

        Returns:
            An Expression modeling the two operands multiplied together. If the inputs were single terms, 
            then the output will be an Interaction object. If at least one was a combination of terms, 
            then the output will most likely be a Combination object consisting of interactions as distribution will occur.
        '''
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
        ''' See __mul__.'''
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

    def __truediv__(self, other):
        ''' Divide one expression by another. This is a special case of __mul__ and __pow__. '''
        return self * (other ** -1)

    def __rtruediv__(self, other):
        ''' See __truediv__. '''
        return other * (self ** -1)
    
    def __pow__(self, other): 
        ''' Raise an Expression object to a constant power. 
        
        Arguments:
            other - A real value.
        
        Returns:
            An Expression object that represents the original object raised to a constant power.
            More than likely this will either be a PowerVar object if the original object was a single term,
            or a Combination / Interaction if the original was one of those.
        '''
        # This particualr function assumes a single term variable as this operation for Combinations and Interactions are overloaded
        if other == 1:
            return self
        elif other == 0:
            return 1
        elif isinstance(other, (int, float)):# and other > 0: # Proper power
            return PowerVar(self, other)
        else:
            raise Exception("Can only raise to a constant power.")
        #elif isinstance(other, (float)): # Improper, will reduce upon evaluation
        #    return TransVar(self, _t.Power(other))

    def __and__(self, other):
        ''' Returns an interaction with main effects between two Expression objects. '''
        return self + other + (self * other)

    def __xor__(self, other):
        ''' Returns an interaction with main effects up to a specified power.
        
        If the base object is not a Combination, then the default __pow__ logic is used.'''

        return self ** other  # see Combination's __xor__ for other logic
        
    def descale(self):
        ''' Remove scaling on an Expression. Done in a functional manner so original object is unaffected. '''
        ret_exp = self.copy()
        ret_exp._descale()
        return ret_exp
        
    def _descale(self): 
        ''' Helper function for descaling Expressions. '''
        # Overwrite on expression containers such as Interactions and Combinations
        self.scale = 1
        
    @abstractmethod
    def evaluate(self, data, fit = True):
        ''' Given data, apply the appropriate transformations, combinations, and interactions.

        Arguments:
            data - A DataFrame whose column names match the names of the base Variable objects.
            fit - A flag to reference when evaluating the data to know when to overwrite Categorical levels.
        
        Returns:
            An appropriate DataFrame consisting of specified columns for each term in the Expression.
        '''
        # Transform the incoming data depending on what type of variable the data is represented by
        pass
    
    def reduce(self):
        ''' Obtain the base Quantitative, Categorical, Constant, and Varable terms. 
        
        Returns:
            A dictionary mapping the respective types of terms to a set collection of the unique terms present.
        '''
        return self._reduce({"Q":set(), "C":set(), "V":set(), "Constant": None})
    
    @abstractmethod
    def _reduce(self, ret_dict):
        ''' A helper method for the reduce function to allow for recursion. '''
        # Reduce an expression to a dictionary containing lists of unique Quantitative and Categorical Variables
        pass
    
    @abstractmethod
    def get_terms(self):
        ''' Return a list of the top level terms. '''
        pass
    
    @abstractmethod
    def get_dof(self):
        ''' Retrun the degrees of freedom for an expression (not including constants) '''
        pass

    def untransform(self, data):
        ''' Untransform data by inverting the applied operations to it for the purposes of plotting. 
        
        Returns:
            A DataFrame object with appropriately inverted columns. 
        '''
        return data * (1 / self.scale)

    def untransform_name(self):
        ''' Get the untransformed name of the model after inversions have been applied. '''
        return str(self * (1 / self.scale))
    
    @abstractmethod
    def contains(self, other):
        ''' Returns true if this object is contains the other. '''
        pass

class Var(Expression):
    ''' A base object that represents a generic single term.
    If a model consists of Y ~ X1 + X2 + X3, X2 would be a single term, as would X1 and X3. 
    
    This is more general when compared to a Quantitative or Categorical object, as it does not impose 
    any restrictions on the data it represents. Upon model fitting, a Var will check to see if the 
    data it represents is better suited for being a Quantitative variable or a Categorical one. 
    '''

    def __init__(self, name, scale = 1):
        ''' Creates a Var object.

        Arguments:
            name - A str that will later be used to access a column in specific DataFrames. 
                   This name should coincide with the data it is representing.
            scale - A real value that will multiplicatively scale the term.
        '''
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
            return Quantitative(self.name, self.scale)
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
    
    def contains(self, other):
        return False  # impossible for a simple variable to contain another

class TransVar(Expression):
    ''' Represents an Expression that has a Transformation object applied to it.

    In the general case, Transformations do not support distributing across terms. As such,
    regardless of what the original Expression pre-transformation was, the new TransVar object
    will be treated as a single term.
    '''
    
    def __init__(self, var, transformation, scale = 1):
        ''' Creates a TransVar object.

        Arguments:
            var - An Expression object to be transformed.
            transformation - A Transformation object to be applied to var.
            scale - A real value that will multiplicatively scale the term.
        '''
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

    def untransform(self, data):
        unscaled = data  * (1 / self.scale)
        inverted = self.transformation.invert(unscaled)
        return self.var.untransform(inverted)

    def untransform_name(self):
        return self.var.untransform_name()

    def contains(self, other):
        if isinstance(other, Combination):
            return any(self.contains(other_term) for other_term in other.terms)

        return self.var.__eq__(other) or self.var.contains(other)
    
class PowerVar(TransVar):
    ''' A special case of the TransVar as a Power transformation can easily be distributed across terms.
    As such, special consideration is taken into account when multiplying two PowerVar objects.
    '''
    def __init__(self, var, power, scale = 1):
        ''' Creates a PowerVar object.

        Arguments:
            var - An Expression object to be raised to a power.
            power - A real value to raise var to.
            scale - A real value that will multiplicatively scale the term.
        '''
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
                if power == 0:
                    return Constant(scalar)
                else:
                    return PowerVar(self.var.copy(), power = power, scale = scalar)
        elif isinstance(other, Expression):
            if self.var.__sim__(other):
                base = self.var.copy()
                scalar = self.scale * (self.var.scale ** self.power) * (other.scale)
                power = self.power + 1
                if power == 0:
                    return Constant(scalar)
                else:
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

    def contains(self, other):
        if isinstance(other, Combination):
            return any(self.contains(other_term) for other_term in other.terms)

        if isinstance(other, PowerVar):
            if self.var.__eq__(other.var):
                return self.power > other.power
            else:
                return self.var.contains(other)
        else:
            return self.var.__eq__(other) or self.var.contains(other)
    
class Quantitative(Var):
    ''' A base term that inherits from Var. This specifically models Quantitative terms. '''
    
    def __init__(self, name, scale = 1):
        ''' Creates a Quantitative object.

        Arguments:
            name - A str that will later be used to access a column in specific DataFrames. 
                   This name should coincide with the data it is representing.
            scale - A real value that will multiplicatively scale the term.
        '''
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
    ''' Represents a standalone term for constant values. '''
    
    def __init__(self, scale = 1):
        ''' Create a Constant object.

        Arguments:
            scale - A real value that IS the constant value.
        '''
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
        return 1

    def contains(self, other):
        return False
    
class Categorical(Var):
    ''' The other base term that stems from the Var class. Represents solely categorical data. '''

    def __init__(self, name, encoding = 'one-hot', levels = None, baseline = None):
        ''' Creates a Categorical object.

        Arguments:
            name - A str that will later be used to access a column in specific DataFrames. 
                   This name should coincide with the data it is representing.
            encoding - A str that represnts the supported encoding scheme to use. Default is one-hot.
            levels - A list object that holds all values to be considered as different levels during encoding. 
                Any left out will be treated similarly as the baseline. A value of None will have levels learned upon fitting. 
            baseline - A list of objects to be collectively treated as a baseline.
        '''
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
    
    #def transform(self, transformation):
    #    raise Exception("Categorical variables cannot be transformed.")
        
    def set_baseline(self, value):
        if isinstance(value, collections.Iterable) and not isinstance(value, str):
            self.baseline = value
        else:
            self.baseline = [value]
        
    def _set_levels(self, data, override_baseline = True):
        unique_values = data[self.name].unique()
        unique_values.sort()
        if self.levels is None:
            self.levels = unique_values[:]
            if self.baseline is None:
                self.set_baseline(unique_values[0])
        else:
            unique_values = set(unique_values)
            diff = unique_values - set(self.levels)
            if len(diff) == 0:
                self.set_baseline(self.levels[0])
            else:
                self.set_baseline(diff)
                self.levels = list(self.levels)
                for element in diff:
                    self.levels.append(element)
        
        
    def _one_hot_encode(self, data):
        return pd.DataFrame({self.name + "{" + str(level) + "}" : (data[self.name] == level) * 1 for level in self.levels if level not in self.baseline})
        
    def evaluate(self, data, fit = True):
        if self.levels is None or self.baseline is None:
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
    ''' An Interaction models two or more terms multiplied together. An Interaction is treated as a single term.
    If multiple terms are the result of a multiplication, then the result will be a Combination of Interactions.
    '''

    def __init__(self, terms, scale = 1):
        ''' Create an Interaction object.

        Arguments:
            terms - A collection of terms to be modeled as multiplied together.
            scale - A real value that will multiplicatively scale the term.
        '''
        self.scale = scale

        if any(not isinstance(t, Expression) for t in terms):
            raise Exception("Interaction takes only Expressions for initialization.")
        
        self.terms = set()
        for term in terms:
            self._add_term(term)
        
    def __eq__(self, other):
        if isinstance(other, Interaction):
            if len(self.terms.difference(other.terms)) == 0 and len(other.terms.difference(self.terms)) == 0:
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
        self.terms = set(term.interpret(data) for term in self.terms)
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

    def contains(self, other):
        if isinstance(other, Combination):
            return any(self.contains(other_term) for other_term in other.terms)

        self_terms = self.terms
        if isinstance(other, Interaction):
            other_terms = other.terms
        else:
            other_terms = [other]
        all_terms_found = True
        for o_t in  other_terms:
            single_term_found = False
            for s_t in self_terms:
                single_term_found = single_term_found or s_t.__eq__(o_t) or s_t.contains(o_t)
            all_terms_found = all_terms_found and single_term_found
        return all_terms_found
    
class Combination(Expression):
    ''' A Combination models several single terms added together. '''

    def __init__(self, terms, scale = 1):
        ''' Create a Combination object.

        Arguments:
            terms - A collection of Expressions to add together.
            scale - A real value that will multiplicatively scale the term.
        '''
        self.scale = scale
        
        terms = [Constant(term) if isinstance(term, (int, float)) else term for term in terms]

        if any(not isinstance(t, Expression) for t in terms):
            raise Exception("Combination takes only Expressions for initialization.")
                        
        self.terms = set()
        for term in terms:
            self._add_term(term)
                    
    def __eq__(self, other):
        if isinstance(other, Combination):
            if len(self.terms.difference(other.terms)) == 0 and len(other.terms.difference(self.terms)) == 0:
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

    def __xor__(self, other):
        if isinstance(other, int) and other >= 0:
            power = min(other, len(self.terms))
            new_terms = set()
            base_terms = self.terms
            last_added_terms = [frozenset()]

            for _ in range(power):
                newly_added_terms = set()
                for new_term in base_terms:
                    for last_added_term in last_added_terms:
                        if new_term not in last_added_term:
                            newly_added_terms.add(last_added_term.union(frozenset([new_term])))

                new_terms = new_terms.union(newly_added_terms)
                last_added_terms = newly_added_terms

            processed_terms = set()
            for term in new_terms:
                if len(term) == 1:
                    processed_terms.add(next(iter(term)))
                else:
                    processed_terms.add(reduce(lambda x,y: x * y, term))

            if len(new_terms) == 0:
                return Constant(1)
            else:
                return Combination(processed_terms)
        else:
            return self ** other
        
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
            addition_result = similar_term + other_term
            if addition_result != 0:
                self.terms.add(addition_result)
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

    def contains(self, other):
        self_terms = self.terms
        if isinstance(other, Combination):
            other_terms = other.terms
        else:
            other_terms = [other]
        for s_t in self_terms:
            for o_t in  other_terms:
                if s_t.__eq__(o_t) or s_t.contains(o_t):
                    return True
        return False
    
def MultinomialCoef(params):
    ''' Calculate the coefficients necessary when raising polynomials to a power.

    Arguments:
        params - A list of powers of individual terms.

    Returns:
        An integer that is the coefficient for that term.
    '''
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * MultinomialCoef(params[:-1]) 

def MultinomialExpansion(terms, power):
    ''' Raise a collection of single terms (polynomial) to a power.

    Arguments:
        terms - A collection of Expression objects representing a polynomial.
        power - An integer to raise the polynomial to.

    Returns:
        A expanded / distributed Combination representing the polynomial raised to the specified power.
    '''
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
    ''' A quick way to create a standard polynomial from one base expression.

    Arguments:
        var - An Expression object. 
        power - An integer to raise var to the power of.

    Returns:
        A Combination object. If var was a single term, then the standard polynomial is returned. 
        If var was a Combination, then a Combination of all Interactions up to order 'power' is 
        returned due to distribution rules.
    '''
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