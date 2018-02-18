from .model import *

def anova(model1, model2 = None):
    if model2 is None:
        return _anova_terms(model1)
    elif is_subset(model1, model2):
        return _anova_models(model1, model2)
    elif is_subset(model2, model1):
        return _anova_models(model2, model1)
    else:
        raise Exception("Parameters must either be one model or two models where one is a subset of the other.")
        
# checks if model1 contains all the terms of model2
def is_subset(model1, model2):
    if not model1.given_re.__sim__(model2.given_re):
        # Models should both have the same response variable
        return False
    
    terms1 = set(model1.ex.get_terms())
    terms2 = set(model2.ex.get_terms())
    return terms2.issubset(terms1)

def _anova_terms(model):
    pass

def _anova_models(model1, model2):