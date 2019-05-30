from .model import *
from scipy.stats import f

import numpy as np
import pandas as pd

def anova(model1, model2 = None):
    ''' User-facing function to execute an Analysis of Variance for one or two models. 
    Should only model be given, then a general F-test will be executed on all of the coefficients.
    Should two models be given, then a partial F-test will be executed. Note that one model needs to be a subset of the other for this to properly evaluate.

    Arguments:
        model1 - A Model object that has been fit on some data
        model2 - A Model object that has been fit on some data

    Returns:
        A DataFrame that contains relevant statistics for the test performed
    '''
    if model2 is None:
        return _anova_terms(model1)
    elif is_subset(model1, model2):
        return _anova_models(model1, model2)
    elif is_subset(model2, model1):
        return _anova_models(model2, model1)
    else:
        raise Exception("Parameters must either be one model or two models where one is a subset of the other.")
        
def is_subset(model1, model2):
    ''' Checks if model1 contains all the terms of model2. In other words, checks if model2 is a subset of model1.

    Arguments:
        model1 - A Model object that has been fit on some data.
        model2 - A Model object that has been fit on some data.

    Returns: 
        A boolean value that is True if model2 is a subset of model1, False if model2 is not a subset of model1.       
    '''
    if not model1.given_re.__sim__(model2.given_re):
        # Models should both have the same response variable
        return False
    
    terms1 = set(model1.ex.get_terms())
    terms2 = set(model2.ex.get_terms())
    return terms2.issubset(terms1)

def _calc_stats(numer_ss, numer_df, denom_ms, denom_df):
    ''' Given the appropriate sum of squares for the numerator and the mean sum 
    of squares for the denominator (with respective degrees of freedom) this will 
    return the relevant statistics of an F-test.

    Arguments:
        numer_ss - Sum of squares for the numerator.
        numer_df - Degrees of freedom for the numerator.
        denom_ms - Mean sum of squares for the denominator.
        denom_df - Degrees of freedom for the denominator.

    Returns:
        A tuple of three values.
            Element 0 contains the mean sum of squares for the numerator.
            Element 1 contains the F statistic calculated.
            Element 2 contains the associated p-value for the generated F statistic.
    '''
    numer_ms = numer_ss / numer_df
    f_val = numer_ms / denom_ms
    p_val = 1 - f.cdf(f_val, numer_df, denom_df)
    return (numer_ms, f_val, p_val)

def _process_term(orig_model, term):
    ''' Obtains needed sum of squared residuals of a model fitted without a specified term/coefficient.

    Arguments:
        orig_model - A fitted Model object.
        term - A Variable object to be left out of the original model when fitting.

    Returns:
        A real value indicated the sum of squared residuals.
    '''
    new_model = LinearModel(orig_model.given_ex - term, orig_model.given_re)
    new_model.fit(orig_model.training_data)
    return new_model.get_sse()

def _extract_dfs(model):
    ''' Obtains the different degrees of freedom for a model in reference to an F-test.

    Arguments:
        model - A fitted Model object

    Returns:
        A tuple containing three elements.
            Element 0 contains the degrees of freedom for the explantory variables.
            Element 1 contains the degrees of freedom for the residuals.
            Element 2 contains the total degrees of freedom for the model.
    '''
    reg_df = model.ex.get_dof()
    total_df = len(model.training_x) - 1
    error_df = total_df - reg_df
    return reg_df, error_df, total_df

def _anova_terms(model):
    ''' Perform a global F-test by analyzing all possible models when you leave one coefficient out while fitting.

    Arguments:
        model - A fitted model object.

    Returns:
        A DataFrame object that contains the degrees of freedom, adjusted sum of squares, 
        adjusted mean sum of squares, F values, and p values for the associated tests performed.
    '''
    reg_df, error_df, total_df = _extract_dfs(model)
  
    # Full model values
    r_sse = model.get_sse()
    r_ssr = model.get_ssr()
    r_sst = model.get_sst()
    
    r_mse = r_sse / error_df
    r_msr, r_f_val, r_p_val = _calc_stats(r_ssr, reg_df, r_mse, error_df)
    
    # Calculate the general terms now
    indices = ["Regression"]
    adj_ss = [r_ssr]
    adj_ms = [r_msr]
    f_vals = [r_f_val]
    p_vals = [r_p_val]
    dfs = [reg_df]
    
    terms = model.ex.get_terms()
    for term in terms:
        term_df = term.get_dof()
        term_ss = r_ssr - _process_term(model, term)
        term_ms, term_f, term_p = _calc_stats(term_ss, term_df, r_mse, error_df)
        indices.append(">> " + str(term))
        adj_ss.append(term_ss)
        adj_ms.append(term_ms)
        dfs.append(term_df)
        f_vals.append(term_f)
        p_vals.append(term_p)
    
    
    # Finish off the dataframe's values
    indices += ["Error", "Total"]
    adj_ss += [r_sse, r_sst]
    adj_ms += [r_mse, ""]
    dfs += [error_df, total_df]
    f_vals += ["", ""]
    p_vals += ["", ""]
    
    return pd.DataFrame({
            "DF" : dfs,
            "Adj_SS": adj_ss,
            "Adj_MS" : adj_ms,
            "F_Value" : f_vals,
            "P_Value" : p_vals
        }, index = indices, columns = ["DF", "Adj_SS", "Adj_MS", "F_Value", "P_Value"])

def _anova_models(full_model, reduced_model):
    ''' Performs a partial F-test to compare two models.

    Arguments:
        full_model - A fitted Model object.
        reduced_model - A fitted Model object that is a subset of the full_model.

    Returns:
        A DataFrame object that contains the resdiuals' degrees of freedom, sum of squares of the regression, 
        degrees of freedom for the model, sum of squared residuals, the F value, and the p value for the associated test performed.

    '''
    full_label = str(full_model)
    reduced_label = str(reduced_model)
    
    f_reg_df, f_error_df, f_total_df = _extract_dfs(full_model)
    r_reg_df, r_error_df, r_total_df = _extract_dfs(reduced_model)
    
    f_sse = full_model.get_sse()
    r_sse = reduced_model.get_sse()
    
    denom_df = f_reg_df
    denom_ms = f_sse / denom_df
    numer_df = f_reg_df - r_reg_df
    numer_ss = r_sse - f_sse
    # what next?    
    _, f_val, p_val = _calc_stats(numer_ss, numer_df, denom_ms, denom_df)
    
    indices = ["Reduced Model", "Full Model"]#[reduced_label, full_label]
    resid_df = [r_error_df, f_error_df]
    ssr = [reduced_model.get_ssr(), full_model.get_ssr()] 
    df = ["", resid_df[0] - resid_df[1]]
    sse = ["", numer_ss]
    f = ["", f_val]
    p = ["", p_val]

    return pd.DataFrame({
        "Residual DF" : resid_df,
        "Explained SS" : ssr, 
        "DF" : df,
        "Residual SS" : sse,
        "F_Value" : f,
        "P_Value" : p},
        index = indices, columns = ["Residual DF", "Explained SS", "DF", "Residual SS", "F_Value", "P_Value"])
    
    
    