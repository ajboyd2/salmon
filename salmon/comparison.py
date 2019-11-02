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

def _calc_stats(numer_ss, numer_df, denom_ss, denom_df):
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
    denom_ms = denom_ss / denom_df
    f_val = numer_ms / denom_ms
    p_val = f.sf(f_val, numer_df, denom_df)
    return f_val, p_val

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
    return new_model.get_sse(), new_model.get_ssr()

def _extract_dfs(model, dict_out=False):
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

    if dict_out:
        return dict(
            model_df=reg_df,
            total_df=total_df,
            error_df=error_df
        )
    else:
        return reg_df, error_df, total_df

def _anova_terms(model):
    ''' Perform a global F-test by analyzing all possible models when you leave one coefficient out while fitting.

    Arguments:
        model - A fitted model object.

    Returns:
        A DataFrame object that contains the degrees of freedom, adjusted sum of squares, 
        adjusted mean sum of squares, F values, and p values for the associated tests performed.
    '''
    full_reg_df, full_error_df, total_df = _extract_dfs(model)
  
    # Full model values
    full_sse = model.get_sse()  # sum of squared errors
    full_ssr = model.get_ssr()  # sum of squares explained by model
    full_sst = model.get_sst()  
    
    global_f_val, global_p_val = _calc_stats(full_ssr, full_reg_df, full_sse, full_error_df)
    
    # Calculate the general terms now
    indices = ["Global Test"]
    sses = [full_ssr]
    ssrs = [full_ssr]
    f_vals = [global_f_val]
    p_vals = [global_p_val]
    dfs = [full_reg_df]
    
    terms = model.ex.get_terms()
    for term in terms:
        term_df = term.get_dof()
        reduced_sse, reduced_ssr = _process_term(model, term)
        reduced_f_val, reduced_p_val = _calc_stats(full_ssr - reduced_ssr, term_df, full_sse, full_error_df)
        indices.append("- " + str(term))
        sses.append(reduced_sse)
        ssrs.append(reduced_ssr)
        dfs.append(term_df)
        f_vals.append(reduced_f_val)
        p_vals.append(reduced_p_val)
        
    # Finish off the dataframe's values
    indices.append("Error")
    sses.append("")
    ssrs.append("")
    dfs.append(full_error_df)
    f_vals.append("")
    p_vals.append("")
    
    return pd.DataFrame({
            "DF" : dfs,
            "SS Err.": sses,
            "SS Reg." : ssrs,
            "F" : f_vals,
            "p" : p_vals
        }, index = indices, columns = ["DF", "SS Err.", "SS Reg.", "F", "p"])

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
    
    f_sse, f_ssr = full_model.get_sse(), full_model.get_ssr()
    r_sse, r_ssr = reduced_model.get_sse(), reduced_model.get_ssr()
  
    f_val, p_val = _calc_stats(r_sse - f_sse, r_error_df - f_error_df, f_sse, f_error_df)
    
    indices = ["Full Model", "- Reduced Model", "Error"]#[reduced_label, full_label]
    df = [f_reg_df, f_reg_df - r_reg_df, f_error_df]
    ssrs = [f_ssr, r_ssr, ""] 
    sses = [f_sse, r_sse, ""]
    f = ["", f_val, ""]
    p = ["", p_val, ""]

    return pd.DataFrame({
        "DF" : df,
        "SS Err." : sses,
        "SS Reg." : ssrs, 
        "F" : f,
        "p" : p},
        index = indices, columns = ["DF", "SS Err.", "SS Reg.", "F", "p"])
    
    
    
