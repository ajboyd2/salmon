import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import product

from .expression import Expression, Var, Quantitative, Categorical, Interaction, Combination

plt.style.use('ggplot')

class Model:
    def __init__(self):
        raise NotImplementedError()
        
    def fit(self, data):
        raise NotImplementedError()
        
    def predict(self, data):
        raise NotImplementedError()
        
    def plot_matrix(self, **kwargs):
        df = pd.concat([self.training_x, self.training_y], axis = 1)
        scatter_matrix(df, **kwargs)
    
    
class LinearModel(Model):
    def __init__(self, explanatory, response, intercept = True):
        self.given_ex = explanatory
        self.given_re = response
        self.ex = None
        self.re = None
        self.intercept = intercept
        self.bhat = None
        self.fitted = None
        self.residuals = None
        self.std_err_est = None
        self.std_err_vars = None
        self.var = None
        self.t_vals = None
        self.p_vals = None
        self.training_data = None
        self.training_x = None
        self.training_y = None
        self.categorical_levels = dict()

    def fit(self, X, Y = None):
        # Wrapper to provide compatibility for sklearn functions
        if Y is None:
            data = X
        else:
            data = pd.concat([X,Y], axis = 1)
        return self._fit(data)
        
    def _fit(self, data):
        # Initialize the categorical levels
        self.categorical_levels = dict()
        self.training_data = data
        
        # Replace all Var's with either Q's or C's
        self.ex = self.given_ex.copy()
        self.ex = self.ex.interpret(data)
        self.re = self.given_re.copy()
        self.re = self.re.interpret(data)       
        
        # Construct X matrix
        X = self.extract_columns(self.ex, data)
        if self.intercept:
            X = pd.concat([LinearModel.ones_column(data), X], axis = 1)
        self.training_x = X
        # Construct Y vector
        y = self.extract_columns(self.re, data)
        self.training_y = y
        # Ensure correct dimensionality of y
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise Exception("Response variable of linear model can only be a single term expression.")
        # Solve equation
        self.bhat = pd.DataFrame(np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)), 
                                 index=X.columns, columns = ["Coefficients"])
        
        n = X.shape[0]
        p = X.shape[1] - 1

        self.fitted = pd.DataFrame({"Fitted" : np.dot(X, self.bhat).sum(axis = 1)})
        self.residuals = pd.DataFrame({"Residuals" : y.iloc[:,0] - self.fitted.iloc[:,0]})
        self.std_err_est = ((self.residuals["Residuals"] ** 2).sum() / (n - p - 1)) ** 0.5
        self.var = np.linalg.solve(np.dot(X.T, X), (self.std_err_est ** 2) * np.identity(p + 1))
        self.std_err_vars = pd.DataFrame({"SE" : (np.diagonal(self.var)) ** 0.5})
        self.t_vals = pd.DataFrame({"t" : self.bhat["Coefficients"].reset_index(drop = True) / self.std_err_vars["SE"]})
        self.p_vals = pd.DataFrame({"p" : pd.Series(stats.t.cdf(self.t_vals["t"], n - p - 1)).apply(lambda x: 2 * x if x < 0.5 else 2 * (1 - x))})
        ret_val = pd.concat([self.bhat.reset_index(), self.std_err_vars, self.t_vals, self.p_vals], axis = 1).set_index("index")
        ret_val.index.name = None # Remove oddity of set_index
        
        return ret_val 
        
    def predict(self, data, for_plot = False):
        # Construct the X matrix
        X = self.extract_columns(self.ex, data, multicolinearity_drop = not for_plot)
        if self.intercept:
            X = pd.concat([LinearModel.ones_column(data), X], axis = 1)

        # For plotting with categorical lines
        if for_plot:
            columns_present = set(list(X)) # Need to check if can do just set(X)
            columns_needed = self.bhat.index.format()
            columns_to_add = set(columns_needed) - columns_present
            for column in columns_to_add:
                X[column] = 0 # Add all non-present columns
            X = X[columns_needed] # Remove unneccessary columns
        
        # Multiply the weights to each column and sum across rows
        return pd.DataFrame({"Predicted " + str(self.re) : np.dot(X, self.bhat).sum(axis = 1)})
    
    def score(self, X, y, **kwargs):
        # Allow interfacing with sklearn's cross fold validation
        #self.fit(X, y)
        pred = self.predict(X)
        sse = ((y.iloc[:,0] - pred.iloc[:,0]) ** 2).sum()
        ssto = ((y.iloc[:,0] - y.iloc[:,0].mean()) ** 2).sum()
        mse = sse / (len(y) - len(self.training_x.columns) - 2)
        msto = ssto / (len(y) - 1)
        return 1 - mse / msto # Adjusted R^2
        
    def plot(self, categorize_residuals = True, jitter = None):
        terms = self.ex.flatten(True)
        unique_quants = list({term.name for term in terms if isinstance(term, Quantitative)})
        unique_cats = list({term.name for term in terms if isinstance(term, Categorical)})
        if len(unique_quants) == 1:
            unique_quant = unique_quants.pop()
            
            x = self.training_data[unique_quant]
            min_x = min(x)
            min_x = min(min_x * 1.05, min_x * 0.95) # Add a small buffer
            max_x = max(x)
            max_x = max(max_x * 1.05, max_x * 0.95)
           
            line_x = pd.DataFrame({unique_quant : np.linspace(min_x, max_x, 100)})
            if len(unique_cats) == 0:
                line_y = self.predict(line_x)
                line_fit, = plt.plot(line_x[unique_quant], line_y["Predicted " + str(self.re)])
            else:
                combinations = set(self.training_data[unique_cats].apply(lambda x: tuple(x), 1))
                plots = []
                labels = []
                linestyles = [':', '-.', '--', '-']
                for combination in combinations:
                    label = []
                    for element, var in zip(combination, unique_cats):
                        name = str(var)
                        line_x[name] = element
                        label.append(str(element))
                    line_type = linestyles.pop()
                    linestyles.insert(0, line_type)
                    line_y = self.predict(line_x, for_plot = True)
                    plot, = plt.plot(line_x[unique_quant], line_y["Predicted " + str(self.re)], linestyle = line_type)
                    plots.append(plot)
                    labels.append(", ".join(label))
                    if categorize_residuals:
                        indices_to_use = pd.Series([True] * len(x))
                        for element, var in zip(combination, unique_cats):
                            indices_to_use = indices_to_use & (self.training_data[var] == element)
                        plt.scatter(x[indices_to_use], self.training_y[str(self.re)][indices_to_use], c = plot.get_color())
                plt.legend(plots, labels, title = ", ".join(unique_cats), loc = "best")
            if not categorize_residuals:
                resids = plt.scatter(x, self.training_y[str(self.re)], c = "black")
                #plots.append(resids)
                #labels.append("Residuals")
            plt.xlabel(unique_quant)
            plt.ylabel(str(self.re))
            plt.grid()
            plt.show()
        elif len(unique_quants) == 0 and len(unique_cats) > 0:
            cats_levels = self.training_data[unique_cats].apply(lambda x: len(set(x)), 0)
            ml_cat = cats_levels.idxmax() # Category with the most levels
            ml_index = unique_cats.index(ml_cat) # Index corresponding to ml_cat
            cats_wo_most = unique_cats[:]
            cats_wo_most.remove(ml_cat) # List of categorical variables without the ml_cat
            single_cat = len(cats_wo_most) == 0
            if single_cat:
                combinations = {None}
            else:
                combinations = set(self.training_data[cats_wo_most].apply(lambda x: tuple(x), 1))
            
            line_x = pd.DataFrame({ml_cat : self.categorical_levels[ml_cat]}).reset_index() # To produce an index column
            points = pd.merge(self.training_data, line_x, on = ml_cat)
            
            plots = []
            labels = []
            linestyles = [':', '-.', '--', '-']
            for combination in combinations:
                points_indices = pd.Series([True] * len(points))
                if not single_cat:
                    label = []
                    for element, var in zip(combination, cats_wo_most):
                        name = str(var)
                        line_x[name] = element
                        label.append(str(element))
                        points_indices = points_indices & (points[name] == element) # Filter out points that don't apply to categories
                    labels.append(", ".join(label))
                line_type = linestyles.pop()
                linestyles.insert(0, line_type)
                line_y = self.predict(line_x, for_plot = True)
                plot, = plt.plot(line_x.index, line_y["Predicted " + str(self.re)], linestyle = line_type)
                if jitter is None or jitter is True:
                    variability = np.random.normal(scale = 0.025, size = sum(points_indices))
                else:
                    variability = 0
                plt.scatter(points.loc[points_indices, 'index'] + variability, self.training_y.loc[points_indices, str(self.re)], c = plot.get_color())
                plots.append(plot)
            if not single_cat and len(cats_wo_most) > 0:
                plt.legend(plots, labels, title = ", ".join(cats_wo_most), loc = "best")
            plt.xlabel(ml_cat)
            plt.xticks(line_x.index, line_x[ml_cat])
            plt.ylabel(str(self.re))
            plt.grid()
            plt.show()
        else:
            raise Exception("Plotting line of best fit only expressions that reference a single variable.")

    def residual_plots(self):
        terms = list(self.training_x)
        plots = []
        for term in terms:
            plots.append(plt.scatter(self.training_x[str(term)], self.residuals))
            plt.xlabel(str(term))
            plt.ylabel("Residuals")
            plt.title(str(term) + " v. Residuals")
            plt.grid()
            plt.show()
        return plots
        
    def partial_plots(self):
        terms = self.ex.flatten(separate_interactions = False)

        for i in range(0, len(terms)):
        
            xi = terms[i]
            sans_xi = sum(terms[:i] + terms[i+1:])
            yaxis = LinearModel(sans_xi, self.re)
            xaxis = LinearModel(sans_xi, xi)
            
            yaxis.fit(self.training_data)
            xaxis.fit(self.training_data)
            
            plt.scatter(xaxis.residuals, yaxis.residuals)
            plt.title("Leverage Plot for " + str(xi))
            plt.show()
            
    # static method
    def ones_column(data):
        return pd.DataFrame({"Intercept" : np.repeat(1, data.shape[0])})
        
    def extract_columns(self, expr, data, drop_dummy = True, update_levels = True, multicolinearity_drop = True):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise Exception("Only DataFrames and Series are supported for LinearModel.")
    
        if isinstance(expr, Combination):
            columns = [self.extract_columns(e, data, drop_dummy = drop_dummy, update_levels = update_levels, multicolinearity_drop = multicolinearity_drop) for e in expr.flatten()]
            return pd.concat(columns, axis = 1)
            
        elif isinstance(expr, Interaction):
            columns = [self.extract_columns(e, data, drop_dummy = False, update_levels = update_levels, multicolinearity_drop = multicolinearity_drop) for e in expr.flatten(True)]
            
            product = columns[0]
            for col in columns[1:]:
                # This is to account for the case where there are interactions between categorical variables
                to_combine = []
                for prior_column in product:
                    for former_column in col:
                        prior_name = prior_column if prior_column[0] == "{" else "{" + prior_column + "}"
                        former_name = former_column if former_column[0] == "{" else "{" + former_column + "}"
                        to_combine.append(pd.DataFrame({prior_name + former_name : product[prior_column] * col[former_column]}))
                product = pd.concat(to_combine, axis = 1)
            
            if multicolinearity_drop:
                product = product.loc[:, (product != 0).any(axis = 0)] # Ensure no multicolinearity
            if product.shape[1] > 1:
                # Must drop the dummy variable at this point
                return product.iloc[:,1:]
            else:
                return LinearModel.transform(expr, product)
        
        elif isinstance(expr, Quantitative):
            if expr.name not in list(data):
                raise Exception("Variable { " + expr.name + " } not found within data.")
            return LinearModel.transform(expr, pd.DataFrame({str(expr) : data[expr.name]}))
            
        elif isinstance(expr, Categorical):
            if expr.name not in list(data):
                raise Exception("Variable { " + expr.name + " } not found within data.")
            
            other_flag = False
            if expr.name in self.categorical_levels:
                levels = self.categorical_levels[expr.name]
            else:
                if expr.levels is None:
                    if expr.name in self.categorical_levels:
                        levels = self.categorical_levels[expr.name]
                    else:
                        levels = data[expr.name].unique()
                        levels.sort() # Give the levels an order
                else:
                    levels = expr.levels
                    data_levels = data[expr.name].unique()
                    for level in levels[:]:
                        if level not in data_levels:
                            levels.remove(level) # Remove levels that are not present in the dataset to avoid multicolinearity
                    for level in data_levels:
                        if level not in levels:
                            levels.append("~other~") # If there are other levels than those specified create a catchall category for them
                            other_flag = True
                            break
                            
            if update_levels:
                self.categorical_levels[str(expr)] = levels
            
            last_index = len(levels) - 1 if other_flag else len(levels)
            if expr.method == "one-hot":
                columns = pd.DataFrame({expr.name + "::" + str(level) : (data[expr.name] == level) * 1.0 for level in levels[:last_index]})
                if other_flag:
                    columns[expr.name + "::~other~"] = (columns.sum(axis = 1) == 0.0) * 1.0 # Whatever rows have not had a 1 yet get a 1 in the ~~other~~ column
                columns = columns[[expr.name + "::" + str(level) for level in levels]] # Make sure columns have correct order

                if drop_dummy:
                    return columns.drop(expr.name + "::" + str(levels[0]), axis = 1)
                else:
                    return columns
            
        else:
            raise Exception("LinearModel only suppoprts expressions consisting of Quantitative, Categorical, Interaction, and Combination.")
            
    # static method
    def transform(expr, data):
        if not isinstance(expr, Var):
            raise Exception("Transformation of data is only supported on singular terms of expressions.")
        
        if isinstance(expr.transformation, (int, float)):
            return ((data + expr.shift) * expr.coefficient) ** expr.transformation
        elif isinstance(expr.transformation, str):
            return getattr(np, expr.transformation)((data + expr.shift) * expr.coefficient)
        else:
            raise Exception("Transformation of data only supported for powers and numpy functions.")

        