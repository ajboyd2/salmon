import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import product

from .expression import Expression, Var, Quantitative, Categorical, Interaction, Combination, Identity

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
        if intercept:
            self.given_ex = explanatory + 1
        else:
            self.given_ex = explanatory    
        constant = self.given_ex.reduce()['Constant']
        self.intercept = constant is not None
        if self.intercept:
            self.given_ex = self.given_ex - constant # This was done to easily check all options for indicating a wanted intercept
                
        self.given_re = Identity(response) # This will collapse any combination of variables into a single column
        self.ex = None
        self.re = None
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
        
    def __str__(self):
        return str(self.given_ex) + " ~ " + str(self.given_re)

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
        
        terms = self.ex.reduce()
        
        # Construct X matrix
        X = self.ex.evaluate(data)
        X_means = X.mean()
        self.training_x = X
        self.training_x_means = X_means
        # Construct Y vector
        y = self.re.evaluate(data)
        y_mean = y.mean()
        self.training_y = y
        self.training_y_mean = y_mean

        # Center if there is an intercept
        if self.intercept:
            X = X - X_means
            y = y - y_mean
        
        # Solve equation
        self.bhat = pd.DataFrame(np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)), 
                                 index=X.columns, columns = ["Coefficients"])
        if self.intercept:
            self.bhat.loc["Intercept"] = [y_mean[0] - X_means.dot(self.bhat)[0]]
            X = X + X_means
            X['Intercept'] = 1
            y = y + y_mean
            
            
    
        n = X.shape[0]
        p = X.shape[1] - (1 if self.intercept else 0)

        self.fitted = pd.DataFrame({"Fitted" : np.dot(X, self.bhat).sum(axis = 1)})
        self.residuals = pd.DataFrame({"Residuals" : y.iloc[:,0] - self.fitted.iloc[:,0]})
        self.std_err_est = ((self.residuals["Residuals"] ** 2).sum() / (n - p - 1)) ** 0.5
        self.var = np.linalg.solve(np.dot(X.T, X), (self.std_err_est ** 2) * np.identity(X.shape[1]))
        self.std_err_vars = pd.DataFrame({"SE" : (np.diagonal(self.var)) ** 0.5})
        # format the covariance matrix
        self.var = pd.DataFrame(self.var, columns = X.columns, index = X.columns)
        # inference
        self.t_vals = pd.DataFrame({"t" : self.bhat["Coefficients"].reset_index(drop = True) / self.std_err_vars["SE"]})
        self.p_vals = pd.DataFrame({"p" : pd.Series(stats.t.cdf(self.t_vals["t"], n - p - 1)).apply(lambda x: 2 * x if x < 0.5 else 2 * (1 - x))})
        ret_val = pd.concat([self.bhat.reset_index(), self.std_err_vars, self.t_vals, self.p_vals], axis = 1).set_index("index")
        ret_val.index.name = None # Remove oddity of set_index
        
        return ret_val 
        
    def predict(self, data, for_plot = False):
        # Construct the X matrix
        X = self.ex.evaluate(data, fit = False)
        if self.intercept:
            X['Intercept'] = 1
            
        # For plotting with categorical lines
        #if for_plot:
        #    columns_present = set(list(X)) # Need to check if can do just set(X)
        #    columns_needed = self.bhat.index.format()
        #    columns_to_add = set(columns_needed) - columns_present
        #    for column in columns_to_add:
        #        X[column] = 0 # Add all non-present columns
        #    X = X[columns_needed] # Remove unneccessary columns
        
        # Multiply the weights to each column and sum across rows
        return pd.DataFrame({"Predicted " + str(self.re) : X.dot(self.bhat).sum(axis = 1)})
    
    def get_sse(self):
        sse = ((self.training_y.iloc[:,0] - self.fitted.iloc[:,0]) ** 2).sum()
        return sse
        
    def get_ssr(self):
        ssr = self.get_sst() - self.get_sse()
        return ssr
    
    def get_sst(self):
        sst = ((self.training_y.iloc[:,0] - self.training_y[:,0].mean()) ** 2).sum()
        return sst
    
    def score(self, X, y, **kwargs):
        # Allow interfacing with sklearn's cross fold validation
        #self.fit(X, y)
        pred = self.predict(X)
        sse = ((y.iloc[:,0] - pred.iloc[:,0]) ** 2).sum()
        ssto = ((y.iloc[:,0] - y.iloc[:,0].mean()) ** 2).sum()
        mse = sse / (len(y) - len(self.training_x.columns) - 2)
        msto = ssto / (len(y) - 1)
        return 1 - mse / msto # Adjusted R^2

    def _prediction_interval_width(self, X_new, alpha = 0.05):
        n = self.training_x.shape[0]
        p = X_new.shape[1]
        mse = self.get_sse() / (n - p)
        s_yhat_squared = (X_new.dot(self.var) * X_new).sum(axis = 1) # X_new_vect * var * X_new_vect^T (equivalent to np.diag(X_new.dot(self.var).dot(X_new.T)))
        s_pred_squared = mse + s_yhat_squared

        t_crit = stats.t.ppf(1 - (alpha / 2), n-p)

        return t_crit * (s_pred_squared ** 0.5)

    def _confidence_interval_width(self, X_new, alpha = 0.05):
        n = self.training_x.shape[0]
        p = X_new.shape[1]
        s_yhat_squared = (X_new.dot(self.var) * X_new).sum(axis = 1) # X_new_vect * var * X_new_vect^T (equivalent to np.diag(X_new.dot(self.var).dot(X_new.T)))
        #t_crit = stats.t.ppf(1 - (alpha / 2), n-p)
        W_crit_squared = p * stats.f.ppf(1 - (alpha / 2), p, n-p)
        return (W_crit_squared ** 0.5) * (s_yhat_squared ** 0.5)
        
    def plot(self, categorize_residuals = True, jitter = None, confidence_band = False, prediction_band = False):
        if confidence_band and prediction_band:
            raise Exception("One one of {confidence_band, prediction_band} may be set to True at a time.")

        terms = self.ex.reduce()
        
        # Plotting Details:
        min_y = min(self.training_y[str(self.re)])
        max_y = max(self.training_y[str(self.re)])
        diff = (max_y - min_y) * 0.05
        min_y = min(min_y - diff, min_y + diff) # Add a small buffer
        max_y = max(max_y - diff, max_y + diff) # TODO: Check if min() and max() are necessary here
                
        fig = plt.figure()
        ax = plt.subplot(111)
        
        if len(terms['Q']) == 1:
            return self._plot_one_quant(categorize_residuals, 
                                        jitter, 
                                        terms,
                                        confidence_band,
                                        prediction_band,
                                        {"figure" : fig, 
                                         "ax" : ax, 
                                         "y" : {"min" : min_y, 
                                         "max" : max_y,
                                         "name" : str(self.re)}})
        elif len(terms['Q']) == 0 and len(terms['C']) > 0:
            return self._plot_zero_quant() # TODO Make function
        else:
            raise Exception("Plotting line of best fit only expressions that reference a single variable.")
                                                      
    def _plot_one_quant(self, categorize_residuals, jitter, terms, confidence_band, prediction_band, plot_objs):
        x_term = next(iter(terms['Q'])) # Get the "first" and only element in the set 
        x_name = str(x_term)
        x = self.training_data[x_name]
        min_x = min(x)
        max_x = max(x)
        diff = (max_x - min_x) * 0.05
        min_x = min(min_x - diff, min_x + diff) # Add a small buffer
        max_x = max(max_x - diff, max_x + diff) # TODO: Check if min() and max() are necessary here
        
        plot_objs['x'] = {"min" : min_x, "max" : max_x, "name" : x_name}
        
        # Quantitative inputs
        line_x = pd.DataFrame({x_name : np.linspace(min_x, max_x, 100)})
        
        if len(terms['C']) == 0:
            self._plot_one_quant_zero_cats(x, line_x, jitter, terms, confidence_band, prediction_band, plot_objs)
        else:
            self._plot_one_quant_some_cats(x, line_x, categorize_residuals, jitter, terms, confidence_band, prediction_band, plot_objs)

        plt.xlabel(x_name)
        plt.ylabel(plot_objs['y']['name'])
        plt.grid()
        plot_objs['ax'].set_xlim([min_x, max_x])
        plot_objs['ax'].set_ylim([plot_objs['y']['min'], plot_objs['y']['max']])
                                                      
    def _plot_one_quant_zero_cats(self, x, line_x, jitter, terms, confidence_band, prediction_band, plot_objs):
        x_name = plot_objs['x']['name']
        line_y = self.predict(line_x)
        y_vals = line_y["Predicted " + plot_objs['y']['name']]
        line_fit, = plt.plot(line_x[x_name], y_vals)

        if confidence_band:
            self._plot_band(line_x, y_vals, line_fit.get_color(), plot_objs, True)
        elif prediction_band:
            self._plot_band(line_x, y_vals, line_fit.get_color(), plot_objs, False)

        plot_objs['ax'].scatter(x, self.training_y[plot_objs['y']['name']], c = "black")

    def _plot_band(self, line_x, y_vals, color, plot_objs, use_confidence = False, alpha = 0.05): # By default will plot prediction bands
        x_name = plot_objs['x']['name']
        X_new = self.ex.evaluate(line_x, fit = False)
        if self.intercept:
            X_new['Intercept'] = 1

        if use_confidence:
            widths = self._confidence_interval_width(X_new, alpha)
        else:
            widths = self._prediction_interval_width(X_new, alpha)
        plot_objs['ax'].fill_between(x = line_x[x_name], y1 = y_vals - widths, y2 = y_vals + widths, color = color, alpha = 0.3)
        
        
    def _plot_one_quant_some_cats(self, x, line_x, categorize_residuals, jitter, terms, confidence_band, prediction_band, plot_objs):
        ax = plot_objs['ax']
        x_name = plot_objs['x']['name']
        y_name = plot_objs['y']['name']
        

        plots = []
        labels = []
        linestyles = [':', '-.', '--', '-']
        
        cats = list(terms['C'])
        cat_names = [str(cat) for cat in cats]
        levels = [cat.levels for cat in cats]
        level_combinations = product(*levels) #cartesian product of all combinations
        
        dummy_data = line_x.copy() # rest of columns set in next few lines
       
        for level_set in level_combinations:
            label = [] # To be used in legend
            for (cat,level) in zip(cats,level_set):
                dummy_data[str(cat)] = level # set dummy data for prediction
                label.append(str(level))
                               
            line_type = linestyles.pop() # rotate through line styles
            linestyles.insert(0, line_type)
            
            line_y = self.predict(dummy_data, for_plot = True)
            y_vals = line_y["Predicted " + y_name]
            plot, = ax.plot(dummy_data[x_name], y_vals, linestyle = line_type)
            plots.append(plot)
            labels.append(", ".join(label))
            
            if categorize_residuals:
                indices_to_use = pd.Series([True] * len(x)) # gradually gets filtered out
                for (cat,level) in zip(cats,level_set):
                    indices_to_use = indices_to_use & (self.training_data[str(cat)] == level)
                ax.scatter(x[indices_to_use], self.training_y[y_name][indices_to_use], c = plot.get_color())
            
            if confidence_band:
                self._plot_band(dummy_data, y_vals, plot.get_color(), plot_objs, True)
            elif prediction_band:
                self._plot_band(dummy_data, y_vals, plot.get_color(), plot_objs, False)

        # Legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(plots, labels, title = ", ".join(cat_names), loc = "center left", bbox_to_anchor = (1, 0.5))
        
        if not categorize_residuals:
            resids = ax.scatter(x, self.training_y[str(self.re)], c = "black")
        
    # Deprecated
    def _plot(self, categorize_residuals = True, jitter = None):
        terms = self.ex.reduce()
        unique_quants = [term.name for term in terms['Q']]
        unique_cats = [term.name for term in terms['C']]
        self.categorical_levels = {term.name : term.levels for term in terms['C']}
        
        min_y = min(self.training_y[str(self.re)])
        max_y = max(self.training_y[str(self.re)])
        diff = (max_y - min_y) * 0.05
        min_y = min(min_y - diff, min_y + diff) # Add a small buffer
        max_y = max(max_y - diff, max_y + diff) # TODO: Check if min() and max() are necessary here
           
        
        fig = plt.figure()
        ax = plt.subplot(111)
        
        if len(unique_quants) == 1:
            unique_quant = unique_quants.pop()
            
            x = self.training_data[unique_quant]
            min_x = min(x)
            max_x = max(x)
            diff = (max_x - min_x) * 0.05
            min_x = min(min_x - diff, min_x + diff) # Add a small buffer
            max_x = max(max_x - diff, max_x + diff) # TODO: Check if min() and max() are necessary here
                        
            line_x = pd.DataFrame({unique_quant : np.linspace(min_x, max_x, 100)})
            if len(unique_cats) == 0:
                line_y = self.predict(line_x)
                line_fit, = plt.plot(line_x[unique_quant], line_y["Predicted " + str(self.re)])
                ax.scatter(x, self.training_y[str(self.re)], c = "black")
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
                    plot, = ax.plot(line_x[unique_quant], line_y["Predicted " + str(self.re)], linestyle = line_type)
                    plots.append(plot)
                    labels.append(", ".join(label))
                    if categorize_residuals:
                        indices_to_use = pd.Series([True] * len(x))
                        for element, var in zip(combination, unique_cats):
                            indices_to_use = indices_to_use & (self.training_data[var] == element)
                        ax.scatter(x[indices_to_use], self.training_y[str(self.re)][indices_to_use], c = plot.get_color())
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(plots, labels, title = ", ".join(unique_cats), loc = "center left", bbox_to_anchor = (1, 0.5))
            if not categorize_residuals:
                resids = ax.scatter(x, self.training_y[str(self.re)], c = "black")
                #plots.append(resids)
                #labels.append("Residuals")
            plt.xlabel(unique_quant)
            plt.ylabel(str(self.re))
            plt.grid()
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            #fig.show()
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
                plot, = ax.plot(line_x.index, line_y["Predicted " + str(self.re)], linestyle = line_type)
                if jitter is None or jitter is True:
                    variability = np.random.normal(scale = 0.025, size = sum(points_indices))
                else:
                    variability = 0
                # Y values must come from points because earlier merge shuffles rows
                ax.scatter(points.loc[points_indices, 'index'] + variability, points.loc[points_indices, str(self.re)], c = plot.get_color())
                plots.append(plot)
            if not single_cat and len(cats_wo_most) > 0:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(plots, labels, title = ", ".join(cats_wo_most), loc = "center left", bbox_to_anchor=(1, 0.5))
            plt.xlabel(ml_cat)
            plt.xticks(line_x.index, line_x[ml_cat])
            plt.ylabel(str(self.re))
            plt.grid()
            ax.set_ylim([min_y, max_y])

        else:
            raise Exception("Plotting line of best fit only expressions that reference a single variable.")

    def residual_plots(self):
        terms = list(self.training_x)
        plots = []
        for term in terms:
            plots.append(plt.scatter(self.training_x[str(term)], self.residuals['Residuals']))
            plt.xlabel(str(term))
            plt.ylabel("Residuals")
            plt.title(str(term) + " v. Residuals")
            plt.grid()
            plt.show()
        return plots
        
    def partial_plots(self):
        #terms = self.ex.flatten(separate_interactions = False)
        term_dict = self.ex.reduce()
        terms = list(terms['Q']) + list(terms['C'])

        for i in range(0, len(terms)):
        
            xi = terms[i]

            sans_xi = Combination(terms[:i] + terms[i+1:])
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

        