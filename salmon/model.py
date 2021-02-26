"""Describes various linear models supported by SALMON."""

import numpy as np

import scipy.stats as stats
from scipy.linalg import solve_triangular, cho_solve

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from itertools import product
from collections import OrderedDict

from .expression import Combination, Identity, Constant

plt.style.use('ggplot')


def _float_format(x):
    abs_x = abs(x)
    if abs_x >= 1e4:
        rep = "{:.3e}".format(x)
    elif abs_x >= 1e0:
        rep = "{:."
        rep += str(3 - int(np.floor(np.log10(abs_x))))
        rep += "f}"
        rep = rep.format(x)
    elif abs_x >= 1e-3:
        rep = "{:.4f}".format(x)
    elif abs_x >= 1e-9:
        rep = "{:.3e}".format(x)
    elif abs_x >= 1e-99:
        rep = "{:.1e}".format(x)
    else:
        rep = "{:.0e}".format(x)
    rep = rep.replace("e-0", "e-").replace("e+0", "e+")
    rep = rep.replace("0e+0", "0.000")
    return rep


pd.set_option("display.float_format", _float_format)


def _confint(estimates, standard_errors, df, crit_prob):
    crit_value = stats.t.ppf(crit_prob, df)
    ci_widths = crit_value * standard_errors
    return estimates - ci_widths, estimates + ci_widths


def qr_solve(Q, R, y):
    """Solve least squares X / y, given QR decomposition of X"""
    _, p = R.shape
    if p:
        return solve_triangular(R, Q.T @ y, check_finite=False)
    else:
        return np.empty(shape=0)


def cho_inv(R):
    """Calculate inverse of X.T @ X, given Cholesky decomposition R.T @ R"""
    _, p = R.shape
    if p:
        return cho_solve((R, False), np.identity(p), check_finite=False)
    else:
        return np.empty(shape=(0, 0))


class Model:
    """A general Model class that both Linear models and (in the future)
    General Linear models stem from."""

    def __init__(self):
        """Create a Model object (only possible through inheritance)."""
        raise NotImplementedError()

    def fit(self, data):
        """Fit a model to given data.

        Arguments:
            data - A DataFrame with column names matching specified terms
                within the Model's explanatory and response Expression
                objects.

        Returns:
            A DataFrame with relevant statistics of fitted Model (coefficients,
            t statistics, p-values, etc.).
        """
        raise NotImplementedError()

    def predict(self, data):
        """Predict response values for a given set of data.

        Arguments:
            data - A DataFrame with column names matching specified terms
                within the Model's explanatory Expression object.

        Returns:
            A Series of the predicted values.
        """
        raise NotImplementedError()

    def plot_matrix(self, **kwargs):
        """Produce a matrix of pairwise scatter plots of the data it was fit
        on. The diagonal of the matrix will feature histograms instead of
        scatter plots.

        Arguments:
            kwargs - One or more named parameters that will be ingested by
                Pandas' scatter_matrix plotting function.

        Returns:
            A matplotlib plot object containing the matrix of scatter plots.
        """
        df = pd.concat([self.X_train_, self.y_train_], axis=1)
        scatter_matrix(df, **kwargs)


class LinearModel(Model):
    """A specific Model that assumes the response variable is linearly related
    to the explanatory variables."""

    def __init__(self, explanatory, response, intercept=True):
        """Create a LinearModel object.

        An intercept is included in the model by default. To fit a model
        without an intercept term, either set intercept=False or subtract '1'
        from the explanatory Expression.

        Arguments:
            explanatory - An Expression that is either a single term or a
                Combination of terms. These are the X's.
            response - An Expression that represents the single term for the
                response variables. This is the y. If this is a Combination,
                the terms will be added together and treated as a single
                variable.
            intercept - A boolean indicating whether an intercept should be
                included (True) or not (False).
        """
        if explanatory is None:
            explanatory = 0

        if isinstance(explanatory, (int, float)):
            explanatory = Constant(explanatory)

        if intercept:
            self.given_ex = explanatory + 1
        else:
            if explanatory == Constant(0):
                raise Exception(
                    "Must have at least one predictor in explanatory "
                    "expression and/or intercept enabled for a valid model."
                )
            self.given_ex = explanatory
        constant = self.given_ex.reduce()['Constant']
        self.intercept = constant is not None
        if self.intercept:
            # This was done to easily check all options for indicating
            # a wanted intercept
            self.given_ex = self.given_ex - constant

        # This will collapse any combination of variables into a single column
        self.given_re = Identity(response)
        self.ex = None
        self.re = None

        self.training_data = None

        self.categorical_levels = dict()

    def __str__(self):
        """Convert a LinearModel to a str format for printing."""
        if self.intercept:
            return str(self.given_re) + " ~ " + str(1 + self.given_ex)
        else:
            return str(self.given_re) + " ~ " + str(self.given_ex)

    def fit(self, X, y=None):
        """Fit a LinearModel to data..

        Data can either be provided as a single DataFrame X that contains both
        the explanatory and response variables, or in separate data
        structures, one containing the explanatory variables and the other
        containing the response variable. The latter is implemented so that
        LinearModel can be used as scikit-learn Estimator.

        It is fine to have extra columns in the DataFrame that are not used by
        the model---they will simply be ignored.

        Arugments:
            X - A DataFrame containing all of the explanatory variables in the
                model and possibly the response variable too.
            y - An optional Series that contains the response variable.

        Returns:
            A DataFrame containing relevant statistics of fitted Model (e.g.,
            coefficients, p-values).
        """
        if y is None:
            data = X
        else:
            data = pd.concat([X, y], axis=1)
        return self._fit(data)

    def _fit(self, data):

        # Initialize the categorical levels
        self.categorical_levels = dict()
        self.training_data = data

        # Replace all Var's with either Q's or C's
        self.re = self.given_re.copy().interpret(data)
        self.ex = self.given_ex.copy().interpret(data)

        # Construct X matrix
        X = self.ex.evaluate(data)
        self.X_train_ = X
        # Construct y vector
        y = self.re.evaluate(data)[:, 0]
        self.y_train_ = y

        # Get dimensions
        self.n, self.p = X.shape

        # Center if there is an intercept
        if self.intercept:
            X_offsets = X.mean(axis=0)
            y_offset = y.mean()
            X -= X_offsets[np.newaxis, :]
        else:
            X_offsets = 0
            y_offset = 0

        # Get coefficients using QR decomposition
        q, r = np.linalg.qr(X)
        coef_ = qr_solve(q, r, y - y_offset)
        cols = X.columns.copy()  # column names

        # Get fitted values and residuals
        self.fitted_ = y_offset + np.dot(X, coef_)
        self.residuals_ = y - self.fitted_

        # Get residual variance
        self.rdf = self.n - self.p - (1 if self.intercept else 0)
        self.resid_var_ = (self.residuals_ ** 2).sum() / self.rdf

        # Get covariance matrix between coefficients
        self.cov_ = self.resid_var_ * cho_inv(r)

        # Update coefficients and covariance matrix with intercept
        # (if applicable)
        if self.intercept:
            cols.append("Intercept")
            coef_ = np.append(coef_, y_offset - (X_offsets * coef_).sum())
            cov_coef_intercept = -1*np.dot(self.cov_, X_offsets)

            var_intercept = self.resid_var_ / self.n
            var_intercept -= (X_offsets * cov_coef_intercept).sum()

            self.cov_ = np.block([
                [self.cov_, cov_coef_intercept[:, np.newaxis]],
                [cov_coef_intercept[np.newaxis, :], var_intercept]
            ])

        # Get standard errors (diagonal of the covariance matrix)
        se_coef_ = np.sqrt(np.diagonal(self.cov_))

        # Get inference for coefficients
        self.t_ = coef_ / se_coef_
        self.p_ = 2 * stats.t.cdf(-abs(self.t_), self.rdf)
        lower_bound, upper_bound = _confint(coef_, se_coef_, self.rdf, .975)

        # Create output table
        table = pd.DataFrame(OrderedDict((
            ("Coefficient", coef_), ("SE", se_coef_),
            ("t", self.t_), ("p", self.p_),
            ("2.5%", lower_bound), ("97.5%", upper_bound)
        )), index=cols)

        self.coef_ = table["Coefficient"]
        self.se_coef_ = table["SE"]

        return table

    def likelihood(self, data=None):
        """Calculate likelihood for a fitted model on either original data or
        new data."""
        return np.exp(self.log_likelihood(data))

    def log_likelihood(self, data=None):
        """Calculate a numerically stable log_likelihood for a fitted model
        on either original data or new data."""

        if data is None:
            residuals = self.residuals_
        else:
            y = self.re.evaluate(data)
            y_hat = self.predict(
                data,
                for_plot=False,
                confidence_interval=False,
                prediction_interval=False,
            )
            residuals = y[:, 0] - y_hat.iloc[:, 0]

        n = len(residuals)

        return (-n / 2 * (np.log(2 * np.pi) + np.log(self.resid_var_)) -
                (1 / (2 * self.resid_var_)) * (residuals ** 2).sum())

    def confidence_intervals(self, alpha=None, conf=None):
        """Calculate confidence intervals for the coefficients.

        This function assumes that Model.fit() has already been called.

        Arguments:
            alpha - A float between 0.0 and 1.0 representing the non-coverage
                probability of the confidence interval. In other words, the
                confidence level is 1 - alpha / 2.
            conf - A float between 0.0 and 1.0 representing the confidence
                level. Only one of alpha or conf needs to be specified. If
                neither are specified, a default value of conf=0.95 will be
                used.

        Returns:
            A DataFrame containing the appropriate confidence intervals for
            all the coefficients.
        """
        if alpha is None:
            if conf is None:
                conf = 0.95
            alpha = 1 - conf

        crit_prob = 1 - (alpha / 2)

        lower_bound, upper_bound = _confint(self.coef_, self.se_coef_,
                                            self.rdf, crit_prob)

        return pd.DataFrame({
            "%.1f%%" % (100 * (1 - crit_prob)): lower_bound,
            "%.1f%%" % (100 * crit_prob): upper_bound
        }, index=self.coef_.index)

    def predict(
        self,
        data,
        for_plot=False,
        confidence_interval=False,
        prediction_interval=False,
    ):
        """Predict response values from a fitted Model.

        Arguments:
            data - A DataFrame containing the values of the explanatory
                variables, for which predictions are desired.
            for_plot - A boolean indicating if these predictions are computed
                for the purposes of plotting.
            confidence_interval - If a confidence interval for the mean
                response is desired, this is a float between 0.0 and 1.0
                indicating the confidence level to use.
            prediction_interval - If a prediction interval is desired, this is
                a float between 0.0 and 1.0 indicating the confidence level to
                use.

        Returns:
            A DataFrame containing the predictions and/or intervals.
        """
        # Construct the X matrix
        X = self.ex.evaluate(data, fit=False)
        if self.intercept:
            n, _ = X.shape
            X = np.hstack((X, np.ones((n, 1))))
        y_vals = np.dot(X, self.coef_)

        predictions = pd.DataFrame(
            {"Predicted " + str(self.re): y_vals},
            index=data.index
        )

        if confidence_interval or prediction_interval:
            if confidence_interval:
                alpha = confidence_interval
                widths = self._confidence_interval_width(
                    X,
                    confidence_interval,
                )
            else:
                alpha = prediction_interval
                widths = self._prediction_interval_width(
                    X,
                    prediction_interval,
                )

            crit_prob = 1 - (alpha / 2)

            lower = y_vals - widths
            upper = y_vals + widths

            predictions[str(round(1 - crit_prob, 5) * 100) + "%"] = lower
            predictions[str(round(crit_prob, 5) * 100) + "%"] = upper

        return predictions

    def get_sse(self):
        """Get the SSE of a fitted model."""
        sse = ((self.y_train_ - self.fitted_) ** 2).sum()
        return sse

    def get_ssr(self):
        """Get the SSR of a fitted model."""
        ssr = self.get_sst() - self.get_sse()
        return ssr

    def get_sst(self):
        """Get the SST of a fitted model."""
        sst = ((self.y_train_ - self.y_train_.mean()) ** 2).sum()
        return sst

    def r_squared(self, X=None, y=None, adjusted=False, **kwargs):
        """Calculate the (adjusted) R^2 value of the model.
        This can be used as a metric within the sklearn ecosystem.

        Arguments:
            X - An optional DataFrame of the explanatory data to be used for
                calculating R^2. Default is the training data.
            y - An optional DataFrame of the response data to be used for
                calculating R^2. Default is the training data.
            adjusted - A boolean indicating if the R^2 value is adjusted
                (True) or not (False).

        Returns:
            A real value of the computed R^2 value.
        """
        # Allow interfacing with sklearn's cross fold validation
        # self.fit(X, y)
        if X is None:
            X = self.training_data
        if y is None:
            y = self.y_train_

        pred = self.predict(X)
        sse = ((y - pred.iloc[:, 0]) ** 2).sum()
        ssto = ((y - y.mean()) ** 2).sum()

        if adjusted:
            numerator = sse
            denominator = ssto
        else:
            numerator = sse / (len(y) - len(self.X_train_.columns) - 2)
            denominator = ssto / (len(y) - 1)

        return 1 - numerator / denominator

    def score(self, X=None, y=None, adjusted=False, **kwargs):
        """Wrapper for sklearn api for cross fold validation.
        
        See LinearModel.r_squared.
        """
        return self.r_squared(X, y, adjusted, **kwargs)

    def _prediction_interval_width(self, X_new, alpha=0.05):
        """Helper function for calculating prediction interval widths."""
        mse = self.get_sse() / self.rdf
        s_yhat_squared = (X_new.dot(self.cov_) * X_new).sum(axis=1)
        s_pred_squared = mse + s_yhat_squared

        t_crit = stats.t.ppf(1 - (alpha / 2), self.rdf)

        return t_crit * (s_pred_squared ** 0.5)

    def _confidence_interval_width(self, X_new, alpha=0.05):
        """Helper function for calculating confidence interval widths."""
        _, p = X_new.shape
        s_yhat_squared = (X_new.dot(self.cov_) * X_new).sum(axis=1)
        # t_crit = stats.t.ppf(1 - (alpha / 2), n-p)
        W_crit_squared = p * stats.f.ppf(1 - (alpha / 2), p, self.rdf)
        return (W_crit_squared ** 0.5) * (s_yhat_squared ** 0.5)

    def plot(
        self,
        categorize_residuals=True,
        jitter=None,
        confidence_band=False,
        prediction_band=False,
        original_y_space=True,
        transformed_y_space=False,
        alpha=0.5,
        **kwargs
    ):
        """Visualizes the fitted LinearModel and its line of best fit.

        Arguments:
            categorize_residuals - A boolean that indicates if the residual
                points should be colored by categories (True) or not (False).
            jitter - A boolean that indicates if residuals should be jittered
                in factor plots (True) or not (False).
            confidence_band - A real value that specifies the width of the
                confidence band to be plotted. If band not desired, parameter
                is set to False.
            prediction_band - A real value that specifies the width of the
                prediction band to be plotted. If band not desired, parameter
                is set to False.
            y_space - A str that indicates the type of output space for the
                y-axis. If set to 't', the transformed space will be plotted.
                If set to 'o', the original or untransformed space will be
                plotted. If set to 'b', both will be plotted side-by-side.
            alpha - A real value that indicates the transparency of the
                residuals. Default is 0.5.
            kwargs - Additional named parameters that will be passed onto
                lower level matplotlib plotting functions.

        Returns:
            A matplotlib plot appropriate visualization of the model.
        """
        if confidence_band and prediction_band:
            raise Exception(
                "Only one of {confidence_band, prediction_band} may "
                "be set to True at a time."
            )

        terms = self.ex.reduce()

        if original_y_space and transformed_y_space:
            fig, (ax_o, ax_t) = plt.subplots(1, 2, **kwargs)
            y_spaces = ['o', 't']
            axs = [ax_o, ax_t]
        elif transformed_y_space:  # at least one of the two is False
            fig, ax_t = plt.subplots(1, 1, **kwargs)
            y_spaces = ['t']
            axs = [ax_t]
        elif original_y_space:
            fig, ax_o = plt.subplots(1, 1, **kwargs)
            y_spaces = ['o']
            axs = [ax_o]
        else:
            raise AssertionError(
                "At least one of either `original_y_space` or "
                "`transformed_y_space` should be True in "
                "`model.plot(...)` call."
            )

        for y_space_type, ax in zip(y_spaces, axs):
            original_y_space = y_space_type == "o"

            # Redundant, we untransform in later function calls
            # TODO: Fix later
            y_vals = self.y_train_
            if original_y_space:
                y_vals = self.re.untransform(y_vals)
            # Plotting Details:
            min_y = min(y_vals)
            max_y = max(y_vals)
            diff = (max_y - min_y) * 0.05
            min_y = min(min_y - diff, min_y + diff)  # Add a small buffer
            max_y = max(max_y - diff, max_y + diff)
            # TODO: Check if min() and max() are necessary here

            plot_args = {
                "categorize_residuals": categorize_residuals,
                "jitter": jitter,
                "terms": terms,
                "confidence_band": confidence_band,
                "prediction_band": prediction_band,
                "original_y_space": original_y_space,
                "alpha": alpha,
                "plot_objs": {
                    "figure": fig,
                    "ax": ax,
                    "y": {
                        "min": min_y,
                        "max": max_y,
                        "name": str(self.re)
                    }
                }
            }

            if len(terms['Q']) == 1:
                self._plot_one_quant(**plot_args)
            elif len(terms['Q']) == 0 and len(terms['C']) > 0:
                self._plot_zero_quant(**plot_args)  # TODO Make function
            else:
                raise Exception(
                    "Plotting line of best fit only expressions that reference"
                    " a single variable."
                )
        return fig

    def _plot_zero_quant(
        self,
        categorize_residuals,
        jitter,
        terms,
        confidence_band,
        prediction_band,
        original_y_space,
        alpha,
        plot_objs,
    ):
        """A helper function for plotting models in the case no quantitiative
        variables are present.
        """

        ax = plot_objs['ax']
        unique_cats = list(terms['C'])
        levels = [cat.levels for cat in unique_cats]
        level_amounts = [len(level_ls) for level_ls in levels]
        ml_index = level_amounts.index(max(level_amounts))
        ml_cat = unique_cats[ml_index]
        ml_levels = levels[ml_index]

        # List of categorical variables without the ml_cat
        cats_wo_most = unique_cats[:]
        cats_wo_most.remove(ml_cat)

        # List of levels for categorical variables without the ml_cat
        levels_wo_most = levels[:]
        levels_wo_most.remove(levels[ml_index])

        single_cat = len(cats_wo_most) == 0
        if single_cat:
            level_combinations = [None]
        else:
            level_combinations = product(*levels_wo_most)  # Cartesian product

        # To produce an index column to be used for the x-axis alignment
        line_x = pd.DataFrame({str(ml_cat): ml_levels}).reset_index()
        points = pd.merge(self.training_data, line_x, on=str(ml_cat))

        plot_objs['x'] = {'name': 'index'}

        points["<Y_RESIDS_TO_PLOT>"] = self.re.evaluate(points)
        if original_y_space:
            points["<Y_RESIDS_TO_PLOT>"] = self.re.untransform(
                points["<Y_RESIDS_TO_PLOT>"]
            )
            # Inefficient due to transforming, then untransforming.
            # Need to refactor later.

        plots = []
        labels = []
        linestyles = [':', '-.', '--', '-']
        for combination in level_combinations:
            points_indices = pd.Series([True] * len(points))
            if not single_cat:
                label = []
                for element, var in zip(combination, cats_wo_most):
                    name = str(var)
                    line_x[name] = element
                    label.append(str(element))

                    # Filter out points that don't apply to categories
                    points_indices = points_indices & (points[name] == element)
                labels.append(", ".join(label))
            line_type = linestyles.pop()
            linestyles.insert(0, line_type)
            line_y = self.predict(line_x, for_plot=True)
            y_vals = line_y["Predicted " + plot_objs['y']['name']]
            if original_y_space:
                y_vals_to_plot = self.re.untransform(y_vals)
            else:
                y_vals_to_plot = y_vals
            plot, = ax.plot(line_x.index, y_vals_to_plot, linestyle=line_type)
            if jitter is None or jitter is True:
                variability = np.random.normal(
                    scale=0.025,
                    size=sum(points_indices),
                )
            else:
                variability = 0
            # Y values must come from points because earlier merge
            # shuffles rows
            ax.scatter(
                points.loc[points_indices, 'index'] + variability,
                points.loc[points_indices, "<Y_RESIDS_TO_PLOT>"],
                c="black" if single_cat else plot.get_color(),
                alpha=alpha,
            )
            plots.append(plot)

            if confidence_band:
                self._plot_band(
                    line_x,
                    y_vals,
                    plot.get_color(),
                    original_y_space,
                    plot_objs,
                    True,
                    confidence_band,
                )
            elif prediction_band:
                self._plot_band(
                    line_x,
                    y_vals,
                    plot.get_color(),
                    original_y_space,
                    plot_objs,
                    False,
                    prediction_band,
                )

        if not single_cat and len(cats_wo_most) > 0:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(
                plots,
                labels,
                title=", ".join([str(cat) for cat in cats_wo_most]),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        ax.set_xlabel(str(ml_cat))
        ax.set_xticks(line_x.index)
        ax.set_xticklabels(line_x[str(ml_cat)])
        if not original_y_space:
            ax.set_ylabel(plot_objs['y']['name'])
        else:
            ax.set_ylabel(self.re.untransform_name())
        ax.grid()
        ax.set_ylim([plot_objs['y']['min'], plot_objs['y']['max']])

    def _plot_one_quant(
        self,
        categorize_residuals,
        jitter,
        terms,
        confidence_band,
        prediction_band,
        original_y_space,
        alpha,
        plot_objs,
    ):
        """A helper function for plotting models in the case only one
        quantitiative variable is present. Also support zero or more
        categorical variables."""
        # Get the "first" and only element in the set
        x_term = next(iter(terms['Q']))

        x_name = str(x_term)
        x = self.training_data[x_name]
        min_x = min(x)
        max_x = max(x)
        diff = (max_x - min_x) * 0.05
        min_x = min(min_x - diff, min_x + diff)  # Add a small buffer
        max_x = max(max_x - diff, max_x + diff)
        # TODO: Check if min() and max() are necessary here

        plot_objs['x'] = {"min" : min_x, "max" : max_x, "name" : x_name}

        # Quantitative inputs
        line_x = pd.DataFrame({x_name: np.linspace(min_x, max_x, 100)})

        if len(terms['C']) == 0:
            self._plot_one_quant_zero_cats(
                x,
                line_x,
                jitter,
                terms,
                confidence_band,
                prediction_band,
                original_y_space,
                alpha,
                plot_objs,
            )
        else:
            self._plot_one_quant_some_cats(
                x,
                line_x,
                categorize_residuals,
                jitter,
                terms,
                confidence_band,
                prediction_band,
                original_y_space,
                alpha,
                plot_objs,
            )

        ax = plot_objs['ax']
        ax.set_xlabel(x_name)
        if not original_y_space:
            ax.set_ylabel(plot_objs['y']['name'])
        else:
            ax.set_ylabel(self.re.untransform_name())
        ax.grid()
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([plot_objs['y']['min'], plot_objs['y']['max']])

    def _plot_one_quant_zero_cats(
        self,
        x,
        line_x,
        jitter,
        terms,
        confidence_band,
        prediction_band,
        original_y_space,
        alpha,
        plot_objs,
    ):
        """A helper function for plotting models in the case only one
        quantitiative variable and no categorical variables are present."""

        x_name = plot_objs['x']['name']
        ax = plot_objs['ax']
        line_y = self.predict(line_x)
        y_vals = line_y["Predicted " + plot_objs['y']['name']]
        if original_y_space:
            y_vals_to_plot = self.re.untransform(y_vals)
        else:
            y_vals_to_plot = y_vals
        line_fit, = ax.plot(line_x[x_name], y_vals_to_plot)

        if confidence_band:
            self._plot_band(
                line_x,
                y_vals,
                line_fit.get_color(),
                original_y_space,
                plot_objs,
                True,
                confidence_band,
            )
        elif prediction_band:
            self._plot_band(
                line_x,
                y_vals,
                line_fit.get_color(),
                original_y_space,
                plot_objs,
                False,
                prediction_band,
            )

        y_train_vals = self.y_train_
        if original_y_space:
            y_train_vals = self.re.untransform(y_train_vals)

        ax.scatter(x, y_train_vals, c = "black", alpha = alpha)

    def _plot_band(
        self,
        line_x,
        y_vals,
        color,
        original_y_space,
        plot_objs,
        use_confidence=False,
        alpha=0.05,
    ):
        """A helper function to plot the confidence or prediction bands for
        a model. By default will plot prediction bands."""
        x_name = plot_objs['x']['name']
        X_new = self.ex.evaluate(line_x, fit = False)
        if self.intercept:
            n, _ = X_new.shape
            X_new = np.hstack((X_new, np.ones((n, 1))))

        if use_confidence:
            widths = self._confidence_interval_width(X_new, alpha)
        else:
            widths = self._prediction_interval_width(X_new, alpha)

        lower = y_vals - widths
        upper = y_vals + widths

        if original_y_space:
            lower = self.re.untransform(lower)
            upper = self.re.untransform(upper)

        plot_objs['ax'].fill_between(
            x=line_x[x_name],
            y1=lower,
            y2=upper,
            color=color,
            alpha=0.3,
        )


    def _plot_one_quant_some_cats(
        self,
        x,
        line_x,
        categorize_residuals,
        jitter,
        terms,
        confidence_band,
        prediction_band,
        original_y_space,
        alpha,
        plot_objs,
    ):
        """A helper function for plotting models in the case only one
        quantitiative variable and one or more categorical variables are
        present."""

        ax = plot_objs['ax']
        x_name = plot_objs['x']['name']
        y_name = plot_objs['y']['name']


        plots = []
        labels = []
        linestyles = [':', '-.', '--', '-']

        cats = list(terms['C'])
        cat_names = [str(cat) for cat in cats]
        levels = [cat.levels for cat in cats]
        # cartesian product of all combinations
        level_combinations = product(*levels)

        dummy_data = line_x.copy() # rest of columns set in next few lines

        y_train_vals = self.y_train_
        if original_y_space:
            y_train_vals = self.re.untransform(y_train_vals)

        for level_set in level_combinations:
            label = [] # To be used in legend
            for (cat,level) in zip(cats,level_set):
                dummy_data[str(cat)] = level # set dummy data for prediction
                label.append(str(level))

            line_type = linestyles.pop() # rotate through line styles
            linestyles.insert(0, line_type)

            line_y = self.predict(dummy_data, for_plot=True)
            y_vals = line_y["Predicted " + y_name]
            if original_y_space:
                y_vals_to_plot = self.re.untransform(y_vals)
            else:
                y_vals_to_plot = y_vals
            plot, = ax.plot(dummy_data[x_name], y_vals_to_plot, linestyle=line_type)
            plots.append(plot)
            labels.append(", ".join(label))


            if categorize_residuals:
                indices_to_use = pd.Series([True] * len(x)) # gradually gets filtered out
                for (cat,level) in zip(cats,level_set):
                    indices_to_use = indices_to_use & (self.training_data[str(cat)] == level)
                ax.scatter(
                    x[indices_to_use],
                    y_train_vals[indices_to_use],
                    c=plot.get_color(),
                    alpha=alpha,
                )

            if confidence_band:
                self._plot_band(
                    dummy_data,
                    y_vals,
                    plot.get_color(),
                    original_y_space,
                    plot_objs,
                    True,
                    confidence_band,
                )
            elif prediction_band:
                self._plot_band(
                    dummy_data,
                    y_vals,
                    plot.get_color(),
                    original_y_space,
                    plot_objs,
                    False,
                    prediction_band,
                )

        # Legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(
            plots,
            labels,
            title=", ".join(cat_names),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        if not categorize_residuals:
            ax.scatter(x, y_train_vals, c="black", alpha=alpha)

    def residual_plots(self, **kwargs):
        """Plot the residual plots of the model.

        Arguments:
            kwargs - Named parameters that will be passed onto lower level
                matplotlib plotting functions.

        Returns:
            A tuple containing the matplotlib (figure, list of axes) for the
            residual plots.
        """
        terms = self.X_train_.columns
        fig, axs = plt.subplots(1, len(terms), **kwargs)
        for term, ax in zip(terms, axs):
            ax.scatter(self.X_train_.get_column(term), self.residuals_)
            ax.set_xlabel(str(term))
            ax.set_ylabel("Residuals")
            ax.set_title(str(term) + " v. Residuals")
            ax.grid()
        return fig, axs

    def partial_plots(self, alpha=0.5, **kwargs):
        """Plot the partial regression plots for the model

        Arguments:
            alpha - A real value indicating the transparency of the residuals.
                Default is 0.5.
            kwargs - Named parameters that will be passed onto lower level
                matplotlib plotting functions.

        Returns:
            A tuple containing the matplotlib (figure, list of axes) for the
            partial plots.
        """
        #terms = self.ex.flatten(separate_interactions = False)
        terms = self.ex.get_terms()
        fig, axs = plt.subplots(1, len(terms), **kwargs)

        for i, ax in zip(range(0, len(terms)), axs):

            xi = terms[i]

            sans_xi = Combination(terms[:i] + terms[i+1:])
            yaxis = LinearModel(sans_xi, self.re)
            xaxis = LinearModel(sans_xi, xi)

            yaxis.fit(self.training_data)
            xaxis.fit(self.training_data)

            ax.scatter(xaxis.residuals_, yaxis.residuals_, alpha=alpha)
            ax.set_title("Leverage Plot for " + str(xi))

        return fig, axs

    @staticmethod
    def ones_column(data):
        """Helper function to create a column of ones for the intercept."""
        return pd.DataFrame({"Intercept" : np.repeat(1, data.shape[0])})

    def plot_residual_diagnostics(self, **kwargs):
        """Produce a matrix of four diagnostic plots:
        the residual v. quantile plot, the residual v. fited values plot,
        the histogram of residuals, and the residual v. order plot.

        Arguments:
            kwargs - Named parameters that will be passed onto lower level
                matplotlib plotting functions.

        Returns:
            A tuple containing the matplotlib (figure, list of axes) for the
            partial plots.
        """

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, **kwargs)
        self.residual_quantile_plot(ax=ax1)
        self.residual_fitted_plot(ax=ax2)
        self.residual_histogram(ax=ax3)
        self.residual_order_plot(ax=ax4)

        f.suptitle("Residal Diagnostic Plots for " + str(self))

        return f, (ax1, ax2, ax3, ax4)

    def residual_quantile_plot(self, ax=None):
        """Produces the residual v. quantile plot of the model.

        Arguments:
            ax - An optional parameter that is a pregenerated Axis object.

        Returns:
            A rendered matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        stats.probplot(self.residuals_, dist="norm", plot=ax)
        ax.set_title("Residual Q-Q Plot")
        return ax

    def residual_fitted_plot(self, ax=None):
        """Produces the residual v. fitted values plot of the model.

        Arguments:
            ax - An optional parameter that is a pregenerated Axis object.

        Returns:
            A rendered matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        ax.scatter(self.fitted_, self.residuals_)
        ax.set_title("Fitted Values v. Residuals")
        ax.set_xlabel("Fitted Value")
        ax.set_ylabel("Residual")

        return ax

    def residual_histogram(self, ax=None):
        """Produces the residual histogram of the model.

        Arguments:
            ax - An optional parameter that is a pregenerated Axis object.

        Returns:
            A rendered matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        ax.hist(self.residuals_)
        ax.set_title("Histogram of Residuals")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")

        return ax

    def residual_order_plot(self, ax=None):
        """Produces the residual v. order plot of the model.

        Arguments:
            ax - An optional parameter that is a pregenerated Axis object.

        Returns:
            A rendered matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        ax.plot(self.training_data.index, self.residuals_, "o-")
        ax.set_title("Order v. Residuals")
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Residual")

        return ax

