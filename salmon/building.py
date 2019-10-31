import numpy as np
import math

from abc import ABC, abstractmethod

from .model import LinearModel
from .comparison import _extract_dfs
from .expression import Constant


class Score(ABC):

    def __init__(self, model, higher_is_better):
        self.higher_is_better = higher_is_better
        self.model = model

        if model is None:
            self._score = np.inf * (-1 if higher_is_better else 1)
        else:
            self._score = self.compute()

    @abstractmethod
    def compute(self):
        pass

    def __str__(self):
        return "{} | {}".format(type(self).__name__, self._score)

    def compare(self, other):
        ''' Return true if self is better than other based on 'higher_is_better' '''
        assert(type(self) is type(other))  # make sure we are not comparing different types of scores
        if self.higher_is_better:
            return self._score < other._score
        else:
            return self._score > other._score


class RSquared(Score):

    def __init__(self, model, adjusted=False):
        self.adjusted=adjusted

        super(RSquared, self).__init__(
            model=model,
            higher_is_better=True
        )

    def __str__(self):
        return "R^2 ({}adjusted) | {}".format("" if self.adjusted else "un", self._score)

    def compute(self):
        ''' Calculate the (adjusted) R^2 value of the model.

        Arguments:
            X - An optional DataFrame of the explanatory data to be used for calculating R^2. Default is the training data.
            Y - An optional DataFrame of the response data to be used for calculating R^2. Default is the training data.
            adjusted - A boolean indicating if the R^2 value is adjusted (True) or not (False).

        Returns:
            A real value of the computed R^2 value.
        '''

        X = self.model.training_data
        y = self.model.training_y

        pred = self.model.predict(X)
        sse = ((y.iloc[:, 0] - pred.iloc[:, 0]) ** 2).sum()
        ssto = ((y.iloc[:, 0] - y.iloc[:, 0].mean()) ** 2).sum()

        if self.adjusted:
            numerator = sse
            denominator = ssto
        else:
            numerator = sse / (len(y) - len(self.model.training_x.columns) - 2)
            denominator = ssto / (len(y) - 1)

        return 1 - numerator / denominator


class MSE(Score):

    def __init__(self, model):
        super(MSE, self).__init__(
            model=model,
            higher_is_better=False
        )

    def compute(self):
        dfs = _extract_dfs(self.model, dict_out=True)
        sse = self.model.get_sse()
        return sse / dfs["error_df"]


class MallowsCp(Score):

    def __init__(self, model):
        super(MallowsCp, self).__init__(
            model=model,
            higher_is_better=False,
        )

    def compute(self):
        dfs = _extract_dfs(self.model, dict_out=True)
        sse = self.model.get_sse()
        sigma_sq = self.model.std_err_est ** 2
        n, p = self.model.n, self.model.p

        return sse / sigma_sq - n + (2 * p)


class AIC(Score):

    def __init__(self, model):
        super(AIC, self).__init__(
            model=model,
            higher_is_better=False,
        )

    def compute(self):
        p = self.model.p
        log_likelihood = self.model.log_likelihood()

        return 2 * (p - log_likelihood)


class BIC(Score):

    def __init__(self, model):
        super(BIC, self).__init__(
            model=model,
            higher_is_better=False,
        )

    def compute(self):
        n, p = self.model.n, self.model.p
        log_likelihood = self.model.log_likelihood()

        return math.log(n) * p - 2 * log_likelihood


_metrics = dict(
    r_squared=RSquared,
    r_squared_adjusted=lambda model: RSquared(model=model, adjusted=True),
    mse=MSE,
    cp=MallowsCp,
    aic=AIC,
    bic=BIC,
)


def stepwise(full_model, metric_name, forward=False, naive=False, data=None, verbose=False):

    if data is not None:
        full_model.fit(data)

    metric_name = metric_name.lower()

    ex_terms = full_model.ex
    re_term = full_model.re
    data = full_model.training_data

    if ex_terms is None or re_term is None:
        raise AssertionError("The full model must be fit prior to undergoing a stepwise procedure.")

    if metric_name not in _metrics:
        raise KeyError("Metric '{}' not supported. The following metrics are supported: {}".format(
            metric_name,
            list(_metrics.keys())
        ))

    metric_func = _metrics[metric_name]

    ex_term_list = ex_terms.get_terms()
    if forward:
        best_model = LinearModel(Constant(1), re_term)
        best_model.fit(data)
    else:
        best_model = full_model

    best_metric = metric_func(best_model)
      

    while len(ex_term_list) > 0:
        best_potential_metric = metric_func(None)
        best_potential_model = None
        best_idx = None
    
        if forward and not naive:
            ex_term_list_expression = None
            for t in ex_term_list:
                if ex_term_list_expression is None:
                    ex_term_list_expression = t
                else:
                    ex_term_list_expression = ex_term_list_expression + t
            leaves = set(term for term in ex_term_list if not term.contains(ex_term_list_expression - term)) # Find all terms that do not depend on other terms
    
        for i, term in enumerate(ex_term_list):
            try:
                if forward:
                    # validate if adding term is valid
                    if not naive:
                        if term not in leaves:
                            continue
                    potential_model = LinearModel(best_model.given_ex + term, re_term)
                else:
                    # validate if removing term is valid
                    if not naive:
                        if (best_model.given_ex - term).contains(term):
                            continue
                    potential_model = LinearModel(best_model.given_ex - term, re_term)

                potential_model.fit(data)
                potential_metric = metric_func(potential_model)

                if best_potential_metric.compare(potential_metric):
                    best_potential_metric = potential_metric
                    best_potential_model = potential_model
                    best_idx = i

                if verbose:
                    print(potential_model)
                    print(potential_metric)
                    print("Current best potential model" if best_idx == i else "Not current best potential")
                    print()

            except np.linalg.linalg.LinAlgError:
                continue

        if best_metric.compare(best_potential_metric):
            best_metric = best_potential_metric
            best_model = best_potential_model
            if verbose:
                print("!!! New model found. Now including", ex_term_list[best_idx])
                print()
            del ex_term_list[best_idx]
        else:
            if verbose:
                print("!!! No potential models better than prior. Exiting search.")
                print()
            break
    else:
        if verbose:
            print("!!! Exhausted all potential terms. None left to consider.")


    return dict(
        forward=forward,
        metric=best_metric,
        metric_name=metric_name,
        best_model=best_model
    )