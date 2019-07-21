from .model import LinearModel
from .comparison import _extract_dfs
from .expression import Constant
import numpy as np
from abc import ABC


class Score(ABC):

    def __init__(self, model, order):
        self.order = order
        self.model = model

        if model is None:
            self._score = np.inf * (-1 if order else 1)
        else:
            self._score = self.compute()

    @abstractmethod
    def compute(self):
        pass

    def compare(self, other):
        ''' Return true if self is better than other based on 'order' '''
        assert(type(self) is type(other))  # make sure we are not comparing different types of scores
        if self.order:
            return self._score < other._score
        else:
            return self._score > other._score


class RSquared(Score):

    def __init__(self, model, adjusted=False):
        self.adjusted=adjusted

        super(RSquared, self).__init__(
            model=model,
            order=True
        )

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
            order=False
        )

    def compute(self):
        dfs = _extract_dfs(self.model, dict_out=True)
        sse = self.model.get_sse()
        return sse / dfs["error_df"]

# TODO: Implement Cp, AIC, BIC


_metrics = dict(
    r_squared=RSquared,
    r_squared_adjusted=lambda model: RSquared(model=model, adjusted=True),
    mse=MSE
)


def stepwise(full_model, metric_name, forward=False, naive=False):

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

    if naive:
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
            for i, term in enumerate(ex_term_list):
                try:
                    if forward:
                        potential_model = LinearModel(best_model.given_ex + term, re_term)
                    else:
                        potential_model = LinearModel(best_model.given_ex - term, re_term)

                    potential_model.fit(data)
                    potential_metric = metric_func(potential_model)

                    if best_potential_metric.compare(potential_metric):
                        best_potential_metric = potential_metric
                        best_potential_model = potential_model
                        best_idx = i

                except np.linalg.linalg.LinAlgError:
                    continue

            if best_metric.compare(best_potential_metric):
                best_metric = best_potential_metric
                best_model = best_potential_model
                del ex_term_list[best_idx]
            else:
                break
    else:
        raise NotImplementedError("Non-naive stepwise model building is not implemented yet.")

    return dict(
        forward=forward,
        metric=best_metric,
        metric_name=metric_name,
        best_model=best_model
    )