from .model import LinearModel
from .comparison import _extract_dfs
from .expression import Constant
import numpy as np

def _r_squared(model):
    return model.r_squared(adjusted=False)


def _r_squared_adjusted(model):
    return model.r_squared(adjusted=True)


def _mse(model):
    dfs = _extract_dfs(model, dict_out=True)
    sse = model.get_sse()
    return sse / dfs["error_df"]

# TODO: Implement Cp, AIC, BIC


_metrics = dict(
    r_squared=_r_squared,
    r_squared_adjusted=_r_squared_adjusted,
    mse=_mse
)

_order_highest = dict(
    r_squared=True,
    r_squared_adjusted=True,
    mse=False
)


def _compare(old_metric, new_metric, order):
    ''' Return true if new_metric is better than old_metric based on 'order' '''
    if order:
        return old_metric < new_metric
    else:
        return old_metric > new_metric


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
    order = _order_highest[metric_name]

    if naive:
        ex_term_list = ex_terms.get_terms()
        if forward:
            best_model = LinearModel(Constant(1), re_term)
            best_model.fit(data)
        else:
            best_model = full_model

        best_metric = metric_func(best_model)

        while len(ex_term_list) > 0:
            potential_models = []
            for i, term in enumerate(ex_term_list):
                try:
                    if forward:
                        potential_model = LinearModel(best_model.given_ex + term, re_term)
                    else:
                        potential_model = LinearModel(best_model.given_ex - term, re_term)

                    potential_model.fit(data)
                    potential_models.append((metric_func(potential_model), potential_model, i))
                except np.linalg.linalg.LinAlgError:
                    potential_models.append((float('inf') * (-1 if order else 1), None, i))

            best_potential_metric, best_potential_model, best_idx = sorted(
                potential_models,
                key=lambda x: x[0],
                reverse=order
            )[0]

            if _compare(best_metric, best_potential_metric, order):
                best_metric = best_potential_metric
                best_model = best_potential_model
                del ex_term_list[best_idx]
            else:
                break
    else:
        raise NotImplementedError("Non-naive forward stepwise model building is not implemented yet.")

    return dict(
        forward=forward,
        metric=best_metric,
        metric_name=metric_name,
        best_model=best_model
    )