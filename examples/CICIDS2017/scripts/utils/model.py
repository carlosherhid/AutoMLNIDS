import ast
import time
from typing import List

import pandas as pd
import torch
from autogluon.tabular.predictor import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel


def test_model_on_subset(predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series,
                         model: TabularNeuralNetTorchModel, subset: List[str] = None):
    if isinstance(model, (list, tuple)):
        model, subset = model

    if subset is not None:
        if isinstance(subset, str):
            subset = ast.literal_eval(subset)
        X = X.copy(deep=True)
        X[X.columns.difference(list(subset))] = 0.0

    start = time.time()
    ypred = model.predict(X)
    end = time.time()

    if subset is not None:
        del X
    scores = predictor.evaluate_predictions(y, pd.Series(ypred), auxiliary_metrics=True, silent=True)
    scores['time'] = end - start
    return scores


def get_prunable(model: torch.nn.Module):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


def network_layers_sparsity(refs: torch.nn.Module):
    single = {}
    total_weight, total_bias = 0, 0
    for x in refs:
        local = (float(torch.sum(x.weight == 0) / x.weight.nelement()),
                 float(torch.sum(x.bias == 0) / x.bias.nelement()))
        single[x] = local
        total_weight += local[0]
        total_bias += local[1]
    total_weight /= len(single)
    total_bias /= len(single)
    return (total_weight, total_bias), single
