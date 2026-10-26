"""
Recover what an AutoGluon search actually used, from the saved predictor alone, for the
paper's description of the AutoML stage: how many neural-network trials were trained,
how long the fit took, and the epoch / patience settings stored in the best model.

Internal attribute names change between AutoGluon versions, so every lookup is
defensive; anything that cannot be read is reported rather than guessed.

Usage:
    python scripts/inspect_automl_budget.py --predictor_path ./balanced_binary/automl_search
"""
import argparse
import datetime
import os
import sys

from autogluon.tabular import TabularPredictor

sys.path.insert(0, os.path.dirname(__file__))
from utils.common import patch_stale_predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Inspect the budget used by a saved AutoGluon search')
    parser.add_argument('--predictor_path', type=str, required=True)
    args = parser.parse_args()

    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    patch_stale_predictor(predictor)

    print(f'predictor path: {args.predictor_path}')
    try:
        leaderboard = predictor.leaderboard(silent=True)
        print(f'models in the leaderboard (trained trials): {len(leaderboard)}')
        print(f"best model: {predictor.model_best} | validation accuracy {leaderboard['score_val'].max():.5f}")
        print(f"fit time per model (s): min {leaderboard['fit_time'].min():.0f} | median {leaderboard['fit_time'].median():.0f} "
              f"| max {leaderboard['fit_time'].max():.0f} | sum {leaderboard['fit_time'].sum():.0f}")
    except Exception as e:  # noqa: BLE001
        print(f'leaderboard failed: {e!r}')

    try:
        info = predictor.info()
        for key, value in info.items():
            if any(token in key.lower() for token in ('time', 'num_models', 'tune', 'hyperparameter')):
                print(f'info[{key!r}] = {value!r}'[:300])
        best = info.get('model_info', {}).get(predictor.model_best, {})
        params = best.get('hyperparameters', {})
        for key in ('num_epochs', 'epochs_wo_improve', 'batch_size', 'learning_rate', 'activation', 'dropout_prob',
                    'num_layers', 'hidden_size', 'use_batchnorm', 'optimizer'):
            if key in params:
                print(f'best model hyperparameter {key} = {params[key]}')
        print('best model hyperparameters (all):', params)
    except Exception as e:  # noqa: BLE001
        print(f'predictor.info() failed: {e!r}')

    # predictor.info() fails on predictors saved by older AutoGluon versions: read the best model directly
    try:
        model = predictor._trainer.load_model(predictor.model_best)
        params = getattr(model, 'params', None) or {}
        print(f'best model {model.name}: fit_time={getattr(model, "fit_time", None)} s | val_score={getattr(model, "val_score", None)}')
        for key in ('num_epochs', 'epochs_wo_improve', 'batch_size', 'learning_rate', 'activation', 'dropout_prob',
                    'num_layers', 'hidden_size', 'use_batchnorm', 'optimizer'):
            if key in params:
                print(f'best model params[{key}] = {params[key]}')
    except Exception as e:  # noqa: BLE001
        print(f'could not read the best model directly: {e!r}')

    trainer = predictor._trainer
    start = getattr(trainer, '_time_train_start', None) or getattr(trainer, '_time_train_start_last', None)
    if start:
        print('search started (UTC):', datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M'))
    for key, value in vars(trainer).items():
        if any(token in key.lower() for token in ('time', 'tune', 'hpo', 'trial')):
            print(f'trainer.{key} = {str(value)[:200]}')


if __name__ == '__main__':
    main()
