"""
Explain why the AutoGluon network receives more inputs (67 for CICIDS2017) than the
dataset has features (61): print how AutoGluon's own preprocessing of the best neural
network turns the dataset columns into network inputs. The PyTorch replica takes the
dataset features directly, so this difference has to be documented in the paper.

Internal attribute names change between AutoGluon versions, so every lookup is
defensive and the script also lists the candidate attributes it could not interpret.

Usage:
    python scripts/inspect_automl_inputs.py --predictor_path ./balanced_binary/automl_search \
        --data_dir ./balanced_binary
"""
import argparse
import os
import sys

import pandas as pd
from autogluon.tabular import TabularPredictor

sys.path.insert(0, os.path.dirname(__file__))
from utils.common import patch_stale_predictor  # noqa: E402


def first_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return name, getattr(obj, name)
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Inspect the inputs of the AutoGluon best neural network')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    patch_stale_predictor(predictor)
    model = predictor._trainer.load_model(predictor._trainer.model_best)

    X = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), index_col='ID', nrows=2000).drop(columns=['Label'])
    print(f'Dataset features (test.csv minus ID and Label): {X.shape[1]}')

    try:
        X_generated = predictor.transform_features(X)
        print(f'After AutoGluon feature generation: {X_generated.shape[1]} columns')
        extra = sorted(set(X_generated.columns) - set(X.columns))
        missing = sorted(set(X.columns) - set(X_generated.columns))
        print(f'  columns added by feature generation: {extra}')
        print(f'  columns removed by feature generation: {missing}')
    except Exception as e:  # noqa: BLE001
        print(f'transform_features failed: {e!r}')
        X_generated = X

    name, types = first_attr(model, ['_types_of_features', 'types_of_features'])
    if isinstance(types, dict):
        print(f'\nNetwork feature groups ({name}):')
        for group, columns in types.items():
            print(f'  {group}: {len(columns)} columns')
    else:
        print('\nNo feature-group dictionary found on the model.')

    name, processor = first_attr(model, ['processor', '_processor'])
    if processor is not None:
        print(f'\nPreprocessor ({name}): {type(processor).__name__}')
        for step_name, step, columns in getattr(processor, 'transformers_', []):
            print(f'  {step_name}: {type(step).__name__} on {len(columns)} columns')
        try:
            width = processor.transform(X_generated[model.features] if hasattr(model, 'features') else X_generated).shape[1]
            print(f'  output width of the preprocessor: {width}')
        except Exception as e:  # noqa: BLE001
            print(f'  could not run the preprocessor directly: {e!r}')

    print('\nCandidate attributes on the model (for debugging):')
    print('  ', sorted(a for a in dir(model) if any(k in a.lower() for k in ('feature', 'processor', 'type', 'embed'))
                       and not a.startswith('__')))


if __name__ == '__main__':
    main()
