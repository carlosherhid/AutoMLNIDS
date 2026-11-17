"""
What the AutoML search chose for one dataset: the best model, its hyperparameters, the exact layer sequence (including where the BatchNorm layers
are) and how many trials were kept. Run it after the search and before Experiments 11 and 12: they only differ from Experiments 3 and 4 if the
chosen network has BatchNorm layers.

Usage:
    python scripts/describe_automl_model.py --predictor_path <data>/balanced_binary/split_0/automl_search
Writes <predictor_path>/automl_model_summary.json next to the predictor.
"""
import argparse
import json
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from autogluon_to_pytorch_qat import get_model_architecture  # noqa: E402
from utils.common import load_predictor  # noqa: E402

HYPERPARAMETERS = ('num_layers', 'hidden_size', 'dropout_prob', 'activation', 'use_batchnorm', 'learning_rate', 'weight_decay',
                   'batch_size', 'optimizer', 'proc.embed_min_categories', 'proc.skew_threshold')


def describe_layers(architecture):
    """One entry per layer of the AutoGluon network, in order, with the shape information that matters."""
    layers = []
    for layer_type, layer in architecture:
        entry = {'type': layer_type.__name__}
        if isinstance(layer, nn.Linear):
            entry.update(inputs=layer.in_features, outputs=layer.out_features)
        elif isinstance(layer, nn.BatchNorm1d):
            entry.update(features=layer.num_features)
        elif isinstance(layer, nn.Dropout):
            entry.update(p=round(layer.p, 4))
        layers.append(entry)
    return layers


def summarize(architecture, best_model, model_name, n_trial_folders, n_features=None):
    layers = describe_layers(architecture)
    network = getattr(best_model, 'model', None)
    parameters = getattr(best_model, 'params', None) or {}
    kinds = [layer['type'] for layer in layers]
    widths = [layer.get('inputs', layer.get('features')) for layer in layers if layer['type'] in ('BatchNorm1d', 'Linear')]
    return {
        'best_model': model_name,
        'trial_folders_saved': n_trial_folders,
        'predictor_features': n_features,
        'network_input_width': widths[0] if widths else None,
        'hyperparameters': {k: parameters[k] for k in HYPERPARAMETERS if k in parameters},
        'n_linear': kinds.count('Linear'),
        'n_batchnorm': kinds.count('BatchNorm1d'),
        'n_dropout': kinds.count('Dropout'),
        'batchnorm_before_next_linear': all(
            next((j for j in range(i + 1, len(kinds)) if kinds[j] == 'Linear'), None) is not None
            and all(kind in ('Dropout', 'Identity') for kind in kinds[i + 1:next(j for j in range(i + 1, len(kinds)) if kinds[j] == 'Linear')])
            for i, kind in enumerate(kinds) if kind == 'BatchNorm1d'),
        'first_layers': kinds[:6],
        'n_parameters': sum(p.numel() for p in network.parameters()) if network is not None else None,
        'layers': layers,
    }


def main():
    parser = argparse.ArgumentParser(description='Describe the model chosen by the AutoML search')
    parser.add_argument('--predictor_path', type=str, required=True)
    args = parser.parse_args()

    predictor = load_predictor(args.predictor_path)
    architecture, best_model, _ = get_model_architecture(predictor, 0)
    trial_root = os.path.join(args.predictor_path, 'models', 'NeuralNetTorch')
    n_trials = len([d for d in os.listdir(trial_root) if os.path.isdir(os.path.join(trial_root, d))]) if os.path.isdir(trial_root) else None
    summary = summarize(architecture, best_model, predictor._trainer.model_best, n_trials, n_features=len(predictor.features()))

    with open(os.path.join(args.predictor_path, 'automl_model_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    hp = summary['hyperparameters']
    print(f"Best model: {summary['best_model']} ({summary['trial_folders_saved']} trial folders saved)")
    print('Hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hp.items()))
    print(f"Layers: {summary['n_linear']} Linear, {summary['n_batchnorm']} BatchNorm1d, {summary['n_dropout']} Dropout; "
          f"{summary['n_parameters']} parameters")
    print('First layers: ' + ' -> '.join(summary['first_layers']))
    print(f"Input width of the AutoGluon network: {summary['network_input_width']}; columns of the data used by the predictor: {summary['predictor_features']}"
          + ('' if summary['network_input_width'] == summary['predictor_features']
             else ' (different: AutoGluon turns some columns into several inputs; the replica of the paper is built on the columns)'))
    if summary['n_batchnorm']:
        print(f"BatchNorm layers: {summary['n_batchnorm']}, each followed (through Dropout only) by a Linear: {summary['batchnorm_before_next_linear']} "
              '-> Experiments 11 and 12 fold them')
    else:
        print('No BatchNorm layers: Experiments 11 and 12 would not fold anything (they would only start the QAT from the FP32 weights)')


if __name__ == '__main__':
    main()
