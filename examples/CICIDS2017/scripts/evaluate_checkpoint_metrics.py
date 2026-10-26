"""
Precision / recall (TPR) / FPR / F1 for the multi-seed checkpoints
Usage:
    python scripts/evaluate_checkpoint_metrics.py --predictor_path <predictor_dir> \
        --data_dir <data_dir> --logs_dir <logs_dir> \
        --important_features_file <features_file> \
        --output_csv <output_csv>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from autogluon_to_pytorch_qat import evaluate_model_detailed, get_model_architecture  # noqa: E402
from build_tradeoff_table import try_load_checkpoint  # noqa: E402

# internal --experiment_number_base -> (paper experiment, rescale near-zero, select features)
EXPERIMENTS = {1: ('Exp. 1', False, False), 2: ('Exp. 2', False, False), 3: ('Exp. 3', False, False),
               6: ('Exp. 4', True, False), 7: ('Exp. 5', True, True)}


def load_test_features(data_dir, rescale, selected_features_file, scale_factor):
    """Same inputs as autogluon_to_pytorch_qat.py's load_data()/prune_features(), in memory
    (load_data() also writes *_scaled.csv files into data_dir, which is unnecessary here)."""
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    labels = LabelEncoder().fit_transform(test['Label'])
    features = test.drop(columns=['ID', 'Label'])
    if rescale:
        features = features.where(features.abs() >= 1e-6, features * scale_factor)
    if selected_features_file:
        keep = pd.read_csv(selected_features_file, header=None).squeeze().tolist()
        features = features.copy()
        features[[c for c in features.columns if c not in keep]] = 0
    return features.values.astype(np.float32), labels


def main():
    parser = argparse.ArgumentParser(description='Detailed metrics for the multi-seed checkpoints')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True, help='compressed_models/ with the checkpoints and multiseed CSVs')
    parser.add_argument('--important_features_file', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--scale_factor', type=float, default=1e6)
    parser.add_argument('--engine', type=str, default=None,
                        help='torch.backends.quantized.engine used to convert and run the QAT models '
                             '(default: PyTorch default; training/logging scripts never set it)')
    args = parser.parse_args()

    if args.engine:
        supported = torch.backends.quantized.supported_engines
        if args.engine not in supported:
            raise SystemExit(f'engine {args.engine!r} is not supported here; supported engines: {supported}')
        torch.backends.quantized.engine = args.engine
    engine = torch.backends.quantized.engine
    print(f'quantized engine: {engine} (supported: {torch.backends.quantized.supported_engines})')

    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    n_features = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), nrows=1).shape[1] - 2  # minus ID, Label
    architecture, _, input_feature_size = get_model_architecture(predictor, n_features)

    rows, mismatches = [], 0
    for base, (paper_name, rescale, select) in EXPERIMENTS.items():
        X, y = load_test_features(args.data_dir, rescale, args.important_features_file if select else None, args.scale_factor)
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)), batch_size=8192)
        logged = pd.read_csv(os.path.join(args.logs_dir, f'multiseed_exp{base}_results.csv')).set_index('seed')

        for seed in args.seeds:
            experiment_number = 100000 + base * 100 + seed
            for kind in ('normal', 'qat'):
                path = os.path.join(args.logs_dir, f'{kind}_pytorch_model_exp{experiment_number}.pth')
                model = try_load_checkpoint(path, kind, architecture, input_feature_size)
                metrics = evaluate_model_detailed(model, loader, torch.device('cpu'))

                expected = logged.loc[seed, f'{kind}_test_accuracy']
                if abs(metrics['accuracy'] * 100 - expected) > 0.011:
                    mismatches += 1
                    if mismatches <= 5:  # a different engine is expected to differ from the logs for QAT models
                        print(f'WARNING: {paper_name} seed {seed} {kind}: recomputed accuracy '
                              f'{metrics["accuracy"] * 100:.2f}% != logged {expected:.2f}%')
                rows.append({'experiment_base': base, 'paper_experiment': paper_name, 'seed': seed,
                             'model': kind, 'engine': engine, 'logged_accuracy': expected / 100, **metrics})
            print(f'{paper_name} seed {seed}: done')

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f'\nSaved {len(df)} rows to {args.output_csv} ({mismatches} accuracy mismatches vs. the logs)')
    summary = df.groupby(['paper_experiment', 'model'])[['accuracy', 'logged_accuracy', 'recall', 'fpr', 'f1']].agg(['mean', 'std'])
    print(summary.to_string(float_format=lambda v: f'{v:.5f}'))


if __name__ == '__main__':
    main()
