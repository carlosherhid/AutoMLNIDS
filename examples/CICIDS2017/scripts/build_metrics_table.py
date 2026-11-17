"""
Detection-metric tables of the paper, for any number of datasets: recall, FPR and F1 of the FP32 and INT8 (QAT) models of every experiment as
mean +/- std over the seeds, the precision they imply at other attack prevalences, and the classical baselines.

The metrics of every run are those of the multiseed_exp<N>_results.csv that run_multiseed.py writes (parsed from the execution logs; the INT8 numbers
use the engine recorded in the logs, onednn by default). The baselines are read from baselines/split_<seed>/ and, when their CSV has no FPR (results
written before the confusion counts were stored), the FPR is derived from the accuracy, the recall and the real share of malicious flows of that
split's test set (split_info.json), not from the assumption of a 50 % test set. See utils/results.py for the layout of NAME=ROOT[,MODELS_DIR].

Usage:
    python scripts/build_metrics_table.py \
        --dataset CICIDS2017_improved=<data>/CICIDS2017_improved/balanced_binary \
                  UNSW_NB15=<data>/UNSW_NB15/balanced_binary,compressed_models_bnfold \
                  TON_IoT=<data>/TON_IoT/balanced_binary \
        [--experiments 1 2 3 4 6 7 8] [--output_dir <dir for the CSV copies of the tables>]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.experiments import EXPERIMENTS  # noqa: E402
from utils.results import load_baselines, load_multiseed, parse_dataset  # noqa: E402

DEFAULT_EXPERIMENTS = [1, 2, 3, 4, 6, 7, 8]       # Experiment 5 (feature selection) is not reported in the paper
BASELINE_LABELS = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic Regr.', 'xgboost': 'XGBoost'}
METRICS = ('recall', 'fpr', 'f1')


def pct(mean, std):
    return f'{mean * 100:.2f} $\\pm$ {std * 100:.2f}'


def projected_precision(recall, fpr, prevalence):
    """Precision at a given share of malicious flows, from class-conditional rates."""
    true_positives = recall * prevalence
    return true_positives / (true_positives + fpr * (1 - prevalence))


def main():
    parser = argparse.ArgumentParser(description='Detection-metric tables for the paper')
    parser.add_argument('--dataset', type=str, nargs='+', required=True, metavar='NAME=ROOT[,MODELS_DIR]')
    parser.add_argument('--experiments', type=int, nargs='+', default=DEFAULT_EXPERIMENTS, choices=sorted(EXPERIMENTS))
    parser.add_argument('--prevalences', type=float, nargs='+', default=[0.1, 0.01])
    parser.add_argument('--output_dir', type=str, default=None, help='also write metrics_table.csv and baselines_table.csv here')
    args = parser.parse_args()
    try:
        datasets = [parse_dataset(spec) for spec in args.dataset]
    except ValueError as error:
        parser.error(str(error))

    summary_rows = []
    print('% --- FP32 and INT8 (QAT) detection metrics (mean +/- std over seeds, %): recall, FPR, F1 of the FP32 model, then of the INT8 model ---')
    for name, root, models_dir in datasets:
        for experiment in args.experiments:
            runs = load_multiseed(root, models_dir, experiment)
            if runs is None:
                continue
            cells = []
            for model in ('normal', 'qat'):
                for metric in METRICS:
                    values = runs[f'{model}_{metric}'].dropna()
                    mean, std = values.mean(), values.std()
                    cells.append(pct(mean, std) if len(values) else 'n/a')
                    summary_rows.append({'dataset': name, 'experiment': experiment, 'label': EXPERIMENTS[experiment][0],
                                         'model': 'fp32' if model == 'normal' else 'int8', 'metric': metric,
                                         'mean': mean, 'std': std, 'n_seeds': len(values)})
            print(f'{name} & {EXPERIMENTS[experiment][0]} & ' + ' & '.join(cells) + ' \\\\')
        print('\\hline')
    summary = pd.DataFrame(summary_rows)

    print('\n% --- precision of the INT8 model projected to other attack prevalences (from its mean recall and FPR) ---')
    for name, _, _ in datasets:
        for experiment in args.experiments:
            rows = summary[(summary.dataset == name) & (summary.experiment == experiment) & (summary.model == 'int8')]
            if rows.empty:
                continue
            recall = rows[rows.metric == 'recall'].iloc[0]['mean']
            fpr = rows[rows.metric == 'fpr'].iloc[0]['mean']
            projected = ', '.join(f'{p:.0%}: {projected_precision(recall, fpr, p) * 100:.1f}%' for p in args.prevalences)
            print(f'{name} {EXPERIMENTS[experiment][0]}: recall={recall * 100:.2f}% FPR={fpr * 100:.2f}% -> precision at prevalence {projected}')

    print('\n% --- classical baselines on the splits of the seeds (mean +/- std over splits, %): accuracy, recall, FPR, F1 ---')
    baseline_rows = []
    for name, root, _ in datasets:
        baselines = load_baselines(root)
        if baselines is None:
            print(f'WARNING: no baseline results under {root}/baselines')
            continue
        for model, group in baselines.groupby('model', sort=False):
            cells = [pct(group[c].mean(), group[c].std()) for c in ('accuracy', 'recall', 'fpr', 'f1')]
            print(f"{name} & {BASELINE_LABELS.get(model, model)} & " + ' & '.join(cells) + f' & n={len(group)} \\\\')
            for metric in ('accuracy', 'recall', 'fpr', 'f1'):
                baseline_rows.append({'dataset': name, 'model': model, 'metric': metric,
                                      'mean': group[metric].mean(), 'std': group[metric].std(), 'n_splits': len(group)})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary.to_csv(os.path.join(args.output_dir, 'metrics_table.csv'), index=False)
        pd.DataFrame(baseline_rows).to_csv(os.path.join(args.output_dir, 'baselines_table.csv'), index=False)
        print(f'\nCSV copies written to {args.output_dir}')


if __name__ == '__main__':
    main()
