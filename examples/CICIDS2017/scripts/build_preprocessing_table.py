"""
Table comparing the near-zero rescaling with alternative input preprocessings under the same
QAT protocol (5 seeds, one data split per seed): accuracy of the FP32 and QAT models, and recall, FPR and
F1-score of the QAT model, as mean +/- std over seeds, plus Welch t-tests of the QAT accuracy
and FPR of the proposed rescaling against each alternative.

Usage:
    python scripts/build_preprocessing_table.py --results_dir <compressed_models dir with the multiseed_exp*_results.csv>
"""
import argparse
import os

import pandas as pd
from scipy import stats

# label -> experiment number (utils/experiments.py); the metrics of every run are in its multiseed CSV
VARIANTS = [
    ('Min-Max only (Exp. 3)', 3),
    ('Standardization', 6),
    ('Log Min-Max', 7),
    ('Quantile transform', 8),
    ('Near-zero rescaling (Exp. 4)', 4),
]
PROPOSED = 'Near-zero rescaling (Exp. 4)'


def load_variant(results_dir, base):
    df = pd.read_csv(os.path.join(results_dir, f'multiseed_exp{base}_results.csv')).sort_values('seed').reset_index(drop=True)
    out = pd.DataFrame({'seed': df['seed'], 'fp32_acc': df['normal_test_accuracy'], 'qat_acc': df['qat_test_accuracy']})
    for metric in ('recall', 'fpr', 'f1'):
        out[f'qat_{metric}'] = df[f'qat_{metric}'] * 100
    return out


def fmt(series):
    return f'{series.mean():.2f} $\\pm$ {series.std():.2f}'


def main():
    parser = argparse.ArgumentParser(description='Preprocessing comparison table')
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    data = {label: load_variant(args.results_dir, base) for label, base in VARIANTS}

    print('% label & FP32 acc & QAT acc & QAT recall & QAT FPR & QAT F1')
    for label, df in data.items():
        print(f"{label} & {fmt(df['fp32_acc'])} & {fmt(df['qat_acc'])} & {fmt(df['qat_recall'])} & "
              f"{fmt(df['qat_fpr'])} & {fmt(df['qat_f1'])} \\\\")

    print('\n% Welch t-tests of the proposed rescaling against each alternative (n = 5 seeds each)')
    for label, df in data.items():
        if label == PROPOSED:
            continue
        for column in ('fp32_acc', 'qat_acc', 'qat_fpr'):
            t, p = stats.ttest_ind(data[PROPOSED][column], df[column], equal_var=False)
            diff = data[PROPOSED][column].mean() - df[column].mean()
            print(f'{label:28s} {column:8s} rescaling - alternative = {diff:+.2f}  Welch p = {p:.3f}')


if __name__ == '__main__':
    main()
