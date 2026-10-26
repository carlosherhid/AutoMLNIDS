"""
Table comparing the near-zero rescaling with alternative input preprocessings under the same
QAT protocol (CICIDS2017, 5 seeds): accuracy of the FP32 and QAT models, and recall, FPR and
F1-score of the QAT model, as mean +/- std over seeds, plus Welch t-tests of the QAT accuracy
and FPR of the proposed rescaling against each alternative.

Inputs (paper_results/CICIDS2017): multiseed_exp{3,6,50,51,52}_results.csv, and
detailed_metrics.csv for the runs (Exp. 3, 4) whose logs predate the recall/FPR/F1 logging.

Usage:
    python scripts/build_preprocessing_table.py --results_dir ./paper_results/CICIDS2017
"""
import argparse
import os

import pandas as pd
from scipy import stats

# label -> (multiseed base, paper experiment in detailed_metrics.csv or None if the log has the metrics)
VARIANTS = [
    ('Min-Max only (Exp. 3)', 3, 'Exp. 3'),
    ('Standardization', 50, None),
    ('Log Min-Max', 51, None),
    ('Quantile transform', 52, None),
    ('Near-zero rescaling (Exp. 4)', 6, 'Exp. 4'),
]
PROPOSED = 'Near-zero rescaling (Exp. 4)'


def load_variant(results_dir, base, paper_experiment, detailed):
    df = pd.read_csv(os.path.join(results_dir, f'multiseed_exp{base}_results.csv')).sort_values('seed').reset_index(drop=True)
    out = pd.DataFrame({'seed': df['seed'], 'fp32_acc': df['normal_test_accuracy'], 'qat_acc': df['qat_test_accuracy']})
    if paper_experiment is not None:  # metrics come from the checkpoint evaluation
        qat = detailed[(detailed.paper_experiment == paper_experiment) & (detailed.model == 'qat')].sort_values('seed')
        for metric in ('recall', 'fpr', 'f1'):
            out[f'qat_{metric}'] = qat[metric].values * 100
    else:
        for metric in ('recall', 'fpr', 'f1'):
            out[f'qat_{metric}'] = df[f'qat_{metric}'] * 100
    return out


def fmt(series):
    return f'{series.mean():.2f} $\\pm$ {series.std():.2f}'


def main():
    parser = argparse.ArgumentParser(description='Preprocessing comparison table')
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    detailed = pd.read_csv(os.path.join(args.results_dir, 'detailed_metrics.csv'))
    data = {label: load_variant(args.results_dir, base, exp, detailed) for label, base, exp in VARIANTS}

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
