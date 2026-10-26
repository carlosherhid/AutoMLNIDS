"""
Build the detection-metric tables of the paper (recall, FPR, F1 of the FP32 and QAT
models, mean +/- std over seeds) from the CSVs written by evaluate_checkpoint_metrics.py
Usage:
    python scripts/build_metrics_table.py --results_dir <results_dir>
Expects <results_dir>/<dataset>/detailed_metrics.csv and baselines_classical_ml_results.csv.
"""
import argparse
import os

import pandas as pd

DATASETS = ['CICIDS2017', 'CICIoT2023']
EXPERIMENT_ORDER = ['Exp. 1', 'Exp. 2', 'Exp. 3', 'Exp. 4', 'Exp. 5']
BASELINE_LABELS = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic Regr.', 'xgboost': 'XGBoost'}


def pct(mean, std):
    return f'{mean * 100:.2f} $\\pm$ {std * 100:.2f}'


def projected_precision(recall, fpr, prevalence):
    """Precision at a given share of malicious flows, from class-conditional rates."""
    true_positives = recall * prevalence
    return true_positives / (true_positives + fpr * (1 - prevalence))


def main():
    parser = argparse.ArgumentParser(description='Detection-metric tables for the paper')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--prevalences', type=float, nargs='+', default=[0.1, 0.01])
    args = parser.parse_args()

    print('% --- QAT/FP32 detection metrics (mean +/- std over seeds, %) ---')
    summaries = {}
    for dataset in DATASETS:
        df = pd.read_csv(os.path.join(args.results_dir, dataset, 'detailed_metrics.csv'))
        grouped = df.groupby(['paper_experiment', 'model'])[['accuracy', 'recall', 'fpr', 'f1']].agg(['mean', 'std'])
        summaries[dataset] = grouped
        for exp in EXPERIMENT_ORDER:
            cells = []
            for model in ('normal', 'qat'):
                row = grouped.loc[(exp, model)]
                cells += [pct(row[('recall', 'mean')], row[('recall', 'std')]),
                          pct(row[('fpr', 'mean')], row[('fpr', 'std')]),
                          pct(row[('f1', 'mean')], row[('f1', 'std')])]
            print(f'{dataset} & {exp.split()[-1]} & ' + ' & '.join(cells) + ' \\\\')
        print('\\hline')

    print('\n% --- precision projected to other attack prevalences (QAT, mean recall/FPR) ---')
    for dataset in DATASETS:
        for exp in EXPERIMENT_ORDER:
            row = summaries[dataset].loc[(exp, 'qat')]
            recall, fpr = row[('recall', 'mean')], row[('fpr', 'mean')]
            projected = ', '.join(f'{p:.0%}: {projected_precision(recall, fpr, p) * 100:.1f}%' for p in args.prevalences)
            print(f'{dataset} {exp}: recall={recall * 100:.2f}% FPR={fpr * 100:.2f}% -> precision at prevalence {projected}')

    print('\n% --- classical baselines: recall and FPR (derived) ---')
    for dataset in DATASETS:
        baselines = pd.read_csv(os.path.join(args.results_dir, dataset, 'baselines_classical_ml_results.csv'))
        for _, r in baselines.iterrows():
            fpr = 1 - 2 * r['accuracy'] + r['recall']
            print(f"{dataset} & {BASELINE_LABELS.get(r['model'], r['model'])} & {r['accuracy'] * 100:.2f} & "
                  f"{r['recall'] * 100:.2f} & {fpr * 100:.2f} & {r['f1']:.3f} \\\\")


if __name__ == '__main__':
    main()
