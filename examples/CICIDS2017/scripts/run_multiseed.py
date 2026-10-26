"""
Repeat one autogluon_to_pytorch_qat.py configuration across several seeds and report
mean +/- std, so that results are not tied to a single run. Used for the 5
experiments the paper reports (internal --experiment_number 1, 2, 3, 6, 7 - see
docs/EXPERIMENTS_LOG.md).

Usage:
    python scripts/run_multiseed.py --predictor_path <predictor_dir> \
        --data_dir <data_dir> --output_dir <output_dir> \
        --experiment_number_base 6 --seeds 0 1 2 3 4 \
        --extra_args "--normal --qat --disable_early_stopping --scale_close_to_zero"
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.log_parsing import parse_log_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Run one experiment config across multiple seeds and aggregate')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--experiment_number_base', type=int, required=True,
                         help='Which paper experiment this reruns, e.g. 6 for paper Experiment 4 (see docs/EXPERIMENTS_LOG.md)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--extra_args', type=str, required=True,
                         help='The experiment-defining flags, e.g. "--normal --qat --disable_early_stopping --scale_close_to_zero"')
    args = parser.parse_args()

    script_path = os.path.join(os.path.dirname(__file__), 'autogluon_to_pytorch_qat.py')
    rows = []

    for seed in args.seeds:
        # 100000 + base*100 + seed keeps multi-seed reruns out of the way of both
        # the original 1-58 experiment numbering and scale_factor_sweep.py's 600+.
        experiment_number = 100000 + args.experiment_number_base * 100 + seed
        cmd = [
            sys.executable, script_path,
            '--predictor_path', args.predictor_path,
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--seed', str(seed),
            '--experiment_number', str(experiment_number),
        ] + args.extra_args.split()

        print(f'\n=== Seed {seed} (experiment_number={experiment_number}) ===')
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        log_path = os.path.join(args.output_dir, f'execution_log_exp{experiment_number}.txt')
        parsed = parse_log_file(log_path)
        parsed['seed'] = seed
        parsed['experiment_number'] = experiment_number
        rows.append(parsed)

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f'multiseed_exp{args.experiment_number_base}_results.csv')
    results_df.to_csv(csv_path, index=False)

    summary = results_df[['normal_test_accuracy', 'qat_test_accuracy', 'qat_size_kb']].agg(['mean', 'std'])
    print(f'\nSaved per-seed results to {csv_path}')
    print('\n=== Summary (mean +/- std across seeds) ===')
    print(summary.to_string())
    print(
        f"\nNormal accuracy: {summary.loc['mean', 'normal_test_accuracy']:.2f}"
        f" +/- {summary.loc['std', 'normal_test_accuracy']:.2f}%"
    )
    print(
        f"QAT accuracy:    {summary.loc['mean', 'qat_test_accuracy']:.2f}"
        f" +/- {summary.loc['std', 'qat_test_accuracy']:.2f}%"
    )

    detailed_columns = [f'{model}_{metric}' for model in ('normal', 'qat') for metric in ('precision', 'recall', 'fpr', 'f1')]
    detailed = results_df[detailed_columns].dropna(axis=1, how='all')  # absent when the logs predate the metrics
    if not detailed.empty:
        print('\n=== Precision / recall / FPR / F1 (mean +/- std across seeds) ===')
        print(detailed.agg(['mean', 'std']).to_string(float_format=lambda v: f'{v:.5f}'))


if __name__ == '__main__':
    main()
