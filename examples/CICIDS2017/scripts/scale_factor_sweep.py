"""
Sweep the near-zero feature rescaling factor. This does not derive the factor
mathematically, but turns "we picked 1e6" into an ablation that shows how accuracy
depends on it (a plateau between 1e4 and 1e7).

Runs scripts/autogluon_to_pytorch_qat.py once per factor as a subprocess (keeps this
script simple and reuses the exact same code path/logging as every other
experiment), then plots accuracy vs. factor.

Usage:
    python scripts/scale_factor_sweep.py --predictor_path <predictor_dir> \
        --data_dir <data_dir> --output_dir <output_dir> \
        --factors 1e2 1e3 1e4 1e5 1e6 1e7 1e8 --epochs 100 --experiment_number_base 600
"""
import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.log_parsing import parse_log_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Sweep --scale_factor and plot accuracy vs. factor')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--factors', type=float, nargs='+', default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_number_base', type=int, default=600,
                         help='Log/checkpoint files are written as exp<base+i> for the i-th factor')
    parser.add_argument('--extra_args', type=str, default='--disable_early_stopping',
                         help='Extra flags forwarded verbatim to autogluon_to_pytorch_qat.py')
    args = parser.parse_args()

    script_path = os.path.join(os.path.dirname(__file__), 'autogluon_to_pytorch_qat.py')
    rows = []

    for i, factor in enumerate(args.factors):
        experiment_number = args.experiment_number_base + i
        cmd = [
            sys.executable, script_path,
            '--predictor_path', args.predictor_path,
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--seed', str(args.seed),
            '--experiment_number', str(experiment_number),
            '--normal', '--qat',
            '--scale_close_to_zero', '--scale_factor', str(factor),
        ] + args.extra_args.split()

        print(f'\n=== Running factor={factor:g} (experiment_number={experiment_number}) ===')
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        log_path = os.path.join(args.output_dir, f'execution_log_exp{experiment_number}.txt')
        parsed = parse_log_file(log_path)
        parsed['scale_factor'] = factor
        parsed['experiment_number'] = experiment_number
        rows.append(parsed)

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'scale_factor_sweep_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'\nSaved sweep results to {csv_path}')
    print(results_df[['scale_factor', 'normal_test_accuracy', 'qat_test_accuracy']].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df['scale_factor'], results_df['normal_test_accuracy'], 'o-', label='Normal (FP32)')
    ax.plot(results_df['scale_factor'], results_df['qat_test_accuracy'], 's-', label='QAT (INT8)')
    ax.set_xscale('log')
    ax.set_xlabel('Near-zero rescaling factor')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy vs. near-zero feature rescaling factor')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, 'scale_factor_sweep.svg')
    fig.savefig(fig_path)
    print(f'Saved plot to {fig_path}')


if __name__ == '__main__':
    main()
