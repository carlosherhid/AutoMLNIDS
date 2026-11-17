"""
Repeat one autogluon_to_pytorch_qat.py configuration across several seeds and report
mean +/- std, so that results are not tied to a single run. The experiments of the paper
are defined in utils/experiments.py (1-5: Table 1; 6-8: alternative input representations)
and selected with --experiment.

Usage (one split per seed, seed i on split_i):
    python scripts/run_multiseed.py --experiment 4 --predictor_path <predictor_dir> \
        --data_dir_template "<data>/balanced_binary/split_{seed}" --output_dir <output_dir> \
        --seeds 0 1 2 3 4
    # Experiment 5 also needs the list written by select_features.py:
    python scripts/run_multiseed.py --experiment 5 --important_features_file <features_file> ...
    # A custom configuration instead of a numbered experiment:
    python scripts/run_multiseed.py --experiment_number_base 99 --extra_args "--normal --qat" ...
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.experiments import EXPERIMENTS, experiment_flags  # noqa: E402
from utils.experiments import experiment_number as experiment_number_of  # noqa: E402
from utils.log_parsing import parse_log_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Run one experiment config across multiple seeds and aggregate')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=None, help='one split used by every seed')
    parser.add_argument('--data_dir_template', type=str, default=None,
                        help='one split per seed instead, e.g. <data>/balanced_binary/split_{seed} (written by dataset_creator.py --seeds). '
                             'The AutoGluon predictor is evaluated only on the split it was trained on (--predictor_split_seed).')
    parser.add_argument('--predictor_split_seed', type=int, default=0,
                        help='seed of the split whose training data the AutoGluon predictor was trained on')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--experiment', type=int, choices=sorted(EXPERIMENTS), default=None,
                        help='numbered experiment of utils/experiments.py (flags and run numbers are derived from it)')
    parser.add_argument('--important_features_file', type=str, default=None,
                        help='features kept by Experiment 5 (written by select_features.py)')
    parser.add_argument('--experiment_number_base', type=int, default=None,
                         help='custom configuration: number of the run series (runs are named 100000 + base*100 + seed)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--extra_args', type=str, default=None,
                         help='custom configuration: the experiment-defining flags, e.g. "--normal --qat --disable_early_stopping"')
    parser.add_argument('--keep_batchnorm', action='store_true',
                        help='run the experiment with the BatchNorm layers of the AutoML network kept during training and folded into the next Linear '
                             'before the quantization (autogluon_to_pytorch_qat.py --keep_batchnorm). For models whose BatchNorm-free replica does not '
                             'train; write the runs to their own --output_dir (cluster/submit_experiments.sh --keep-batchnorm does)')
    args = parser.parse_args()
    if (args.data_dir is None) == (args.data_dir_template is None):
        parser.error('give exactly one of --data_dir and --data_dir_template')
    if args.experiment is not None:
        if args.experiment_number_base is not None or args.extra_args is not None:
            parser.error('--experiment replaces --experiment_number_base and --extra_args')
        try:
            args.extra_args = experiment_flags(args.experiment, args.important_features_file)
        except ValueError as error:
            parser.error(str(error))
        args.experiment_number_base = args.experiment
    elif args.experiment_number_base is None or args.extra_args is None:
        parser.error('give --experiment, or both --experiment_number_base and --extra_args')
    if args.keep_batchnorm and '--keep_batchnorm' not in args.extra_args.split():
        args.extra_args += ' --keep_batchnorm'
    print(f'Experiment {args.experiment_number_base}: {args.extra_args}')

    os.makedirs(args.output_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), 'autogluon_to_pytorch_qat.py')
    rows = []

    for seed in args.seeds:
        # 100000 + base*100 + seed keeps multi-seed runs out of the way of the single
        # runs (--experiment_number 1-99) and of scale_factor_sweep.py's 600+.
        experiment_number = experiment_number_of(args.experiment_number_base, seed)
        data_dir = args.data_dir_template.format(seed=seed) if args.data_dir_template else args.data_dir
        cmd = [
            sys.executable, script_path,
            '--predictor_path', args.predictor_path,
            '--data_dir', data_dir,
            '--output_dir', args.output_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--seed', str(seed),
            '--experiment_number', str(experiment_number),
        ] + args.extra_args.split()
        if args.data_dir_template and seed != args.predictor_split_seed:
            cmd.append('--skip_autogluon_eval')

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
