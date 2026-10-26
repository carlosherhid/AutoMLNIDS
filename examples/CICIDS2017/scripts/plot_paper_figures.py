"""
Publication-ready figures for the paper.

Two data sources, both optional but at least one required:
  1. run_multiseed.py's multiseed_exp<N>_results.csv (N in {1,2,3,6,7}) - gives
     mean +/- std accuracy across 5 seeds per paper experiment, drawn as bar charts
     with error bars.
  2. build_tradeoff_table.py's tradeoff_table.csv - adds latency percentiles and
     weight sparsity for the accuracy-vs-latency trade-off plot, and,
     if it was run with --latency_batch_sizes covering more than batch=1, a
     latency-vs-batch-size plot: QAT is consistently *slower* than
     FP32 per-sample at batch=1 (quantize/dequantize call overhead dominates
     for models this small), and whether/where that crosses over as batch
     size grows depends on model size - it crosses for CICIDS2017 (213 KB)
     around batch=128 but not for CICIoT2023 (12 KB, still ~2x slower at 128).

Usage (run once per dataset you have, or both for the cross-dataset comparison):
    python scripts/plot_paper_figures.py \
        --cicids_logs_dir ~/automl_nids/data/CICIDS2017/balanced_binary/compressed_models \
        --ciciot_logs_dir ~/automl_nids/data/CICIoT2023/balanced_binary/compressed_models \
        --output_dir ~/automl_nids/figures
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.log_parsing import parse_log_file  # noqa: E402

# Defined here instead of imported from utils.common, which pulls in AutoGluon and
# torch - this script only needs matplotlib/pandas and should run on any machine.
DEFAULT_PLOT_STYLE = os.path.join(os.path.dirname(__file__), 'utils', 'presentation.mplstyle')

# Internal --experiment_number_base -> paper Experiment (see docs/EXPERIMENTS_LOG.md /
# build_tradeoff_table.py's PAPER_EXPERIMENTS - kept in sync manually, both small dicts).
PAPER_EXPERIMENTS = {1: 'Exp. 1', 2: 'Exp. 2', 3: 'Exp. 3', 6: 'Exp. 4', 7: 'Exp. 5'}
EXPERIMENT_ORDER = [1, 2, 3, 6, 7]

COLOR_NORMAL = '#4C72B0'
COLOR_QAT = '#DD8452'
COLOR_DATASET_A = '#4C72B0'
COLOR_DATASET_B = '#55A868'
COLOR_AUTOGLUON = '#8172B3'

# Must match build_tradeoff_table.py's --latency_batch_sizes default.
LATENCY_BATCH_SIZES = [1, 8, 32, 64, 128]


def load_multiseed_summary(logs_dir):
    """One row per paper experiment: mean+/-std accuracy across seeds, plus the
    (seed-invariant, since only training data/config differs per seed, not
    architecture) model sizes."""
    rows = []
    for base in EXPERIMENT_ORDER:
        path = os.path.join(logs_dir, f'multiseed_exp{base}_results.csv')
        if not os.path.exists(path):
            print(f'WARNING: missing {path}, skipping {PAPER_EXPERIMENTS[base]}', file=sys.stderr)
            continue
        df = pd.read_csv(path)
        rows.append({
            'experiment_number': base,
            'paper_experiment': PAPER_EXPERIMENTS[base],
            'normal_mean': df['normal_test_accuracy'].mean(),
            'normal_std': df['normal_test_accuracy'].std(),
            'qat_mean': df['qat_test_accuracy'].mean(),
            'qat_std': df['qat_test_accuracy'].std(),
            'normal_size_kb': df['normal_size_kb'].iloc[0],
            'qat_size_kb': df['qat_size_kb'].iloc[0],
            'n_seeds': len(df),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_accuracy_bars(summary, dataset_label, output_path):
    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width / 2, summary['normal_mean'], width, yerr=summary['normal_std'],
           capsize=3, label='Normal (FP32)', color=COLOR_NORMAL)
    ax.bar(x + width / 2, summary['qat_mean'], width, yerr=summary['qat_std'],
           capsize=3, label='QAT (INT8)', color=COLOR_QAT)

    ax.set_xticks(x)
    ax.set_xticklabels(summary['paper_experiment'])
    ax.set_ylabel('Test accuracy (%)')
    lower = min(summary['normal_mean'].min() - summary['normal_std'].max(),
                summary['qat_mean'].min() - summary['qat_std'].max())
    ax.set_ylim(max(0, lower - 5), 101)
    ax.set_title(f'{dataset_label}: Normal vs. QAT accuracy\n(mean $\\pm$ std, {int(summary["n_seeds"].iloc[0])} seeds)')
    # Bars span the full y-range (ylim doesn't start at 0), so the style
    # sheet's default legend.loc='lower left' sits on top of the bars -
    # place it below the x-axis instead, which is always empty.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_cross_dataset_qat(summary_a, label_a, summary_b, label_b, output_path):
    # Only plot experiments present in both summaries, in EXPERIMENT_ORDER.
    common = sorted(set(summary_a['experiment_number']) & set(summary_b['experiment_number']),
                     key=EXPERIMENT_ORDER.index)
    a = summary_a.set_index('experiment_number').loc[common]
    b = summary_b.set_index('experiment_number').loc[common]

    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(common))
    width = 0.35

    ax.bar(x - width / 2, a['qat_mean'], width, yerr=a['qat_std'],
           capsize=3, label=label_a, color=COLOR_DATASET_A)
    ax.bar(x + width / 2, b['qat_mean'], width, yerr=b['qat_std'],
           capsize=3, label=label_b, color=COLOR_DATASET_B)

    ax.set_xticks(x)
    ax.set_xticklabels([PAPER_EXPERIMENTS[e] for e in common])
    ax.set_ylabel('QAT test accuracy (%)')
    lower = min(a['qat_mean'].min() - a['qat_std'].max(), b['qat_mean'].min() - b['qat_std'].max())
    ax.set_ylim(max(0, lower - 5), 101)
    ax.set_title('QAT accuracy across datasets\n(mean $\\pm$ std, 5 seeds)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_model_size(summary, dataset_label, output_path):
    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    normal_kb = summary['normal_size_kb'].iloc[0]
    qat_kb = summary['qat_size_kb'].iloc[0]
    reduction = (1 - qat_kb / normal_kb) * 100

    bars = ax.bar(['Normal\n(FP32)', 'QAT\n(INT8)'], [normal_kb, qat_kb],
                   color=[COLOR_NORMAL, COLOR_QAT])
    for bar, val in zip(bars, [normal_kb, qat_kb]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f} KB',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Model size (KB)')
    ax.set_ylim(0, normal_kb * 1.15)
    ax.set_title(f'{dataset_label}: model size\n({reduction:.1f}% reduction)')

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def _stack_labels(ax, points, color, dx=8, dy=13, col_gap=26):
    """Annotate (x, y, label) points with a two-column staircase of label
    offsets (in screen-space points, not data units), so labels for points
    that are nearly identical in data still don't overlap - e.g. CICIoT2023,
    where several experiments tie exactly at the same 2-decimal accuracy
    (the log only records 2 decimals), so a single-column stack put labels
    so close together the leader lines and text visually collided. Two
    columns approximately halve the vertical footprint per anchor point.
    A thin leader line connects each label back to its point."""
    for i, (x, y, label) in enumerate(sorted(points, key=lambda p: p[1])):
        col, row = i % 2, i // 2
        xytext = (dx + col * col_gap, 4 + row * dy)
        ax.annotate(label, (x, y), xytext=xytext, textcoords='offset points',
                    fontsize=6, color=color,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=color, shrinkA=0, shrinkB=2))


def plot_tradeoff_scatter(tradeoff_df, dataset_label, output_path):
    """Accuracy vs. p50 latency, one point per paper experiment x {Normal, QAT}.
    Requires build_tradeoff_table.py to have been run WITHOUT --skip_latency."""
    # Filter by experiment_number, not the paper_experiment string: build_tradeoff_table.py
    # writes '' for non-paper rows, but pd.read_csv round-trips that back as NaN, and
    # `!= ''` doesn't catch NaN - it let every experiment through (with a "nan" label).
    df = tradeoff_df[tradeoff_df['experiment_number'].isin(EXPERIMENT_ORDER)].copy()
    if 'normal_latency_p50_ms' not in df.columns or df['normal_latency_p50_ms'].isna().all():
        print(f'WARNING: no latency columns in tradeoff table for {dataset_label}, '
              f'skipping trade-off scatter (re-run build_tradeoff_table.py without --skip_latency)',
              file=sys.stderr)
        return

    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    normal_points, qat_points = [], []
    for _, row in df.iterrows():
        label = row['paper_experiment']
        if pd.notna(row.get('normal_latency_p50_ms')) and pd.notna(row.get('normal_test_accuracy')):
            ax.scatter(row['normal_latency_p50_ms'], row['normal_test_accuracy'],
                       marker='o', color=COLOR_NORMAL, s=40, zorder=3)
            normal_points.append((row['normal_latency_p50_ms'], row['normal_test_accuracy'], label))
        if pd.notna(row.get('qat_latency_p50_ms')) and pd.notna(row.get('qat_test_accuracy')):
            ax.scatter(row['qat_latency_p50_ms'], row['qat_test_accuracy'],
                       marker='^', color=COLOR_QAT, s=40, zorder=3)
            qat_points.append((row['qat_latency_p50_ms'], row['qat_test_accuracy'], label))

    _stack_labels(ax, normal_points, COLOR_NORMAL)
    _stack_labels(ax, qat_points, COLOR_QAT)

    ax.scatter([], [], marker='o', color=COLOR_NORMAL, label='Normal (FP32)')
    ax.scatter([], [], marker='^', color=COLOR_QAT, label='QAT (INT8)')
    ax.set_xlabel('CPU inference latency, p50 (ms)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'{dataset_label}: accuracy vs. latency trade-off')
    # Extra margin so the staircase of per-point "Exp. N" annotations doesn't
    # get clipped at the plot edges, and the legend goes below the x-axis
    # (outside the data area) instead of overlapping points.
    ax.margins(x=0.25, y=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_latency_vs_batch(tradeoff_df, dataset_label, output_path, experiment_numbers=None):
    """Per-sample CPU latency (p50) vs. batch size, overlaying every paper
    experiment (not just one) so the trend isn't mistaken for an artifact of
    a single run. At batch_size=1, QAT's per-call quantize/dequantize overhead
    can make it *slower* than FP32 despite being smaller; this shows whether/
    where that overhead amortizes as batch size grows (latency detail beyond a
    single mean number). Requires build_tradeoff_table.py to have
    been run with --latency_batch_sizes covering more than just 1."""
    if experiment_numbers is None:
        experiment_numbers = EXPERIMENT_ORDER

    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    all_batch_sizes = set()
    plotted_count = 0
    for exp_number in experiment_numbers:
        match = tradeoff_df[tradeoff_df['experiment_number'] == exp_number]
        if match.empty:
            print(f'WARNING: experiment_number={exp_number} not in tradeoff table for '
                  f'{dataset_label}, skipping it in the latency-vs-batch plot', file=sys.stderr)
            continue
        row = match.iloc[0]

        batch_sizes, normal_lat, qat_lat = [], [], []
        for bs in LATENCY_BATCH_SIZES:
            suffix = '' if bs == 1 else f'_b{bs}'
            normal_col, qat_col = f'normal_latency{suffix}_p50_ms', f'qat_latency{suffix}_p50_ms'
            if normal_col not in row.index or pd.isna(row[normal_col]) or qat_col not in row.index or pd.isna(row[qat_col]):
                continue
            batch_sizes.append(bs)
            normal_lat.append(row[normal_col])
            qat_lat.append(row[qat_col])

        if len(batch_sizes) < 2:
            continue

        all_batch_sizes.update(batch_sizes)
        # Only label the first plotted experiment, so the legend has one
        # entry per {Normal, QAT} instead of one per experiment; the curves
        # overlap almost exactly since latency depends on architecture/size,
        # not on the training procedure that differs between experiments.
        label_normal = 'Normal (FP32)' if plotted_count == 0 else None
        label_qat = 'QAT (INT8)' if plotted_count == 0 else None
        ax.plot(batch_sizes, normal_lat, marker='o', color=COLOR_NORMAL, alpha=0.6, label=label_normal)
        ax.plot(batch_sizes, qat_lat, marker='^', color=COLOR_QAT, alpha=0.6, label=label_qat)
        plotted_count += 1

    if plotted_count == 0:
        print(f'WARNING: no experiment had >=2 batch sizes with latency data for {dataset_label}, '
              f'skipping latency-vs-batch plot (re-run build_tradeoff_table.py without --skip_latency)',
              file=sys.stderr)
        plt.close(fig)
        return

    batch_sizes_sorted = sorted(all_batch_sizes)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(batch_sizes_sorted)
    ax.set_xticklabels([str(b) for b in batch_sizes_sorted])
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Per-sample CPU latency, p50 (ms)')
    ax.set_title(f'{dataset_label}: latency vs. batch size\n({plotted_count} paper experiments overlaid)')
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_scale_factor_sweep(sweep_df, dataset_label, output_path, chosen_factor=1e6):
    """Test accuracy vs. near-zero rescaling factor (scale_factor_sweep.py's
    scale_factor_sweep_results.csv). Single seed per factor, so the plateau
    rather than the exact argmax is the finding - the chosen factor is marked
    to show it sits inside the plateau."""
    df = sweep_df.sort_values('scale_factor')

    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(df['scale_factor'], df['normal_test_accuracy'], marker='o', color=COLOR_NORMAL, label='Normal (FP32)')
    ax.plot(df['scale_factor'], df['qat_test_accuracy'], marker='^', color=COLOR_QAT, label='QAT (INT8)')
    ax.margins(y=0.1)
    ax.axvline(chosen_factor, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.annotate('used in\nExp. 4/5', (chosen_factor, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords='offset points', fontsize=7, color='gray', va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('Near-zero rescaling factor')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'{dataset_label}: accuracy vs. rescaling factor\n(single seed per factor)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_noise_robustness(panels, output_path):
    """Accuracy under Gaussian input noise, FP32 vs. QAT (robustness_noise_eval.py's
    robustness_noise_eval.csv, accuracies stored as fractions). One panel per
    dataset on a shared y-axis so the two datasets' opposite FP32-vs-QAT
    ordering is directly comparable. Noise levels are roughly log-spaced
    (0, 0.1, 0.25, ... 2x feature std), so they are plotted equally spaced
    rather than on a linear axis that would crowd the first four points."""
    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.1), sharey=True, squeeze=False)
    for ax, (dataset_label, df) in zip(axes[0], panels):
        x = np.arange(len(df))
        ax.plot(x, df['normal_accuracy'] * 100, marker='o', color=COLOR_NORMAL, label='Normal (FP32)')
        ax.plot(x, df['qat_accuracy'] * 100, marker='^', color=COLOR_QAT, label='QAT (INT8)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{level:g}' for level in df['noise_level']])
        ax.set_xlabel('Gaussian noise ($\\times$ feature std)')
        ax.set_title(dataset_label)
    axes[0][0].set_ylabel('Test accuracy (%)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.3, wspace=0.08)
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def parse_qat_val_curve(log_path):
    """Per-epoch QAT validation accuracy (%) from an execution_log_exp<N>.txt
    (each log has a Normal-training section followed by a QAT one)."""
    with open(log_path, encoding='utf-8') as f:
        content = f.read()
    if 'Training QAT PyTorch Model:' not in content:
        raise ValueError(f'{log_path} has no QAT training section')
    qat_section = content.split('Training QAT PyTorch Model:')[1]
    return [float(v) * 100 for v in re.findall(r'Validation Accuracy: ([0-9.]+)', qat_section)]


def plot_qat_val_curves(curves, titles, output_path):
    """QAT validation accuracy per epoch, one panel per experiment. The star marks
    the epoch whose checkpoint is restored for the final test evaluation (best
    validation accuracy)."""
    plt.style.use(DEFAULT_PLOT_STYLE)
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n if n > 1 else 3.8, 2.8), squeeze=False)
    for ax, acc, title in zip(axes[0], curves, titles):
        epochs = np.arange(1, len(acc) + 1)
        best = int(np.argmax(acc))
        ax.plot(epochs, acc, color=COLOR_QAT, linewidth=1, label='QAT validation accuracy')
        ax.scatter([epochs[best]], [acc[best]], marker='*', s=60, color='black', zorder=3,
                   label='Restored checkpoint (best epoch)')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
    axes[0][0].set_ylabel('Validation accuracy (%)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32, wspace=0.3)
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_exp1_final_accuracy(autogluon_acc, normal_acc, qat_acc, output_path):
    """AutoGluon vs. Normal (FP32) vs. QAT test accuracy for Experiment 1, single run."""
    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    values = [autogluon_acc, normal_acc, qat_acc]
    bars = ax.bar(['AutoGluon', 'Normal\n(FP32)', 'QAT\n(INT8)'], values,
                  color=[COLOR_AUTOGLUON, COLOR_NORMAL, COLOR_QAT])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(np.floor(min(values) - 3), 100.5)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('CICIDS2017: Experiment 1\n(single run, seed 42)')

    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate publication-ready figures for the paper')
    parser.add_argument('--cicids_logs_dir', type=str, default=None,
                         help='CICIDS2017 balanced_binary/compressed_models directory')
    parser.add_argument('--ciciot_logs_dir', type=str, default=None,
                         help='CICIoT2023 balanced_binary/compressed_models directory')
    parser.add_argument('--cicids_tradeoff_csv', type=str, default=None,
                         help='Optional: build_tradeoff_table.py output for CICIDS2017 (adds latency scatter)')
    parser.add_argument('--ciciot_tradeoff_csv', type=str, default=None,
                         help='Optional: build_tradeoff_table.py output for CICIoT2023 (adds latency scatter)')
    parser.add_argument('--scale_sweep_csv', type=str, default=None,
                         help='Optional: scale_factor_sweep.py output (CICIDS2017) -> scale_factor_sweep.pdf')
    parser.add_argument('--cicids_noise_csv', type=str, default=None,
                         help='Optional: robustness_noise_eval.py output for CICIDS2017')
    parser.add_argument('--ciciot_noise_csv', type=str, default=None,
                         help='Optional: robustness_noise_eval.py output for CICIoT2023')
    parser.add_argument('--exp_logs_dir', type=str, default=None,
                         help='Optional: CICIDS2017 directory with execution_log_exp{1,2,3,6,7}.txt - regenerates '
                              'fig1.pdf and the QAT validation-curve figures from the current runs')
    parser.add_argument('--autogluon_accuracy', type=float, default=None,
                         help='AutoGluon test accuracy (%%) for fig1.pdf; not in the exp1 log when the AutoGluon '
                              'evaluation was skipped (take it from verify_functional_equivalence.py)')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp_logs_dir:
        log = lambda n: os.path.join(args.exp_logs_dir, f'execution_log_exp{n}.txt')  # noqa: E731
        plot_qat_val_curves([parse_qat_val_curve(log(n)) for n in (1, 2, 3)], ['Experiment 1', 'Experiment 2', 'Experiment 3'],
                            os.path.join(args.output_dir, 'experiment_exp1_3_val_accuracy.pdf'))
        plot_qat_val_curves([parse_qat_val_curve(log(6))], ['Experiment 4'],
                            os.path.join(args.output_dir, 'experiment_4_qat_val_accuracy.pdf'))
        plot_qat_val_curves([parse_qat_val_curve(log(7))], ['Experiment 5'],
                            os.path.join(args.output_dir, 'experiment_5_qat_val_accuracy.pdf'))
        if args.autogluon_accuracy is not None:
            exp1 = parse_log_file(log(1))
            plot_exp1_final_accuracy(args.autogluon_accuracy, exp1['normal_test_accuracy'], exp1['qat_test_accuracy'],
                                      os.path.join(args.output_dir, 'fig1.pdf'))

    summary_cicids = summary_ciciot = None

    if args.cicids_logs_dir:
        summary_cicids = load_multiseed_summary(args.cicids_logs_dir)
        if summary_cicids is not None:
            summary_cicids.to_csv(os.path.join(args.output_dir, 'cicids2017_multiseed_summary.csv'), index=False)
            plot_accuracy_bars(summary_cicids, 'CICIDS2017',
                                os.path.join(args.output_dir, 'cicids2017_accuracy_multiseed.pdf'))
            plot_model_size(summary_cicids, 'CICIDS2017',
                             os.path.join(args.output_dir, 'cicids2017_model_size.pdf'))

    if args.ciciot_logs_dir:
        summary_ciciot = load_multiseed_summary(args.ciciot_logs_dir)
        if summary_ciciot is not None:
            summary_ciciot.to_csv(os.path.join(args.output_dir, 'ciciot2023_multiseed_summary.csv'), index=False)
            plot_accuracy_bars(summary_ciciot, 'CICIoT2023',
                                os.path.join(args.output_dir, 'ciciot2023_accuracy_multiseed.pdf'))
            plot_model_size(summary_ciciot, 'CICIoT2023',
                             os.path.join(args.output_dir, 'ciciot2023_model_size.pdf'))

    if summary_cicids is not None and summary_ciciot is not None:
        plot_cross_dataset_qat(summary_cicids, 'CICIDS2017', summary_ciciot, 'CICIoT2023',
                                os.path.join(args.output_dir, 'cross_dataset_qat_accuracy.pdf'))

    if args.cicids_tradeoff_csv:
        cicids_tradeoff = pd.read_csv(args.cicids_tradeoff_csv)
        plot_tradeoff_scatter(cicids_tradeoff, 'CICIDS2017',
                               os.path.join(args.output_dir, 'cicids2017_tradeoff_scatter.pdf'))
        plot_latency_vs_batch(cicids_tradeoff, 'CICIDS2017',
                               os.path.join(args.output_dir, 'cicids2017_latency_vs_batch.pdf'))

    if args.ciciot_tradeoff_csv:
        ciciot_tradeoff = pd.read_csv(args.ciciot_tradeoff_csv)
        plot_tradeoff_scatter(ciciot_tradeoff, 'CICIoT2023',
                               os.path.join(args.output_dir, 'ciciot2023_tradeoff_scatter.pdf'))
        plot_latency_vs_batch(ciciot_tradeoff, 'CICIoT2023',
                               os.path.join(args.output_dir, 'ciciot2023_latency_vs_batch.pdf'))

    if args.scale_sweep_csv:
        plot_scale_factor_sweep(pd.read_csv(args.scale_sweep_csv), 'CICIDS2017',
                                 os.path.join(args.output_dir, 'scale_factor_sweep.pdf'))

    noise_panels = [(label, pd.read_csv(path)) for label, path in
                    (('CICIDS2017', args.cicids_noise_csv), ('CICIoT2023', args.ciciot_noise_csv)) if path]
    if noise_panels:
        plot_noise_robustness(noise_panels, os.path.join(args.output_dir, 'noise_robustness.pdf'))


if __name__ == '__main__':
    main()
