"""
Table comparing the near-zero rescaling with the alternative input preprocessings under the same QAT protocol (5 seeds, one data split per
seed), for any number of datasets: accuracy of the FP32 and INT8 (QAT) models, recall, FPR and F1 of the INT8 model as mean +/- std over seeds,
and the number of seeds in which the INT8 model loses more than 2 points with respect to the FP32 one; then paired comparisons of the proposed
rescaling with every alternative.

The seeds are paired: seed i of every experiment is trained and tested on split_i, so two experiments are compared seed by seed. For each
comparison (difference = proposed - alternative) it prints the mean difference with its 95 % t-interval, the median difference, the number of seeds
in which the proposed rescaling is ahead, the paired t-test, the exact Wilcoxon signed-rank test and the Holm-adjusted p of the t-tests of the
dataset (all alternatives and metrics together). With 5 non-zero pairs the two-sided Wilcoxon p cannot be below 0.0625, and a collapsed seed (INT8
accuracy near 50 %) makes the differences far from normal: read the median, the counts and the intervals before the p-values.

Usage:
    python scripts/build_preprocessing_table.py \
        --dataset CICIDS2017_improved=<data>/CICIDS2017_improved/balanced_binary \
                  UNSW_NB15=<data>/UNSW_NB15/balanced_binary,compressed_models_bnfold \
        [--variants 3 6 7 8 4] [--proposed 4] [--output_dir <dir for the CSV copies>]
(see utils/results.py for NAME=ROOT[,MODELS_DIR])
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils.experiments import EXPERIMENTS  # noqa: E402
from utils.results import load_multiseed, parse_dataset  # noqa: E402

# experiment number -> label of the table; the metrics of every run are in its multiseed CSV
LABELS = {3: 'Min-Max only (Exp. 3)', 6: 'Standardization', 7: 'Log Min-Max', 8: 'Quantile transform', 4: 'Near-zero rescaling (Exp. 4)'}
DEFAULT_VARIANTS = [3, 6, 7, 8, 4]
COLLAPSE_PP = 2.0
# column, name, direction (+1: higher is better, -1: lower is better)
COMPARED = (('fp32_acc', 'FP32 accuracy', +1), ('qat_acc', 'INT8 accuracy', +1), ('qat_fpr', 'INT8 FPR', -1))


def label_of(experiment):
    return LABELS.get(experiment, EXPERIMENTS[experiment][0])


def load_variant(root, models_dir, experiment):
    df = load_multiseed(root, models_dir, experiment)
    if df is None:
        return None
    out = pd.DataFrame({'seed': df['seed'], 'fp32_acc': df['normal_test_accuracy'], 'qat_acc': df['qat_test_accuracy']})
    for metric in ('recall', 'fpr', 'f1'):
        out[f'qat_{metric}'] = df[f'qat_{metric}'] * 100
    return out


def fmt(series):
    return f'{series.mean():.2f} $\\pm$ {series.std():.2f}'


def holm(pvalues):
    """Holm step-down adjustment; NaN p-values are left out of the family and stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def exact_wilcoxon_p(difference, max_pairs=16):
    """Two-sided exact p of the Wilcoxon signed-rank test, from the 2^n sign patterns of the non-zero differences (mid-ranks for ties). It does not
    depend on the scipy version, which switches to a normal approximation, useless with 5 pairs, as soon as there is a zero or a tie."""
    d = difference[difference != 0]
    n = len(d)
    if n == 0:
        return float('nan')
    if n > max_pairs:
        return float(stats.wilcoxon(d).pvalue)
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    positive = ranks[d > 0].sum()
    observed = min(positive, total - positive)
    patterns = np.array(list(itertools.product((0.0, 1.0), repeat=n)))
    sums = patterns @ ranks
    return float((np.minimum(sums, total - sums) <= observed + 1e-9).mean())


def paired_comparison(proposed, alternative, column, direction):
    """Seed-by-seed comparison of `proposed` with `alternative` on one column (difference = proposed - alternative)."""
    merged = proposed[['seed', column]].merge(alternative[['seed', column]], on='seed', suffixes=('_p', '_a')).dropna()
    x, y = merged[f'{column}_p'].to_numpy(), merged[f'{column}_a'].to_numpy()
    difference = x - y
    n = len(difference)
    result = {'n': n, 'mean_diff': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'median_diff': np.nan,
              'ahead': int((direction * difference > 0).sum()), 'p_t': np.nan, 'p_wilcoxon': np.nan}
    if n == 0:
        return result
    result['mean_diff'], result['median_diff'] = difference.mean(), float(np.median(difference))
    if n >= 2 and difference.std(ddof=1) > 0:
        half_width = stats.t.ppf(0.975, n - 1) * difference.std(ddof=1) / np.sqrt(n)
        result['ci_low'], result['ci_high'] = difference.mean() - half_width, difference.mean() + half_width
        result['p_t'] = float(stats.ttest_rel(x, y).pvalue)
    result['p_wilcoxon'] = exact_wilcoxon_p(difference)
    return result


def main():
    parser = argparse.ArgumentParser(description='Preprocessing comparison table')
    parser.add_argument('--dataset', type=str, nargs='+', required=True, metavar='NAME=ROOT[,MODELS_DIR]')
    parser.add_argument('--variants', type=int, nargs='+', default=DEFAULT_VARIANTS, choices=sorted(EXPERIMENTS))
    parser.add_argument('--proposed', type=int, default=4, help='experiment compared with every other variant (default 4, the near-zero rescaling)')
    parser.add_argument('--output_dir', type=str, default=None, help='also write preprocessing_table.csv and preprocessing_tests.csv here')
    args = parser.parse_args()
    try:
        datasets = [parse_dataset(spec) for spec in args.dataset]
    except ValueError as error:
        parser.error(str(error))
    if args.proposed not in args.variants:
        parser.error('--proposed must be one of the --variants')

    table_rows, test_rows = [], []
    for name, root, models_dir in datasets:
        data = {n: load_variant(root, models_dir, n) for n in args.variants}
        data = {n: df for n, df in data.items() if df is not None}
        print(f'\n% ===== {name} =====')
        print('% label & FP32 acc & INT8 acc & INT8 recall & INT8 FPR & INT8 F1 & seeds with an INT8 loss > 2 pp')
        for n, df in data.items():
            lost = int(((df['fp32_acc'] - df['qat_acc']) > COLLAPSE_PP).sum())
            print(f"{label_of(n)} & {fmt(df['fp32_acc'])} & {fmt(df['qat_acc'])} & {fmt(df['qat_recall'])} & "
                  f"{fmt(df['qat_fpr'])} & {fmt(df['qat_f1'])} & {lost}/{len(df)} \\\\")
            table_rows.append({'dataset': name, 'experiment': n, 'label': label_of(n), 'n_seeds': len(df), 'seeds_lost_gt_2pp': lost,
                               **{f'{c}_{s}': getattr(df[c], s)() for c in ('fp32_acc', 'qat_acc', 'qat_recall', 'qat_fpr', 'qat_f1')
                                  for s in ('mean', 'std')}})

        if args.proposed not in data:
            print(f'% proposed experiment {args.proposed} has no results for {name}: no comparison')
            continue
        comparisons = []
        for n, df in data.items():
            if n == args.proposed:
                continue
            for column, column_name, direction in COMPARED:
                comparisons.append((n, column, column_name, paired_comparison(data[args.proposed], df, column, direction)))
        adjusted = holm([c[3]['p_t'] for c in comparisons])

        print(f'\n% paired comparison of {label_of(args.proposed)} with each alternative (difference = proposed - alternative, in points; '
              f'ahead = seeds in which the proposed one is better; Holm over the {len(comparisons)} t-tests of this dataset)')
        for (n, column, column_name, result), p_holm in zip(comparisons, adjusted):
            interval = '[n/a]' if np.isnan(result['ci_low']) else f"[{result['ci_low']:+.2f}, {result['ci_high']:+.2f}]"
            print(f"{label_of(n):30s} {column_name:14s} mean diff {result['mean_diff']:+.2f} {interval:18s} median {result['median_diff']:+.2f}  "
                  f"ahead {result['ahead']}/{result['n']}  p_t={result['p_t']:.3f}  p_holm={p_holm:.3f}  p_wilcoxon={result['p_wilcoxon']:.3f}")
            test_rows.append({'dataset': name, 'proposed': args.proposed, 'alternative': n, 'alternative_label': label_of(n), 'metric': column,
                              **result, 'p_holm': p_holm})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(table_rows).to_csv(os.path.join(args.output_dir, 'preprocessing_table.csv'), index=False)
        pd.DataFrame(test_rows).to_csv(os.path.join(args.output_dir, 'preprocessing_tests.csv'), index=False)
        print(f'\nCSV copies written to {args.output_dir}')


if __name__ == '__main__':
    main()
