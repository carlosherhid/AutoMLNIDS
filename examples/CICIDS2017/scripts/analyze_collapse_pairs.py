"""
Paired analysis of how often the QAT model "collapses" with and without a treatment (for example Experiment 3, Min-Max input, against
Experiment 4, near-zero rescaling), over several data splits and several thread configurations.

A PAIR is one (run directory, seed): the two experiments share the data split, the training seed and the number of threads, so they differ only
in the treatment. The run directories are the `compressed_models_<name>` folders written by `cluster/submit_experiments.sh --run-name <name>`
(each holds the `multiseed_exp<N>_results.csv` files).

Outcome fixed before looking at the data: a run COLLAPSES when its QAT test accuracy is more than --margin percentage points below the FP32
accuracy of the SAME run. The analysis is repeated for each scoring of the QAT weights that the logs contain: the INT8 model under the default
engine (`qat_test_accuracy`), the fake-quantized model (`qat_fake_accuracy`) and the INT8 model under the other engines.

Reported: collapse rates with Wilson intervals, the discordant pairs, the exact one-sided McNemar test (naive collapses where the treated model does
not), the Wilcoxon signed-rank test on the accuracy cost (FP32 - QAT) and, as a reproducibility check, how much the QAT accuracy of the same seed moves
across thread configurations compared with how much it moves across seeds.

Usage:
    python scripts/analyze_collapse_pairs.py --dirs <data>/balanced_binary/compressed_models_t4 <...>_t6 <...>_t12 --naive 3 --treated 4
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

SCORINGS = [('qat_test_accuracy', 'INT8, default engine of the run'), ('qat_fake_accuracy', 'fake-quantized model'),
            ('qat_onednn_accuracy', 'INT8, onednn engine'), ('qat_x86_accuracy', 'INT8, x86 engine')]


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load_pairs(dirs, naive, treated):
    rows = []
    for directory in dirs:
        name = os.path.basename(os.path.normpath(directory)).replace('compressed_models_', '').replace('compressed_models', 'default')
        a = pd.read_csv(os.path.join(directory, f'multiseed_exp{naive}_results.csv')).set_index('seed')
        b = pd.read_csv(os.path.join(directory, f'multiseed_exp{treated}_results.csv')).set_index('seed')
        for seed in sorted(set(a.index) & set(b.index)):
            row = {'run': name, 'seed': seed, 'fp32_naive': a.loc[seed, 'normal_test_accuracy'], 'fp32_treated': b.loc[seed, 'normal_test_accuracy']}
            for column, _ in SCORINGS:
                row[f'{column}_naive'] = a.loc[seed, column] if column in a.columns else np.nan
                row[f'{column}_treated'] = b.loc[seed, column] if column in b.columns else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Paired collapse analysis')
    parser.add_argument('--dirs', nargs='+', help='run directories (compressed_models_<name>)')
    parser.add_argument('--glob', type=str, default=None, help='alternative to --dirs, e.g. "<data>/balanced_binary/compressed_models_t*"')
    parser.add_argument('--naive', type=int, default=3, help='experiment without the treatment')
    parser.add_argument('--treated', type=int, default=4, help='experiment with the treatment')
    parser.add_argument('--margin', type=float, default=2.0, help='collapse = QAT accuracy more than this many points below the FP32 of the same run')
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()
    dirs = list(args.dirs or []) + (sorted(glob.glob(args.glob)) if args.glob else [])
    if not dirs:
        parser.error('give --dirs or --glob')

    pairs = load_pairs(dirs, args.naive, args.treated)
    print(f'{len(pairs)} pairs from {pairs.run.nunique()} run directories ({", ".join(sorted(pairs.run.unique()))}); '
          f'collapse = QAT more than {args.margin} points below the FP32 of the same run\n')
    summary = []
    for column, label in SCORINGS:
        naive, treated = pairs[f'{column}_naive'], pairs[f'{column}_treated']
        if naive.isna().all() or treated.isna().all():
            continue
        valid = naive.notna() & treated.notna()
        n = int(valid.sum())
        collapse_naive = ((pairs.fp32_naive - naive) > args.margin)[valid]
        collapse_treated = ((pairs.fp32_treated - treated) > args.margin)[valid]
        b = int((collapse_naive & ~collapse_treated).sum())      # naive collapses, treated does not
        c = int((~collapse_naive & collapse_treated).sum())      # treated collapses, naive does not
        p_mcnemar = stats.binomtest(b, b + c, 0.5, alternative='greater').pvalue if b + c > 0 else float('nan')
        cost_naive, cost_treated = (pairs.fp32_naive - naive)[valid], (pairs.fp32_treated - treated)[valid]
        try:
            p_wilcoxon = stats.wilcoxon(cost_naive, cost_treated, alternative='greater').pvalue
        except ValueError:
            p_wilcoxon = float('nan')
        lo_n, hi_n = wilson(int(collapse_naive.sum()), n)
        lo_t, hi_t = wilson(int(collapse_treated.sum()), n)
        print(f'[{label}] n = {n} pairs')
        print(f'   collapse rate: naive {collapse_naive.mean() * 100:.0f}% (95% CI {lo_n * 100:.0f}-{hi_n * 100:.0f}), '
              f'treated {collapse_treated.mean() * 100:.0f}% (95% CI {lo_t * 100:.0f}-{hi_t * 100:.0f})')
        print(f'   discordant pairs: naive-only {b}, treated-only {c}; exact McNemar one-sided p = {p_mcnemar:.4f}')
        print(f'   accuracy cost FP32-QAT (mean): naive {cost_naive.mean():.2f} pp, treated {cost_treated.mean():.2f} pp; Wilcoxon one-sided p = {p_wilcoxon:.4f}')
        summary.append({'scoring': column, 'pairs': n, 'collapse_naive': collapse_naive.mean(), 'collapse_treated': collapse_treated.mean(),
                        'naive_only': b, 'treated_only': c, 'mcnemar_p': p_mcnemar, 'cost_naive': cost_naive.mean(), 'cost_treated': cost_treated.mean(),
                        'wilcoxon_p': p_wilcoxon})

    if pairs.run.nunique() > 1:
        print('\nReproducibility check: how much does the QAT accuracy move?')
        for column, label in SCORINGS[:2]:
            for side in ('naive', 'treated'):
                values = pairs[f'{column}_{side}']
                if values.isna().all():
                    continue
                table = pairs.assign(v=values).pivot(index='seed', columns='run', values='v')
                across_threads = (table.max(axis=1) - table.min(axis=1)).median()     # same seed (same split), different thread configuration
                across_seeds = table.std(axis=0).median()                             # same thread configuration, different seeds
                print(f'   {label:34s} {side:8s} median range across thread configurations (same seed) {across_threads:.2f} pp; '
                      f'median std across seeds (same threads) {across_seeds:.2f} pp')
    if args.output_csv:
        pairs.to_csv(args.output_csv, index=False)
        pd.DataFrame(summary).to_csv(args.output_csv.replace('.csv', '_summary.csv'), index=False)


if __name__ == '__main__':
    main()
