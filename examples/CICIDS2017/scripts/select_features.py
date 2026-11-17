"""
Experiment 5 keeps only the most important features. This script turns the sequential backward elimination of
feature_ranking.py into the list of kept features (`important_features.txt`, read by
autogluon_to_pytorch_qat.py --important_features_file), with a rule that is the same for every dataset: the smallest subset
whose accuracy on the fine-tuning split (the elimination curve, leaderboard.csv) is within --tolerance percentage points of the
accuracy with all the features. --k fixes the size instead.

Usage:
    python scripts/select_features.py --ranking_dir <predictor_dir>/feature_ranking_custom_sbe [--tolerance 1.0 | --k 10]
"""
import argparse
import json
import os

import pandas as pd


def select(ranking_dir, tolerance=1.0, k=None):
    leaderboard = pd.read_csv(os.path.join(ranking_dir, 'leaderboard.csv'), index_col='ID').sort_index()
    order = pd.read_csv(os.path.join(ranking_dir, 'elimination_order.csv'))
    all_features = list(pd.read_csv(os.path.join(ranking_dir, 'feature_importance.csv'), index_col='ID', nrows=0).columns)
    n_features = len(all_features)
    curve = leaderboard['accuracy'] * 100
    full_accuracy = float(curve.loc[n_features])

    if k is None:
        eligible = curve[(curve >= full_accuracy - tolerance) & (curve.index >= 1)]
        if eligible.empty:
            raise SystemExit('no subset reaches the tolerance')
        k = int(eligible.index.min())
    removed = order.loc[order['features_after'] >= k, 'feature'].tolist()   # eliminated while more than k features were left
    kept = [feature for feature in all_features if feature not in removed]
    if len(kept) != k:
        raise SystemExit(f'inconsistent elimination order: {len(kept)} features kept for k={k}')
    return kept, k, full_accuracy, float(curve.loc[k]), curve


def main():
    parser = argparse.ArgumentParser(description='Write the list of features kept by Experiment 5')
    parser.add_argument('--ranking_dir', type=str, required=True, help='feature_ranking_<algorithm>/ written by feature_ranking.py')
    parser.add_argument('--tolerance', type=float, default=1.0,
                        help='accepted drop (percentage points) of the fine-tuning accuracy with respect to all the features')
    parser.add_argument('--k', type=int, default=None, help='keep exactly this many features instead of applying the tolerance')
    parser.add_argument('--output', type=str, default=None, help='default: <ranking_dir>/important_features.txt')
    args = parser.parse_args()

    kept, k, full_accuracy, accuracy, curve = select(args.ranking_dir, args.tolerance, args.k)
    output = args.output or os.path.join(args.ranking_dir, 'important_features.txt')
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kept) + '\n')
    summary = {'features_total': int(curve.index.max()),'features_kept': k, 'rule': f'tolerance {args.tolerance} pp' if args.k is None else f'k={args.k}',
               'accuracy_all_features': full_accuracy, 'accuracy_kept_features': accuracy, 'kept': kept}
    with open(os.path.join(os.path.dirname(output), 'feature_selection.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('accuracy of the elimination curve on the fine-tuning split (features -> accuracy %):')
    print('  ' + ', '.join(f'{int(n)}: {a:.2f}' for n, a in curve.loc[curve.index <= min(30, curve.index.max())].items()))
    print(f'kept {k} of {summary["features_total"]} features ({accuracy:.2f} % against {full_accuracy:.2f} % with all): {kept}')
    print(f'written to {output}')


if __name__ == '__main__':
    main()
