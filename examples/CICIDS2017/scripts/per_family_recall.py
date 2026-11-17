"""
Accuracy of the trained networks per attack family: for every experiment and seed, the FP32 and the INT8 (QAT) checkpoint classify the test
set of that seed's split, and the rows are grouped by the original attack label (the `test_attack_types.csv` that
`dataset_creator.py --keep-attack-type` writes next to `test.csv`).

Why: the overall accuracy counts rows, so the families that dominate the test set (DoS Hulk and DDoS in CICIDS2017) hide the rare ones. For an
attack family the figure is its recall (share of its flows detected); for a benign family (BENIGN, the '... - Attempted' flows that the
pipeline labels benign) it is the share classified as benign (1 - its false-positive rate).

Every network sees the input it was trained with: the experiment number gives the flags of utils/experiments.py (near-zero rescaling,
standardization / log-minmax / quantile transform fitted on the training split, feature selection). The overall accuracy recomputed here is
compared with the one in multiseed_exp<N>_results.csv; a difference means that the input or the checkpoint is not the one of the run.

Usage (CPU only, a few minutes):
    python scripts/per_family_recall.py --predictor_path <data>/balanced_binary/split_0/automl_search \
        --data_dir_template "<data>/balanced_binary/split_{seed}" --logs_dir <data>/balanced_binary/compressed_models \
        --important_features_file <predictor>/feature_ranking_custom_sbe/important_features.txt \
        --output_csv <data>/balanced_binary/analyses/per_family_recall.csv
    # Experiments 11/12 (BatchNorm folded) live in their own folder:
    python scripts/per_family_recall.py ... --experiments 11 12 --logs_dir <data>/balanced_binary/compressed_models_bn
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from autogluon_to_pytorch_qat import get_model_architecture  # noqa: E402
from build_tradeoff_table import try_load_checkpoint  # noqa: E402
from utils.cpu_loading import allow_cpu_loading  # noqa: E402
from utils.experiments import EXPERIMENTS, SELECTS_FEATURES, experiment_number  # noqa: E402
from utils.preprocessing import apply_preprocessing  # noqa: E402
from utils.quantization import DEFAULT_ENGINE, set_quantized_engine  # noqa: E402

FEATURE_EXCLUDED = ('ID', 'Label')
BATCH_SIZE = 8192


def input_recipe(base):
    """How the networks of experiment `base` were fed, read from the flags that define it."""
    flags = EXPERIMENTS[base][2].split()
    return {'rescale': '--scale_close_to_zero' in flags,
            'preprocess': flags[flags.index('--preprocess') + 1] if '--preprocess' in flags else 'none',
            'select': base in SELECTS_FEATURES}


def load_test_inputs(data_dir, recipe, scale_factor, features_file):
    """Test features, labels and original attack labels of one split, transformed as autogluon_to_pytorch_qat.py does:
    near-zero rescaling, then the alternative input transformation, then feature selection."""
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    columns = [c for c in test.columns if c not in FEATURE_EXCLUDED]
    if recipe['rescale']:
        test[columns] = test[columns].where(test[columns].abs() >= 1e-6, test[columns] * scale_factor)
    method = recipe['preprocess']
    if method in ('standard', 'quantile'):
        train = pd.read_csv(os.path.join(data_dir, 'train.csv'))   # the statistics are those of the training split
        _, test, _ = apply_preprocessing(train, test, test, method)
    elif method != 'none':
        _, test, _ = apply_preprocessing(test, test, test, method)
    if recipe['select']:
        keep = set(pd.read_csv(features_file, header=None).iloc[:, 0].tolist())
        test[[c for c in columns if c not in keep]] = 0

    labels = test['Label'].to_numpy()
    if not set(np.unique(labels)) <= {0, 1}:
        raise SystemExit(f'{data_dir}: expected binary labels 0/1 in test.csv, found {np.unique(labels)}')
    types = pd.read_csv(os.path.join(data_dir, 'test_attack_types.csv'), index_col='ID')['AttackType'].astype(str).str.strip()
    families = types.reindex(test['ID']).to_numpy()
    if pd.isna(families).any():
        raise SystemExit(f'{data_dir}: test_attack_types.csv does not cover every ID of test.csv (was the split made with --keep-attack-type?)')
    mixed = pd.Series(labels).groupby(families).mean()
    mixed = mixed[(mixed > 0) & (mixed < 1)]
    if len(mixed):
        print(f'WARNING: {data_dir}: families with both labels: {dict(mixed.round(3))}', file=sys.stderr)
    return test[columns].to_numpy(dtype=np.float32), labels, families


def predict(model, features):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            chunks.append(model(torch.from_numpy(features[start:start + BATCH_SIZE])).argmax(dim=1))
    return torch.cat(chunks).numpy()


def summarize(df, bases, min_rows):
    """Tables pooled over the seeds: % correct per family (INT8), what quantization costs per family, and the INT8 errors."""
    short = {base: f'E{base}' for base in bases}
    pooled = df.groupby(['experiment_base', 'model', 'family', 'class'], as_index=False)[['n', 'correct']].sum()
    totals = df.groupby(['experiment_base', 'model', 'class'], as_index=False)[['n', 'correct']].sum()
    totals['family'] = totals['class'].map({'attack': 'ALL ATTACKS', 'benign': 'ALL BENIGN'})
    pooled = pd.concat([pooled, totals], ignore_index=True)
    pooled['accuracy'] = 100 * pooled['correct'] / pooled['n']
    pooled['errors'] = pooled['n'] - pooled['correct']

    reference = pooled[(pooled.experiment_base == bases[0]) & (pooled.model == pooled.model.iloc[0])]
    order, label = [], {}
    for cls, total in (('attack', 'ALL ATTACKS'), ('benign', 'ALL BENIGN')):
        rows = reference[(reference['class'] == cls) & (reference.family != total)].sort_values('n', ascending=False)
        order += rows.family.tolist() + [total]
    for _, row in reference.iterrows():
        small = '*' if row.family not in ('ALL ATTACKS', 'ALL BENIGN') and row.n < min_rows else ''
        label[row.family] = f'{row.family}{small} (n={int(row.n)})'

    def table(model, value):
        wide = pooled[pooled.model == model].pivot_table(index='family', columns='experiment_base', values=value)
        return wide.reindex(order)

    def show(wide, title, fmt):
        wide = wide.rename(index=label, columns=short)
        wide.index.name = wide.columns.name = None
        print(f'\n=== {title} ===')
        print(wide.to_string(float_format=lambda v: format(v, fmt), na_rep='-'))

    seeds = sorted(df.seed.unique())
    print(f'\nPooled over seeds {seeds}; n = test flows of the family summed over the seeds; * = fewer than {min_rows} flows (indicative only).')
    print('Attack family: % of its flows detected. Benign family: % classified as benign. '
          + ', '.join(f'{short[b]} = {EXPERIMENTS[b][0]}' for b in bases))
    show(table('qat', 'accuracy'), 'INT8 (QAT) model: % of the family classified correctly', '.2f')
    show(table('normal', 'accuracy') - table('qat', 'accuracy'),
         'What quantization costs: FP32 minus INT8, in points of the family (positive = INT8 worse)', '+.2f')
    show(table('qat', 'errors'), 'INT8 (QAT) model: misclassified flows (summed over the seeds)', '.0f')


def main():
    parser = argparse.ArgumentParser(description='Accuracy of the FP32 and INT8 checkpoints per attack family')
    parser.add_argument('--predictor_path', type=str, required=True, help='AutoGluon predictor (only to rebuild the architecture of the replica)')
    parser.add_argument('--data_dir_template', type=str, required=True, help='one split per seed, e.g. <data>/balanced_binary/split_{seed}')
    parser.add_argument('--logs_dir', type=str, required=True, help='folder with the *.pth checkpoints and multiseed_exp<N>_results.csv')
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--experiments', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8], choices=sorted(EXPERIMENTS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--important_features_file', type=str, default=None, help='needed by Experiment 5 (written by select_features.py)')
    parser.add_argument('--scale_factor', type=float, default=1e6)
    parser.add_argument('--engine', type=str, default=DEFAULT_ENGINE, help='quantized engine that runs the INT8 models (default: the one of every INT8 number)')
    parser.add_argument('--min_rows', type=int, default=100, help='families with fewer pooled flows are marked with * in the tables')
    args = parser.parse_args()
    if any(base in SELECTS_FEATURES for base in args.experiments) and not args.important_features_file:
        parser.error('Experiment 5 needs --important_features_file')

    print(f'quantized engine: {set_quantized_engine(args.engine)}')
    allow_cpu_loading()
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(args.predictor_path, require_version_match=False, require_py_version_match=False)
    first_dir = args.data_dir_template.format(seed=args.seeds[0])
    n_features = pd.read_csv(os.path.join(first_dir, 'test.csv'), nrows=1).shape[1] - len(FEATURE_EXCLUDED)
    architecture, _, input_feature_size = get_model_architecture(predictor, n_features)

    rows, mismatches = [], 0
    for base in args.experiments:
        recipe = input_recipe(base)
        logged_path = os.path.join(args.logs_dir, f'multiseed_exp{base}_results.csv')
        logged = pd.read_csv(logged_path).set_index('seed') if os.path.exists(logged_path) else None
        if logged is None:
            print(f'WARNING: {logged_path} not found: the accuracy of {EXPERIMENTS[base][0]} cannot be checked against the run', file=sys.stderr)
        for seed in args.seeds:
            features, labels, families = load_test_inputs(
                args.data_dir_template.format(seed=seed), recipe, args.scale_factor, args.important_features_file)
            classes = np.where(labels == 1, 'attack', 'benign')
            for kind in ('normal', 'qat'):
                path = os.path.join(args.logs_dir, f'{kind}_pytorch_model_exp{experiment_number(base, seed)}.pth')
                model = try_load_checkpoint(path, kind, architecture, input_feature_size)
                if model is None:
                    print(f'WARNING: {path} not found, skipped', file=sys.stderr)
                    continue
                hit = predict(model, features) == labels
                accuracy = 100 * hit.mean()
                if logged is not None and seed in logged.index:
                    expected = logged.loc[seed, f'{kind}_test_accuracy']
                    if abs(accuracy - expected) > 0.011:
                        mismatches += 1
                        print(f'WARNING: {EXPERIMENTS[base][0]} seed {seed} {kind}: recomputed accuracy {accuracy:.2f}% != logged {expected:.2f}%',
                              file=sys.stderr)
                frame = pd.DataFrame({'family': families, 'class': classes, 'hit': hit})
                for (family, cls), group in frame.groupby(['family', 'class']):
                    rows.append({'experiment_base': base, 'paper_experiment': EXPERIMENTS[base][0], 'seed': seed, 'model': kind,
                                 'family': family, 'class': cls, 'n': len(group), 'correct': int(group.hit.sum())})
            print(f'{EXPERIMENTS[base][0]} seed {seed}: done', flush=True)

    df = pd.DataFrame(rows)
    df['accuracy'] = 100 * df['correct'] / df['n']
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f'\nSaved {len(df)} rows to {args.output_csv} ({mismatches} accuracy mismatches vs. the logs)')
    if df.empty:
        raise SystemExit('no checkpoint was found')
    summarize(df, [b for b in args.experiments if b in set(df.experiment_base)], args.min_rows)


if __name__ == '__main__':
    main()
