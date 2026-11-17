"""
Readers of the result files of one dataset, shared by the table scripts (build_metrics_table.py, build_preprocessing_table.py).

A dataset is given as NAME=ROOT[,MODELS_DIR]:
  ROOT        <data>/<dataset>/balanced_binary, the folder with split_<seed>/, baselines/split_<seed>/ and the runs
  MODELS_DIR  folder of ROOT with the runs and their multiseed_exp<N>_results.csv (default compressed_models; compressed_models_bnfold for
              the runs made with the BatchNorm kept and folded)
e.g.  CICIDS2017_improved=<data>/CICIDS2017_improved/balanced_binary  UNSW_NB15=<data>/UNSW_NB15/balanced_binary,compressed_models_bnfold
"""
import glob
import json
import os
import re

import pandas as pd

DEFAULT_MODELS_DIR = 'compressed_models'


def parse_dataset(spec):
    """'NAME=ROOT[,MODELS_DIR]' -> (name, root, models_dir)."""
    name, separator, rest = spec.partition('=')
    if not separator or not name or not rest:
        raise ValueError(f'a dataset is given as NAME=ROOT[,MODELS_DIR], got {spec!r}')
    root, _, models_dir = rest.partition(',')
    return name, root, models_dir or DEFAULT_MODELS_DIR


def load_multiseed(root, models_dir, experiment):
    """Per-seed results of one experiment (the CSV written by run_multiseed.py), sorted by seed; None if the file does not exist.
    The INT8 numbers (qat_*) are those of the engine recorded in the logs; the fractions (recall, fpr, f1...) are in [0, 1], the accuracies in %."""
    path = os.path.join(root, models_dir, f'multiseed_exp{experiment}_results.csv')
    if not os.path.exists(path):
        print(f'WARNING: {path} not found, experiment {experiment} skipped')
        return None
    return pd.read_csv(path).sort_values('seed').reset_index(drop=True)


def attack_share(root, seed):
    """Share of malicious flows of the test set of split_<seed> (from its split_info.json); None if it is not available."""
    path = os.path.join(root, f'split_{seed}', 'split_info.json')
    if not os.path.exists(path):
        return None
    test = json.load(open(path))['splits']['test']
    return test['attack_rows'] / test['rows']


def derive_fpr(accuracy, recall, share):
    """False-positive rate from accuracy, recall and the share of malicious flows of the test set:
    accuracy = share * recall + (1 - share) * (1 - fpr). It is the exact inverse for any prevalence (1 - 2 * accuracy + recall only for 50 %)."""
    return 1 - (accuracy - share * recall) / (1 - share)


def load_baselines(root):
    """The classical baselines of every split (baselines/split_<seed>/baselines_classical_ml_results.csv) in one table with a `split` column.
    Results written before the confusion counts were stored have no `fpr` column: it is derived from the accuracy, the recall and the real
    share of malicious flows of that split's test set. None if there are no baseline results."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, 'baselines', 'split_*', 'baselines_classical_ml_results.csv'))):
        seed = int(re.search(r'split_(\d+)', path).group(1))
        frame = pd.read_csv(path)
        frame['split'] = seed
        if 'fpr' not in frame:
            share = attack_share(root, seed)
            if share is None:
                print(f'WARNING: {path} has no fpr column and split_{seed}/split_info.json is missing: FPR not available for that split')
                frame['fpr'] = float('nan')
            else:
                frame['fpr'] = derive_fpr(frame['accuracy'], frame['recall'], share)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None
