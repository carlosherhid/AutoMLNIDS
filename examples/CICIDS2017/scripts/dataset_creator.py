import functools
import json
import os
import re
from multiprocessing.pool import ThreadPool

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          set_seed)

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def _cicids2017_improved_benign(labels):
    """BENIGN plus the flows that the corrected release labels '<attack> - Attempted': they belong to an attack that was
    launched but carry no malicious payload, and the authors recommend treating them as benign."""
    s = labels.astype(str).str.strip()
    return s.eq('BENIGN') | s.str.lower().str.endswith('- attempted')


def get_dataset_parameters(filename):
    """Label column, benign value(s) and columns to drop for each dataset, matched on the (absolute) path of the data
    directory. The benign value can be one label, a collection of labels or a function that returns a boolean mask."""
    filename_lower = filename.lower()  # dataset names are matched case-insensitively
    normalized = filename_lower.replace('-', '_')
    if 'cicids2017_improved' in normalized:   # must be checked before 'cicids2017'
        return 'Label', _cicids2017_improved_benign, ['id', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port',
                                                       'Timestamp', 'Attempted Category']
    if 'unsw_nb15' in normalized:
        return 'label', 0, ['id', 'attack_cat']
    if 'ton_iot' in normalized:
        return 'label', 0, ['type', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']
    if 'cicids2017' in filename_lower:
        return 'Label', 'BENIGN', ['Destination Port']
    if 'cicids2019' in filename_lower:
        return 'Label', 'BENIGN', ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
                                   'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP']
    if 'ciciot2023' in filename_lower:
        return 'label', 'BenignTraffic', ['Protocol Type', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                                          'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    if 'edge2022' in filename_lower:
        return 'Attack_type', 'Normal', [
            'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'tcp.srcport',
            'tcp.dstport', 'udp.port', 'http.tls_port', 'tcp.options', 'http.file_data', 'tcp.payload', 'mqtt.msg',
            'http.request.uri.query', 'http.request.full_uri', 'icmp.transmit_timestamp', 'Attack_label']
    if 'ics-d1' in filename_lower:
        return 'marker', 'Natural', []
    if 'ics-d2' in filename_lower:
        return 'Label', 'Good', []
    if 'ics-d3' in filename_lower:
        return 'result', 0, ['time']
    raise ValueError(filename)


def get_attack_column(filename):
    """Column with the attack type when it is not the label column itself (used by --keep-attack-type)."""
    normalized = filename.lower().replace('-', '_')
    if 'unsw_nb15' in normalized:
        return 'attack_cat'
    if 'ton_iot' in normalized:
        return 'type'
    return None


def is_benign(labels, benign):
    if callable(benign):
        return benign(labels)
    if isinstance(benign, (list, tuple, set, frozenset)):
        return labels.isin(list(benign))
    return labels.eq(benign)


def find_csv_files(directory):
    """The CSV files under `directory`, in alphabetical order. os.walk lists them in the order of the file system, which differs from one machine
    to another; the files are concatenated in this order, so the same seed would give different splits on two machines (see --file-order)."""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        csv_files.extend(os.path.join(root, file) for file in sorted(files) if file.endswith('.csv'))
    return csv_files

def derive_label_from_filename(file):
    """Some datasets (e.g. CICIoT2023's official per-attack-type CSV release) split
    each class into its own file with no label column at all - the file's own name
    (minus a trailing per-file index and its extension) *is* the label, e.g.
    'DDoS-PSHACK_Flood10.pcap.csv' -> 'DDoS-PSHACK_Flood'."""
    base = os.path.basename(file)
    base = re.sub(r'\.pcap\.csv$|\.csv$', '', base, flags=re.IGNORECASE)
    return re.sub(r'\d+$', '', base)

def read_csv_in_chunks(file, chunksize=10000, type_column=None, max_rows_per_file=None, seed=0, benign_label=None):
    df_chunks = pd.read_csv(file, chunksize=chunksize, low_memory=False)
    df = pd.concat(df_chunks, ignore_index=True)
    df.columns = df.columns.astype(str).str.strip()
    type_column = type_column.strip() if type_column is not None else None

    if type_column is not None and type_column not in df.columns:
        label = derive_label_from_filename(file)
        logger.debug(f'{file}: no {type_column!r} column found, using filename-derived label {label!r}')
        df[type_column] = label

    # Never cap a file that is *entirely* the benign class: for datasets split one
    # file per class (see derive_label_from_filename above), benign traffic is
    # naturally the minority class that the final `is_balanced` step sizes the
    # whole dataset around - throttling it here would shrink the achievable
    # dataset size for no reason. Only the (much larger, oversupplied) attack
    # files actually need capping to bound peak memory.
    is_purely_benign = (
        benign_label is not None and type_column in df.columns
        and is_benign(df[type_column], benign_label).all())

    if max_rows_per_file is not None and len(df) > max_rows_per_file and not is_purely_benign:
        # Subsample large per-class files while reading, instead of after - keeps
        # peak memory bounded for datasets released as one huge file per class
        # (e.g. CICIoT2023's multi-GB DDoS-flood captures), which get downsampled
        # to the minority class size a few steps later anyway (see `is_balanced`).
        df = df.sample(n=max_rows_per_file, random_state=seed)

    return df


def duplicate_columns(df, columns):
    """Columns that are an exact copy of an earlier column (a cheap fingerprint first, the full comparison only on a match)."""
    seen, duplicated = {}, []
    for c in columns:
        values = df[c].to_numpy()
        key = (float(np.nansum(values)), float(np.nanmin(values)), float(np.nanmax(values)))
        if any(np.array_equal(df[other].to_numpy(), values) for other in seen.get(key, [])):
            duplicated.append(c)
        else:
            seen.setdefault(key, []).append(c)
    return duplicated


def create(
        directory: str, test_size: float, finetune_size: float, validation_size: float,
        composition: str, problem_type: str, cpus: int, keep_attack_type: bool = False,
        max_rows_per_file: int = None, seeds: str = '0', dedup: bool = True, conflict_policy: str = 'drop',
        onehot: str = '', onehot_max_levels: int = 20, exclude_attack: str = '', official_split: bool = False,
        drop_columns: str = '', file_order: str = ''):
    """Cleans the raw files once and then writes one train/validation/test/finetune split per seed to
    <directory>/../<composition>_<problem_type>/split_<seed>/. Everything that learns from the data (the Min-Max scaling) is fitted on the
    training part of each split only; exact duplicates are removed before splitting, so that no test row has an identical copy in the
    training data. Feature vectors that appear with both labels are either removed entirely (conflict_policy='drop') or kept once with
    their majority label (conflict_policy='majority'; exact ties are removed). Text columns named in `onehot` (e.g. 'proto,conn_state')
    are coded as 0/1 indicator columns for their `onehot_max_levels` most frequent values (the rest go to 'other'); the other text
    columns are dropped. `exclude_attack` removes every row of the named attack families (comma-separated, matched case-insensitively on the
    attack-type column, a trailing * matches a prefix) before anything else. With `official_split` the dataset must come as an official
    training file and an official testing file (file names containing 'train' and 'test'): duplicates are removed inside each of them, the
    official test set is kept as the test split (not balanced, not subsampled) and validation and fine-tuning splits are taken from the
    training set. `drop_columns` removes the named columns (comma-separated, after stripping the spaces of the header) on top of the ones the
    dataset always excludes; unlike those it raises if a name is not found, so that a sensitivity variant (e.g. UNSW-NB15 without `sttl` and
    `ct_state_ttl`) cannot silently run with the column still in."""
    global logger
    seed = set_seed()
    split_seeds = [int(s) for s in str(seeds).replace(',', ' ').split()]
    is_balanced = composition == 'balanced'
    is_binary = problem_type == 'binary'

    store_path = os.path.join(directory, os.pardir, f'{composition}_{problem_type}')
    create_dir(store_path, overwrite=False)

    logger.info(f'Setting variable for dataset {directory}')
    # Match against the absolute path, not the literal --directory string, so this
    # works regardless of which relative path the caller happens to pass (e.g.
    # `--directory ./original` run from inside examples/CICIoT2023/ still resolves
    # to a path containing "CICIoT2023").
    type_column, benign_label, excluded_columns = get_dataset_parameters(os.path.abspath(directory))
    attack_column = get_attack_column(os.path.abspath(directory))

    logger.info(f'Checking contents of directory: {directory}')
    files = find_csv_files(directory)

    if not files:
        logger.error(f'No CSV files found in directory: {directory}')
        raise FileNotFoundError(f'No CSV files found in directory: {directory}')

    if file_order:
        # the order of the files decides the row order, and with it the splits of a seed: this reproduces the order of a previous run
        # (split_info.json records it under "files_in_reading_order")
        wanted = [name.strip() for name in file_order.split(',') if name.strip()]
        by_name = {os.path.basename(file): file for file in files}
        if len(by_name) != len(files) or sorted(wanted) != sorted(by_name):
            raise ValueError(f'--file-order must name every CSV file of the directory exactly once: found {sorted(by_name)}, given {wanted}')
        files = [by_name[name] for name in wanted]

    logger.info(f'Found {len(files)} CSV files in directory: {directory}')
    logger.debug(f'CSV files: {files}')

    logger.info(f'Loading {len(files)} dataframes'
                + (f' (capped at {max_rows_per_file} rows/file)' if max_rows_per_file else ''))
    def official_part_of(file):
        name = os.path.basename(file).lower()
        if ('train' in name) == ('test' in name):
            raise ValueError(f'--official-split needs file names with exactly one of "train" or "test", got {os.path.basename(file)!r}')
        return 'train' if 'train' in name else 'test'

    if official_split:
        for file in files:
            official_part_of(file)
    df = pd.DataFrame()
    reader = functools.partial(
        read_csv_in_chunks, type_column=type_column, max_rows_per_file=max_rows_per_file,
        seed=seed, benign_label=benign_label)
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(files)) as pbar:
        for file, x in zip(files, pool.imap(reader, files)):
            logger.debug(f'Read dataframe with columns: {x.columns.tolist()} and shape: {x.shape}')
            if official_split:
                x['_official_part'] = official_part_of(file)
            df = pd.concat((df, x), ignore_index=True)
            pbar.update()

    if df.empty:
        logger.error('Concatenated dataframe is empty after reading CSV files.')
        raise ValueError('Concatenated dataframe is empty after reading CSV files.')

    # Names are stripped first so that the exclusions below also match columns written with a leading space
    # (the raw CICIDS2017 files have ' Destination Port').
    df.columns = df.columns.astype(str).str.strip()
    type_column = type_column.strip()
    excluded_columns = [c.strip() for c in excluded_columns]
    extra_columns = [c.strip() for c in str(drop_columns).split(',') if c.strip()]
    missing_columns = [c for c in extra_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f'--drop-columns {missing_columns} not found among the columns of the data: {df.columns.tolist()}')
    excluded_columns = excluded_columns + extra_columns
    official_parts = df.pop('_official_part') if official_split else None

    attack_type_categories = None
    attack_type_code_col = None
    if keep_attack_type:
        logger.info('Recording original per-row attack type (--keep-attack-type) for later per-class evaluation')
        source = attack_column if attack_column in df.columns else type_column
        codes, attack_type_categories = pd.factorize(df[source].astype(str))
        attack_type_code_col = pd.Series(codes, index=df.index, name='_attack_type_code')

    excluded_attacks = {}
    if str(exclude_attack).strip():
        source = attack_column if attack_column in df.columns else type_column
        values = df[source].astype(str).str.strip().str.lower()
        drop_mask = pd.Series(False, index=df.index)
        for name in [n.strip().lower() for n in str(exclude_attack).split(',') if n.strip()]:
            hit = values.str.startswith(name[:-1]) if name.endswith('*') else values.eq(name)
            excluded_attacks[name] = int(hit.sum())
            if not hit.any():
                raise ValueError(f'--exclude-attack {name!r} matches no row of column {source!r}')
            drop_mask |= hit
        df = df[~drop_mask]
        if official_parts is not None:
            official_parts = official_parts[~drop_mask]
        logger.info(f'Excluded attack families {excluded_attacks} (column {source!r}); {len(df)} rows remain')

    logger.info(f'Dropping columns {excluded_columns}')
    df.drop(excluded_columns, axis=1, errors='ignore', inplace=True)

    logger.info(f'Shape after dropping columns: {df.shape}')

    logger.info(f'Dropping columns with only 1 value')
    df.dropna(thresh=2, axis=1, inplace=True)

    logger.info(f'Shape after dropping single-value columns: {df.shape}')

    logger.info(f"Renaming {type_column} to 'Label'")
    df.rename({type_column: 'Label'}, axis=1, inplace=True)
    type_column = 'Label'

    logger.info(f"Columns after renaming: {df.columns.tolist()}")

    if type_column not in df.columns:
        logger.error(f"Column {type_column} not found in dataframe columns: {df.columns.tolist()}")
        raise KeyError(f"Column {type_column} not found in dataframe")

    if is_binary:
        logger.info(f'Binary problem, changing to 1 every label that is not benign')
        df[type_column] = (~is_benign(df[type_column], benign_label)).astype(int)

    logger.info(f'Shape after binary conversion: {df.shape}')

    # Text columns (protocol names, hosts, user agents...) cannot be fed to the network: they are dropped, except the ones asked for in
    # `onehot`, which become indicator columns (levels chosen by frequency only, never by label)
    onehot_levels = {}
    onehot_columns = [c.strip() for c in str(onehot).replace(',', ' ').split() if c.strip()]
    indicators = []
    for c in onehot_columns:
        if c not in df.columns:
            raise KeyError(f'--onehot column {c!r} not found; available: {df.columns.tolist()}')
        values = df[c].astype(str).str.strip()
        levels = values.value_counts().index[:onehot_max_levels].tolist()
        onehot_levels[c] = levels
        indicators.append(pd.get_dummies(values.where(values.isin(levels), 'other'), prefix=c).astype('int8'))
        df.drop(columns=c, inplace=True)
    if indicators:
        df = pd.concat([df] + indicators, axis=1)
        logger.info(f'One-hot coded {onehot_columns}: ' + ', '.join(f'{c}: {len(v)} levels' for c, v in onehot_levels.items()))
    for c in [c for c in df.columns if c != type_column and df[c].dtype == bool]:
        df[c] = df[c].astype('int8')
    text_columns = [c for c in df.columns if c != type_column and df[c].dtype == object
                    and pd.to_numeric(df[c], errors='coerce').notna().mean() < 0.9]
    if text_columns:
        logger.info(f'Dropping non-numeric columns {text_columns}')
        df.drop(columns=text_columns, inplace=True)

    logger.info(f'Transforming columns to numeric')
    df = df.apply(pd.to_numeric, errors='coerce')

    logger.info(f'Shape after transforming to numeric: {df.shape}')

    logger.info(f'Removing rows with NaN values')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    logger.info(f'Shape after removing NaN values: {df.shape}')

    feature_columns = [c for c in df.columns if c != type_column]
    constant_columns = [c for c in feature_columns if df[c].nunique(dropna=False) <= 1]
    feature_columns = [c for c in feature_columns if c not in constant_columns]
    copy_columns = duplicate_columns(df, feature_columns)
    feature_columns = [c for c in feature_columns if c not in copy_columns]
    df = df[feature_columns + [type_column]]
    logger.info(f'Dropped {len(constant_columns)} constant columns {constant_columns} and {len(copy_columns)} copies of another '
                f'column {copy_columns}; {len(feature_columns)} features remain')

    if official_parts is not None:
        official_parts = official_parts.loc[df.index]

    def deduplicate(frame):
        """Exact duplicates out; feature vectors that appear with both labels out (conflict_policy='drop') or kept once with the
        majority label (conflict_policy='majority'). Returns the kept rows and the number of conflicting vectors."""
        hashes = pd.util.hash_pandas_object(frame[feature_columns], index=False).to_numpy()
        labels = frame[type_column].to_numpy()
        conflicts = np.intersect1d(hashes[labels == 0], hashes[labels == 1])
        if conflict_policy == 'drop':
            keep = ~pd.Series(hashes).duplicated().to_numpy() & ~np.isin(hashes, conflicts)
            note = 'and are removed entirely'
        elif conflict_policy == 'majority':
            # a conflicting vector stays once, with the label of the majority of its copies; exact ties are removed
            share = pd.Series(labels).groupby(hashes).transform('mean').to_numpy()
            tie = np.isin(hashes, conflicts) & (share == 0.5)
            majority_label = (share > 0.5).astype(labels.dtype)
            candidate = ~tie & (labels == majority_label)
            first = ~pd.Series(hashes).where(candidate).duplicated().to_numpy()
            keep = candidate & first
            note = f'each is kept once with its majority label, {int(len(np.unique(hashes[tie])))} exact ties are removed'
        else:
            raise ValueError(f'unknown conflict policy {conflict_policy!r}')
        logger.info(f'Deduplication: {len(frame)} -> {int(keep.sum())} rows ({len(conflicts)} feature vectors appear with both labels; {note})')
        return frame[keep], int(len(conflicts))

    rows_before_dedup = len(df)
    conflicting_hashes = 0
    official_info = None
    if dedup:
        if official_parts is None:
            df, conflicting_hashes = deduplicate(df)
        else:
            kept = []
            for name in ('train', 'test'):
                frame, n_conflicts = deduplicate(df[official_parts == name])
                kept.append(frame)
                conflicting_hashes += n_conflicts
            df = pd.concat(kept)
            official_parts = official_parts.loc[df.index]
    if official_parts is not None:
        train_hashes = pd.util.hash_pandas_object(df.loc[official_parts == 'train', feature_columns], index=False)
        test_hashes = pd.util.hash_pandas_object(df.loc[official_parts == 'test', feature_columns], index=False)
        twins = float(test_hashes.isin(set(train_hashes.to_numpy())).mean())
        official_info = {'train_rows': int((official_parts == 'train').sum()), 'test_rows': int((official_parts == 'test').sum()),
                         'test_rows_with_identical_twin_in_train': twins}
        logger.info(f'Official split: {official_info}')

    if keep_attack_type:
        df['_attack_type_code'] = attack_type_code_col

    if df.empty:
        logger.error('DataFrame is empty after preprocessing steps.')
        raise ValueError('DataFrame is empty after preprocessing steps.')

    for split_seed in split_seeds:
        logger.info(f'=== split with seed {split_seed} ===')
        if official_parts is not None:
            train_pool = df[official_parts == 'train']
            if is_balanced:
                min_class_size = train_pool.groupby(type_column).size().min()
                train_pool = pd.concat([g.sample(min_class_size, replace=False, random_state=split_seed)
                                        for _, g in train_pool.groupby(type_column)])
            train_pool = train_pool.reset_index(drop=True)
            logger.info(f'Official training set after balancing: {train_pool.shape}; the official test set is kept as it is')
            rest, finetune_data = train_test_split(train_pool, test_size=0.15, shuffle=True,
                                                   stratify=train_pool[type_column], random_state=split_seed)
            train_data, validation_data = train_test_split(rest, test_size=validation_size, shuffle=True,
                                                           stratify=rest[type_column], random_state=split_seed)
            # the official test file keeps its own order (possibly sorted by class): shuffle it so that the row IDs written below carry no information
            test_data = df[official_parts == 'test'].sample(frac=1, random_state=split_seed).reset_index(drop=True)
        else:
            part = df
            if is_balanced:
                logger.info(f'Balanced problem, picking a maximum samples of the minor class')
                min_class_size = part.groupby(type_column).size().min()
                if min_class_size == 0:
                    logger.error('Not enough samples in the minority class to balance the dataset.')
                    raise ValueError('Not enough samples in the minority class to balance the dataset.')
                part = pd.concat([g.sample(min_class_size, replace=False, random_state=split_seed)
                                  for _, g in part.groupby(type_column)]).reset_index(drop=True)
            else:
                part = part.reset_index(drop=True)
            logger.info(f'Shape after balancing: {part.shape}')

            logger.info('First, splitting dataset into train and test')
            big_train, big_test = train_test_split(part, test_size=test_size, shuffle=True,
                                                   stratify=part[type_column], random_state=split_seed)
            logger.info('Now sub-splitting train into train and validation')
            train_data, validation_data = train_test_split(big_train, test_size=validation_size, shuffle=True,
                                                           stratify=big_train[type_column], random_state=split_seed)
            logger.info('Now sub-splitting test into test and finetune')
            test_data, finetune_data = train_test_split(big_test, test_size=finetune_size, shuffle=True,
                                                        stratify=big_test[type_column], random_state=split_seed)

        # Min-Max scaling with the statistics of the training part only; the other parts are clipped to [0, 1]
        scaler = MinMaxScaler().fit(train_data[feature_columns])
        split_dir = os.path.join(store_path, f'split_{split_seed}')
        os.makedirs(split_dir, exist_ok=True)
        counts = {}
        with tqdm(total=4) as pbar:
            for x, y in zip((train_data, validation_data, test_data, finetune_data),
                            ('train', 'validation', 'test', 'finetune')):
                # the rows of every part are already in random order (train_test_split shuffles); their IDs are renumbered from 0 so that they say
                # nothing about the class (the index of the balanced sample is sorted by class: with it, a threshold on ID alone gave 100 % accuracy)
                x = x.reset_index(drop=True)
                x[feature_columns] = np.clip(scaler.transform(x[feature_columns]), 0.0, 1.0)
                x.index.name = 'ID'
                counts[y] = {'rows': int(len(x)), 'attack_rows': int(x[type_column].sum())}
                logger.info(f'Storing {y} with label value counts {x[type_column].value_counts().to_dict()}')

                if keep_attack_type:
                    # Pop the bookkeeping column back out so {y}.csv keeps its usual
                    # schema, and write it to a separate, ID-aligned side file instead
                    # (read by per_family_recall.py, which gives the accuracy per attack family).
                    codes = x.pop('_attack_type_code')
                    attack_types = pd.Series(
                        attack_type_categories[codes.values], index=x.index, name='AttackType')
                    attack_types.to_csv(os.path.join(split_dir, f'{y}_attack_types.csv'))

                x[feature_columns + [type_column]].to_csv(os.path.join(split_dir, f'{y}.csv'))
                pbar.update()

        info = {'seed': split_seed, 'directory': os.path.basename(os.path.abspath(directory)), 'composition': composition,
                'files_in_reading_order': [os.path.basename(file) for file in files],
                'sizes': {'test_size': test_size, 'finetune_size': finetune_size, 'validation_size': validation_size},
                'splits': counts, 'features': feature_columns,
                'dropped': {'excluded': excluded_columns, 'text_columns': text_columns, 'constant': constant_columns,
                            'copies_of_another_column': copy_columns},
                'deduplication': {'enabled': dedup, 'rows_before': rows_before_dedup, 'rows_after': int(len(df)),
                                  'feature_vectors_with_both_labels': conflicting_hashes, 'conflict_policy': conflict_policy},
                'onehot': onehot_levels, 'excluded_attacks': excluded_attacks, 'drop_columns': extra_columns, 'official_split': official_info,
                'min_max_scaler': {'fitted_on': 'train', 'data_min': scaler.data_min_.tolist(), 'data_max': scaler.data_max_.tolist()}}
        with open(os.path.join(split_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

@click.command(help='Create and adjust the final dataset', context_settings={'show_default': True}, name='create')
@click.option('--directory', type=str, required=True,
              help='working directory with the original dataset')
@click.option('--test-size', type=float, default=0.3,
              help='size of the test portion (the remaining is left for training)')
@click.option('--finetune-size', type=float, default=0.5,
              help='size of the finetune portion inside the test one')
@click.option('--validation-size', type=float, default=0.2,
              help='size of the validation portion inside the training one')
@click.option('--composition', type=click.Choice(['balanced', 'imbalanced'], case_sensitive=False),
              help='whether to be a balanced or imbalanced problem', default='balanced')
@click.option('--problem-type', type=click.Choice(['binary', 'multiclass'], case_sensitive=False),
              help='set the problem to binary classification', default='binary')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--seeds', type=str, default='0',
              help='comma-separated seeds; one split (and one balancing draw) is written to split_<seed>/ for each')
@click.option('--dedup/--no-dedup', default=True,
              help='remove exact duplicate rows, and the rows whose features appear with both labels, before splitting')
@click.option('--conflicts', 'conflict_policy', type=click.Choice(['drop', 'majority']), default='drop',
              help='feature vectors that appear with both labels: remove them entirely, or keep one copy with the majority label')
@click.option('--onehot', type=str, default='',
              help='comma-separated text columns to code as 0/1 indicator columns instead of dropping them (e.g. "proto,conn_state")')
@click.option('--onehot-max-levels', type=int, default=20,
              help='levels kept per --onehot column (the most frequent ones); the rest are coded as "other"')
@click.option('--exclude-attack', type=str, default='',
              help='comma-separated attack families to remove entirely (matched case-insensitively on the attack-type column; a trailing * '
                   'matches a prefix), e.g. "DoS Hulk"')
@click.option('--drop-columns', type=str, default='',
              help='comma-separated columns to remove on top of the ones the dataset always excludes (raises if one is not found), '
                   'e.g. "sttl,ct_state_ttl" for the UNSW-NB15 variant without the TTL variables')
@click.option('--official-split', is_flag=True, default=False,
              help='the directory holds an official training file and an official testing file (names containing "train" and "test"): '
                   'the official test set becomes the test split (not balanced) and validation/finetune come from the training set')
@click.option('--keep-attack-type', is_flag=True, default=False,
              help='also write {split}_attack_types.csv with the original (pre-binarization) '
                   'per-row attack label, ID-aligned with {split}.csv, for per-attack-class '
                   'evaluation. Does not change the schema of {split}.csv itself.')
@click.option('--file-order', type=str, default='',
              help='comma-separated names of the CSV files, in the order in which they are read. By default they are read in alphabetical order. '
                   'The order decides the row order and therefore the splits of a seed: give the "files_in_reading_order" of the split_info.json '
                   'of an earlier run to reproduce its splits exactly.')
@click.option('--max-rows-per-file', type=int, default=None,
              help='randomly subsample each input CSV to at most this many rows while reading, '
                   'to bound peak memory for datasets released as one huge file per class '
                   '(e.g. CICIoT2023). The final dataset is downsampled to the minority class '
                   'size a few steps later regardless when --composition balanced is used, so '
                   'this mainly matters for how much RAM the intermediate concatenation needs.')
def create_command(*args, **kwargs):
    create(*args, **kwargs)

if __name__ == '__main__':
    create_command()
