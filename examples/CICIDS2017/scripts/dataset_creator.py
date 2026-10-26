import functools
import os
import re
from multiprocessing.pool import ThreadPool

import click
import numpy as np
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          set_seed)

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")

def get_dataset_parameters(filename):
    filename_lower = filename.lower()  # dataset names are matched case-insensitively
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

def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
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
        and (df[type_column] == benign_label).all())

    if max_rows_per_file is not None and len(df) > max_rows_per_file and not is_purely_benign:
        # Subsample large per-class files while reading, instead of after - keeps
        # peak memory bounded for datasets released as one huge file per class
        # (e.g. CICIoT2023's multi-GB DDoS-flood captures), which get downsampled
        # to the minority class size a few steps later anyway (see `is_balanced`).
        df = df.sample(n=max_rows_per_file, random_state=seed)

    return df

def create(
        directory: str, test_size: float, finetune_size: float, validation_size: float,
        composition: str, problem_type: str, cpus: int, keep_attack_type: bool = False,
        max_rows_per_file: int = None):
    global logger
    seed = set_seed()
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

    logger.info(f'Checking contents of directory: {directory}')
    files = find_csv_files(directory)

    if not files:
        logger.error(f'No CSV files found in directory: {directory}')
        raise FileNotFoundError(f'No CSV files found in directory: {directory}')

    logger.info(f'Found {len(files)} CSV files in directory: {directory}')
    logger.debug(f'CSV files: {files}')

    logger.info(f'Loading {len(files)} dataframes'
                + (f' (capped at {max_rows_per_file} rows/file)' if max_rows_per_file else ''))
    df = pd.DataFrame()
    reader = functools.partial(
        read_csv_in_chunks, type_column=type_column, max_rows_per_file=max_rows_per_file,
        seed=seed, benign_label=benign_label)
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(files)) as pbar:
        for x in pool.imap(reader, files):
            logger.debug(f'Read dataframe with columns: {x.columns.tolist()} and shape: {x.shape}')
            df = pd.concat((df, x), ignore_index=True)
            pbar.update()

    if df.empty:
        logger.error('Concatenated dataframe is empty after reading CSV files.')
        raise ValueError('Concatenated dataframe is empty after reading CSV files.')

    logger.info(f'Dropping columns {excluded_columns}')
    df.drop(excluded_columns, axis=1, errors='ignore', inplace=True)

    logger.info(f'Shape after dropping columns: {df.shape}')

    logger.info(f'Dropping columns with only 1 value')
    df.dropna(thresh=2, axis=1, inplace=True)

    logger.info(f'Shape after dropping single-value columns: {df.shape}')

    logger.info("Stripping leading/trailing spaces from column names")
    df.columns = df.columns.astype(str).str.strip()

    logger.info(f"Renaming {type_column} to 'Label'")
    df.rename({type_column: 'Label'}, axis=1, inplace=True)
    type_column = 'Label'

    logger.info(f"Columns after renaming: {df.columns.tolist()}")

    if type_column not in df.columns:
        logger.error(f"Column {type_column} not found in dataframe columns: {df.columns.tolist()}")
        raise KeyError(f"Column {type_column} not found in dataframe")

    attack_type_categories = None
    attack_type_code_col = None
    if keep_attack_type:
        # Snapshot the original (pre-binarization) fine-grained label as an integer
        # code column, *before* any binarization/numeric-coercion touches `Label`.
        # Held OUTSIDE df (not as a df column) until after Min-Max normalization,
        # re-attached (index-aligned) right before balancing - see below. It must
        # NOT sit inside df across that stretch: `Transforming columns to numeric`
        # is a no-op on it (already numeric), but `Min-Max normalization` is not -
        # it would rescale these integer category codes into floats in [0, 1] and
        # silently corrupt them (this crashed IndexError the first time round: the
        # "codes" used to index into attack_type_categories were no longer ints).
        # Row-count-changing steps (dropna) are fine to let it sit out for, since
        # pandas' index-aligned assignment on re-attach naturally drops whatever
        # rows dropna removed.
        logger.info('Recording original per-row attack type (--keep-attack-type) for later per-class evaluation')
        codes, attack_type_categories = pd.factorize(df[type_column].astype(str))
        attack_type_code_col = pd.Series(codes, index=df.index, name='_attack_type_code')

    if is_binary:
        logger.info(f'Binary problem, changing to 1 all columns different from {benign_label}')
        df[type_column] = df[type_column].ne(benign_label).mul(1)

    logger.info(f'Shape after binary conversion: {df.shape}')

    logger.info(f'Transforming columns to numeric')
    df = df.apply(pd.to_numeric, errors='coerce')

    logger.info(f'Shape after transforming to numeric: {df.shape}')

    cat_columns = [col_name for col_name,
                   dtype in df.dtypes.items() if dtype == object and col_name != type_column]
    if cat_columns:
        logger.info(f'Transforming {cat_columns} categorical columns also to numeric')
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df[cat_columns] = df[cat_columns].astype('int')

    logger.info(f'Shape after transforming categorical columns: {df.shape}')

    logger.info(f'Removing rows with NaN values')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    logger.info(f'Shape after removing NaN values: {df.shape}')

    logger.info(f'Min-Max normalization')
    df = df.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if np.issubdtype(x.dtype, np.number) else x)

    logger.info(f'Shape after Min-Max normalization: {df.shape}')

    if keep_attack_type:
        # Re-attach now (index-aligned - see the long comment above for why it was
        # held out), so it rides along as a normal column through balancing/
        # splitting like everything else from here on.
        df['_attack_type_code'] = attack_type_code_col

    if is_balanced:
        logger.info(f'Balanced problem, picking a maximum samples of the minor class')
        g = df.groupby(type_column)
        min_class_size = g.size().min()
        if min_class_size == 0:
            logger.error('Not enough samples in the minority class to balance the dataset.')
            raise ValueError('Not enough samples in the minority class to balance the dataset.')
        df = g.apply(lambda x: x.sample(min_class_size, replace=False)).reset_index(drop=True)

    logger.info(f'Shape after balancing: {df.shape}')

    if df.empty:
        logger.error('DataFrame is empty after preprocessing steps.')
        raise ValueError('DataFrame is empty after preprocessing steps.')

    # Feature generation should only ever see genuine input features - pull the
    # attack-type bookkeeping column out first and re-attach it (index-aligned)
    # afterwards.
    attack_type_code_col = df.pop('_attack_type_code') if keep_attack_type else None

    a = AutoMLPipelineFeatureGenerator()
    df = a.fit_transform(df)

    if keep_attack_type:
        df['_attack_type_code'] = attack_type_code_col

    logger.info(f'Shape after feature generation: {df.shape}')

    logger.info('First, splitting dataset into train and test')
    big_train, big_test, _, _ = train_test_split(df, df[type_column],
                                                 test_size=test_size,
                                                 shuffle=True,
                                                 stratify=df[type_column])
    logger.info(f'Shape of train data: {big_train.shape}, Shape of test data: {big_test.shape}')

    logger.info('Now sub-splitting train into train and validation')
    train_data, validation_data, _, _ = train_test_split(big_train, big_train[type_column],
                                                         test_size=validation_size,
                                                         shuffle=True,
                                                         stratify=big_train[type_column])
    logger.info(f'Shape of train data: {train_data.shape}, Shape of validation data: {validation_data.shape}')

    logger.info('Now sub-splitting test into test and finetune')
    test_data, finetune_data, _, _ = train_test_split(big_test, big_test[type_column],
                                                      test_size=finetune_size,
                                                      shuffle=True,
                                                      stratify=big_test[type_column])
    logger.info(f'Shape of test data: {test_data.shape}, Shape of finetune data: {finetune_data.shape}')

    with tqdm(total=4) as pbar:
        for x, y in zip((train_data, validation_data, test_data, finetune_data),
                        ('train', 'validation', 'test', 'finetune')):
            x: pd.DataFrame
            logger.info(f'Storing {y} with label value counts {x[type_column].value_counts()}')
            x.index.name = 'ID'

            if keep_attack_type:
                # Pop the bookkeeping column back out so {y}.csv keeps its usual
                # schema, and write it to a separate, ID-aligned side file instead
                # (see scripts/evaluate_per_attack_class.py).
                codes = x.pop('_attack_type_code')
                attack_types = pd.Series(
                    attack_type_categories[codes.values], index=x.index, name='AttackType')
                attack_types.to_csv(os.path.join(store_path, f'{y}_attack_types.csv'))

            x.to_csv(os.path.join(store_path, f'{y}.csv'))
            pbar.update()

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
@click.option('--keep-attack-type', is_flag=True, default=False,
              help='also write {split}_attack_types.csv with the original (pre-binarization) '
                   'per-row attack label, ID-aligned with {split}.csv, for per-attack-class '
                   'evaluation (see scripts/evaluate_per_attack_class.py). Does not change '
                   'the schema of {split}.csv itself.')
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
