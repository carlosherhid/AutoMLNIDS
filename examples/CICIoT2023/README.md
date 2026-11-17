# CICIoT2023

One of the four datasets of the paper, run with exactly the same experiments as the others. It reuses the scripts of [`../CICIDS2017/scripts/`](../CICIDS2017/scripts): every script
takes its data and output directories as arguments, so nothing is duplicated.

[`dataset_creator.py`](../CICIDS2017/scripts/dataset_creator.py) recognizes CICIoT2023 from the directory name
(`get_dataset_parameters()`: label column `label`, benign value `BenignTraffic`, one-hot protocol-flag columns
excluded). Several repackaged versions of this dataset circulate (the full set of about 169 features and 34 classes,
and reduced subsets); check those column names against the CSV variant you download.

## Setup

1. Download the CICIoT2023 CSVs (Neto et al., 2023) from the
   [official page](https://www.unb.ca/cic/datasets/iotdataset-2023.html) into `original/`.
2. Build the balanced binary dataset, from `examples/CICIoT2023/`:
   ```bash
   python ../CICIDS2017/scripts/dataset_creator.py --directory ./original --seeds 0,1,2,3,4 --keep-attack-type
   ```
   `--keep-attack-type` additionally writes `{split}_attack_types.csv` next to each split.
3. Every following stage (AutoML search, translation to PyTorch, QAT, feature ranking, baselines) is the one of
   CICIDS2017 with the paths pointing to `./balanced_binary/split_<seed>` of this folder (the AutoML search on `split_0`); see
   [`../CICIDS2017/README.md`](../CICIDS2017/README.md) and
   [`../CICIDS2017/docs/EXPERIMENTS_LOG.md`](../CICIDS2017/docs/EXPERIMENTS_LOG.md).

The dataset files are not included in this repository.
