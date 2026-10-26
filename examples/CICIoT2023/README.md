# CICIoT2023: second dataset

Second validation dataset. It reuses the scripts of [`../CICIDS2017/scripts/`](../CICIDS2017/scripts): every script
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
   python ../CICIDS2017/scripts/dataset_creator.py --directory ./original --test-size 0.3 --finetune-size 0.5 --validation-size 0.2 --composition balanced --problem-type binary --keep-attack-type
   ```
   `--keep-attack-type` additionally writes `{split}_attack_types.csv` next to each split.
3. Every following stage (AutoML search, translation to PyTorch, QAT, feature ranking, baselines) is the one of
   CICIDS2017 with `--data_dir` / `--directory` pointing to `./balanced_binary` of this folder; see
   [`../CICIDS2017/README.md`](../CICIDS2017/README.md) and
   [`../CICIDS2017/docs/EXPERIMENTS_LOG.md`](../CICIDS2017/docs/EXPERIMENTS_LOG.md).

The dataset files are not included in this repository.
