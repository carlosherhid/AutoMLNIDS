# AutoML -> PyTorch -> QAT pipeline

Trains a binary (benign/malicious) intrusion-detection model, selects an architecture with AutoGluon, replicates it in
PyTorch and applies Quantization-Aware Training (QAT) with a near-zero feature rescaling. The folder is named after
CICIDS2017, the first dataset; the same scripts, with the same experiments, run on `CICIDS2017_improved` (the corrected
release), UNSW-NB15, TON_IoT and CICIoT2023 (see [`../CICIoT2023/README.md`](../CICIoT2023/README.md)).

## Pipeline

```
scripts/dataset_creator.py            raw CSVs -> one balanced binary train/validation/test/finetune split per seed
        |
scripts/automl_model_search.py        AutoGluon HPO -> best NeuralNetTorch architecture ("automl_search/")
        |
scripts/feature_ranking.py            sequential backward elimination; select_features.py keeps the features of Experiment 5
        |
scripts/autogluon_to_pytorch_qat.py   replicates the AutoGluon architecture in PyTorch and trains the FP32 and QAT
                                      models, with the near-zero feature rescaling and optional feature selection
                                      -> normal/qat *.pth checkpoints and execution_log_exp<N>.txt
```

The commands of the experiments of the paper are in [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md).
`autogluon_to_pytorch_qat.py` also contains options from an exploratory weight-pruning stage
(`--prune_model`, `--structured_pruning`, `--prune_amount`) that the paper does not use.

## Analysis scripts

| script | what it produces |
|---|---|
| `run_multiseed.py` | one numbered experiment (`--experiment N`) repeated over several seeds, mean +/- std |
| `run_analyses.py` | runs the analyses below for one dataset with the arguments that tie them to the multi-seed runs |
| `evaluate_checkpoint_metrics.py` | precision, recall, FPR and F1 of the saved checkpoints under other quantized engines |
| `build_metrics_table.py` | detection-metric tables (recall, FPR, F1 of the FP32 and INT8 models per experiment, projected precision, classical baselines) for any number of datasets, from the per-seed CSV of each experiment |
| `verify_functional_equivalence.py` | agreement between the AutoGluon model and the PyTorch replica |
| `build_tradeoff_table.py` | accuracy / F1 / size / latency / sparsity table |
| `baselines_classical_ml.py` | Random Forest, Logistic Regression and XGBoost baselines |
| `post_training_quantization.py` | static post-training quantization baseline |
| `build_preprocessing_table.py`, `input_resolution_diagnostic.py` | near-zero rescaling against other input preprocessings (table with seed-paired tests, Holm correction and confidence intervals; resolution that survives 8-bit input quantization) |
| `scale_factor_sweep.py` | accuracy as a function of the rescaling factor |
| `robustness_noise_eval.py` | accuracy under Gaussian input noise, FP32 vs. QAT |

`scripts/patches/apply_autogluon_patch.py` is **not** part of the pipeline: it is run once per environment (`cluster/setup_env.sbatch` does it) to guard
one expression of the installed AutoGluon that the hyperparameter search needs, see [`scripts/patches/README.md`](scripts/patches/README.md).

## Data setup

1. Download the raw CSVs of the dataset into `<data>/original/`, where `<data>` is a folder named after the dataset
   (`CICIDS2017_improved`, `UNSW_NB15`, `TON_IoT` or `CICIoT2023`: `dataset_creator.py` recognizes the dataset from that
   name). For the corrected CICIDS2017 release, download the daily CSVs of `CICIDS2017_improved.zip` (Engelen et al.).
2. Build the balanced binary splits, one per seed:
   ```bash
   python scripts/dataset_creator.py --directory <data>/original --seeds 0,1,2,3,4 --cpus 8
   ```
   This writes `train.csv`, `validation.csv`, `test.csv` and `finetune.csv` under `<data>/balanced_binary/split_<seed>/`
   for every seed, after removing exact duplicates and feature vectors that appear with both labels (fitting the Min-Max
   scaling on the training part of each split only). With `--keep-attack-type` it also writes `{split}_attack_types.csv`,
   the original (non-binarized) attack label of each row.
3. Run the AutoML search and then the QAT pipeline with the commands of
   [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md).
