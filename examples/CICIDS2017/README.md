# CICIDS2017: AutoML -> PyTorch -> QAT pipeline

Trains a binary (benign/malicious) intrusion-detection model on
[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), selects an architecture with AutoGluon, replicates it in
PyTorch and applies Quantization-Aware Training (QAT) with a near-zero feature rescaling. The same scripts are used for
CICIoT2023 (see [`../CICIoT2023/README.md`](../CICIoT2023/README.md)).

## Pipeline

```
scripts/dataset_creator.py            raw CICIDS2017 CSVs -> balanced binary train/validation/test/finetune splits
        |
scripts/automl_model_search.py        AutoGluon HPO -> best NeuralNetTorch architecture ("automl_search/")
        |
scripts/feature_ranking.py            sequential backward elimination (feature selection of Experiment 5)
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
| `run_multiseed.py` | one configuration repeated over several seeds, mean +/- std |
| `evaluate_checkpoint_metrics.py`, `build_metrics_table.py` | precision, recall, FPR and F1 of the saved checkpoints, and the detection-metric tables |
| `verify_functional_equivalence.py` | agreement between the AutoGluon model and the PyTorch replica |
| `build_tradeoff_table.py` | accuracy / F1 / size / latency / sparsity table |
| `baselines_classical_ml.py` | Random Forest, Logistic Regression and XGBoost baselines |
| `post_training_quantization.py` | static post-training quantization baseline |
| `build_preprocessing_table.py`, `input_resolution_diagnostic.py` | near-zero rescaling against other input preprocessings |
| `scale_factor_sweep.py` | accuracy as a function of the rescaling factor |
| `robustness_noise_eval.py` | accuracy under Gaussian input noise, FP32 vs. QAT |

`scripts/patches/autogluon_auto_trainer.py` is **not** run directly: it replaces a file inside the installed
AutoGluon package that the hyperparameter search needs, see [`scripts/patches/README.md`](scripts/patches/README.md).

## Data setup

1. Download the daily CICIDS2017 CSVs from the
   [official page](https://www.unb.ca/cic/datasets/ids-2017.html) into `original/`.
2. Build the balanced binary dataset:
   ```bash
   python scripts/dataset_creator.py --directory ./original --test-size 0.3 --finetune-size 0.5 --validation-size 0.2 --composition balanced --problem-type binary
   ```
   This writes `train.csv`, `validation.csv`, `test.csv` and `finetune.csv` under `balanced_binary/`. With
   `--keep-attack-type` it also writes `{split}_attack_types.csv`, the original (non-binarized) attack label of
   each row.
3. Run the AutoML search and then the QAT pipeline with the commands of
   [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md).
