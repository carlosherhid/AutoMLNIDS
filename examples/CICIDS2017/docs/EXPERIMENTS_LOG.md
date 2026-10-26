# Experiment log

Commands behind the results of the paper: dataset creation, AutoML search, feature ranking, the five experiments of
Table 1 and their multi-seed repetitions. A saved checkpoint (`compressed_models/*_exp<N>.pth`) and its log
(`compressed_models/execution_log_exp<N>.txt`) can be traced to the exact flags used through the internal
`--experiment_number`:

| Paper (Table 1) | Internal `--experiment_number` | Checkpoint / log |
|---|---|---|
| Exp. 1 — baseline, no early stopping, no LR scheduling | **1** | `*_exp1.pth`, `execution_log_exp1.txt` |
| Exp. 2 — early stopping enabled | **2** | `*_exp2.pth`, `execution_log_exp2.txt` |
| Exp. 3 — LR scheduling enabled | **3** | `*_exp3.pth`, `execution_log_exp3.txt` |
| Exp. 4 — near-zero feature scaling (×1e6) | **6** | `*_exp6.pth`, `execution_log_exp6.txt` |
| Exp. 5 — Exp. 4 + feature selection | **7** | `*_exp7.pth`, `execution_log_exp7.txt` |

All commands are run from `examples/CICIDS2017/`; on a SLURM cluster, wrap them with `cluster/run_python.sbatch` (see
[`../../../cluster/README.md`](../../../cluster/README.md)).

## 1. Dataset creation

```bash
python scripts/dataset_creator.py --directory ./datasets/CICIDS2017 --test-size 0.3 --finetune-size 0.5 --validation-size 0.2 --cpus 6 --composition balanced --problem-type binary
```

## 2. AutoML baseline (AutoGluon)

```bash
python scripts/automl_model_search.py search --directory ./datasets/CICIDS2017/balanced_binary --cpus 4 --gpus 0 --attempts 1000 --epochs 1000 --patience 20 --time-limit 40000 --metric accuracy
```

Detailed results: `datasets/CICIDS2017/balanced_binary/automl_search/models/NeuralNetTorch`

`--epochs`, `--patience` and `--attempts` are the script defaults. The search that produced the saved predictor trained
28 trials of up to 1000 epochs with patience 20. `--cpus` and `--time-limit` are the
values of the original command notes, which were written before the run. This search was run earlier on a different
machine (AutoGluon 1.1.1, Python 3.10); the experiments below load that predictor unchanged under the stack pinned in
[`requirements.txt`](../../../requirements.txt).

## 3. Feature ranking (used for Experiment 5's feature selection)

```bash
python scripts/feature_ranking.py --dataset-directory ./datasets/CICIDS2017/balanced_binary --predictor-directory ./datasets/CICIDS2017/balanced_binary/automl_search --algorithm custom_sbe --cpus 6 --time-limit 40000 --metric accuracy --recursive
```

## 4. AutoGluon → PyTorch → QAT experiments

Each command is one run of `scripts/autogluon_to_pytorch_qat.py`, producing `normal_pytorch_model_exp<N>.pth` /
`qat_pytorch_model_exp<N>.pth` and `execution_log_exp<N>.txt` in `compressed_models/`.

```bash
# Experiment 1
python scripts/autogluon_to_pytorch_qat.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --epochs 100 --batch_size 256 --experiment_number 1 --normal --qat --disable_early_stopping --disable_lr_scheduler

# Experiment 2
python scripts/autogluon_to_pytorch_qat.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --epochs 100 --batch_size 256 --experiment_number 2 --normal --qat --disable_lr_scheduler

# Experiment 3
python scripts/autogluon_to_pytorch_qat.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --epochs 100 --batch_size 256 --experiment_number 3 --normal --qat --disable_early_stopping

# Experiment 4 (internal 6)
python scripts/autogluon_to_pytorch_qat.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --epochs 100 --batch_size 256 --experiment_number 6 --normal --qat --disable_early_stopping --scale_close_to_zero

# Experiment 5 (internal 7)
python scripts/autogluon_to_pytorch_qat.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --epochs 100 --batch_size 256 --experiment_number 7 --normal --qat --disable_early_stopping --scale_close_to_zero --prune_features --important_features_file ./datasets/CICIDS2017/balanced_binary/automl_search/feature_ranking_custom_sbe/important_features.txt
```

## 5. Multi-seed runs

The paper reports means over 5 seeds (0–4). `scripts/run_multiseed.py` repeats one configuration with `--seed` and writes
`multiseed_exp<base>_results.csv`; `--extra_args` takes the flags of the corresponding command above (without the paths,
`--epochs`, `--batch_size` and `--experiment_number`). For Experiment 4:

```bash
python scripts/run_multiseed.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --experiment_number_base 6 --seeds 0 1 2 3 4 --extra_args "--normal --qat --disable_early_stopping --scale_close_to_zero"
```

Experiments 1, 2, 3 and 5 use `--experiment_number_base` 1, 2, 3 and 7 with their own flags. The input preprocessings
compared with the near-zero rescaling (`--preprocess`, in place of `--scale_close_to_zero`) are bases 50
(`standard`), 51 (`logminmax`) and 52 (`quantile`):

```bash
python scripts/run_multiseed.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --experiment_number_base 50 --seeds 0 1 2 3 4 --extra_args "--normal --qat --disable_early_stopping --preprocess standard"
python scripts/run_multiseed.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --experiment_number_base 51 --seeds 0 1 2 3 4 --extra_args "--normal --qat --disable_early_stopping --preprocess logminmax"
python scripts/run_multiseed.py --predictor_path ./datasets/CICIDS2017/balanced_binary/automl_search --data_dir ./datasets/CICIDS2017/balanced_binary --output_dir ./datasets/CICIDS2017/balanced_binary/compressed_models --experiment_number_base 52 --seeds 0 1 2 3 4 --extra_args "--normal --qat --disable_early_stopping --preprocess quantile"
```

## 6. Other analyses

The remaining tables and figures of the paper (classical baselines, post-training quantization, scale-factor sweep, input
noise, trade-off table, functional equivalence, ...) come from the scripts listed in [`../README.md`](../README.md), each
with its command in its docstring. The same pipeline was run on CICIoT2023: see
[`../../CICIoT2023/README.md`](../../CICIoT2023/README.md) and [`../../../cluster/README.md`](../../../cluster/README.md).
