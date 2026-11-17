# Experiment log

Commands behind the results of the paper, the same for every dataset (`CICIDS2017_improved`, `UNSW_NB15`, `TON_IoT`,
`CICIoT2023`): data splits, AutoML search, feature selection, the experiments of Table 1 and their repetitions, and the
analyses. On a SLURM cluster the whole sequence after the AutoML search is submitted by
[`cluster/submit_experiments.sh`](../../../cluster/submit_experiments.sh) (see [`../../../cluster/README.md`](../../../cluster/README.md));
this file lists what it runs.

The experiments are defined once, in [`scripts/utils/experiments.py`](../scripts/utils/experiments.py), and selected with
`run_multiseed.py --experiment N`:

| N | Experiment | Flags of `autogluon_to_pytorch_qat.py` |
|---|---|---|
| 1 | Exp. 1: baseline, no early stopping, no LR scheduling | `--normal --qat --disable_early_stopping --disable_lr_scheduler` |
| 2 | Exp. 2: early stopping enabled | `--normal --qat --disable_lr_scheduler` |
| 3 | Exp. 3: LR scheduling enabled | `--normal --qat --disable_early_stopping` |
| 4 | Exp. 4: near-zero rescaling (x1e6) | Exp. 3 + `--scale_close_to_zero` |
| 5 | Exp. 5: Exp. 4 + feature selection | Exp. 4 + `--prune_features --important_features_file <file>` |
| 6 | standardization instead of the rescaling | Exp. 3 + `--preprocess standard` |
| 7 | logarithmic Min-Max instead of the rescaling | Exp. 3 + `--preprocess logminmax` |
| 8 | quantile transform instead of the rescaling | Exp. 3 + `--preprocess quantile` |

Experiments 9-12 are not part of the default sequence (`--experiments "9 10 11 12"`): 9 and 10 repeat Exp. 3 and Exp. 4 with `--qconfig x86` (the
QAT configuration with `reduce_range`), and 11 and 12 repeat them with `--keep_batchnorm` (the BatchNorm layers of the AutoGluon network are kept
while training the FP32 network, folded into the Linear that follows each of them, and the QAT starts from the folded weights; the saved
checkpoints have the same layers as those of the other experiments).

**Quantized engine and QAT configuration.** The QAT configuration is `qnnpack` (unsigned 8-bit activations, symmetric per-tensor int8 weights: one
requantization multiplier per layer, as the firmware implements), and every INT8 number is obtained with an explicitly pinned engine,
`--engine onednn` by default in `autogluon_to_pytorch_qat.py`, `post_training_quantization.py`, `evaluate_checkpoint_metrics.py`,
`build_tradeoff_table.py` and `robustness_noise_eval.py` (PyTorch would otherwise use `x86`, whose kernels saturate on CPUs without VNNI).
The training log also prints the accuracy of the fake-quantized model and of the `x86` engine as diagnostics. The data copies that `--scale_close_to_zero`
or `--prune_features` produce are no longer written next to the dataset (parallel runs overwrote each other's files); `--save_processed_csv` writes them
into the run's `--output_dir`.

Seed *S* of experiment *N* is run number `100000 + N*100 + S`: `normal_pytorch_model_exp<run>.pth`,
`qat_pytorch_model_exp<run>.pth` and `execution_log_exp<run>.txt` in `compressed_models/`, and `multiseed_exp<N>_results.csv`
with the five seeds. Seed *S* is trained and tested on its own data split, `split_<S>`.

All commands are run from `examples/CICIDS2017/`. `<data>` is the folder of one dataset, with the raw CSVs in
`<data>/original/` (the dataset is recognized from the name of that folder).

## 1. Data splits

```bash
python scripts/dataset_creator.py --directory <data>/original --seeds 0,1,2,3,4 --cpus 8 --keep-attack-type
```

Writes `<data>/balanced_binary/split_<seed>/{train,validation,test,finetune}.csv` for each seed (and
`split_info.json` with the sizes, the dropped columns and the deduplication counts). Exact duplicates, and feature vectors
that appear with both labels, are removed before splitting; each seed draws its own balanced sample and its own split, and the
Min-Max scaling is fitted on the training part only.

Two options change how the data is cleaned and are used for sensitivity analyses:

* `--conflicts drop|majority`: feature vectors that appear with both labels are removed entirely (`drop`, default) or kept once with their majority label (`majority`; exact ties are removed).
* `--onehot proto,conn_state,service`: text columns coded as 0/1 indicator columns (the 20 most frequent values of each, the rest as `other`) instead of being dropped.
* `--exclude-attack "DoS Hulk"`: removes every row of the named attack families (comma-separated, matched case-insensitively on the attack-type column; a trailing `*` matches a prefix) before deduplicating and balancing.
* `--drop-columns "sttl,ct_state_ttl"`: removes the named columns on top of the ones the dataset always excludes (raises an error if one does not exist), e.g. for the UNSW-NB15 variant without the TTL variables.
* `--official-split`: for datasets released with an official training file and an official testing file (file names containing `train` and `test`, e.g. UNSW-NB15): duplicates are removed inside each file, the official test set
  becomes the test split (not balanced; `split_info.json` records the share of its rows with an identical twin in train) and the validation and fine-tuning splits are taken from the training set.

For CICIoT2023 pass `--max-rows-per-file 20000`, the per-file cap of the attack files with which its published balanced file was built.

## 2. AutoML baseline (AutoGluon)

```bash
python scripts/automl_model_search.py search --directory <data>/balanced_binary/split_0 --cpus 12 --gpus 1 --attempts 1000 --epochs 1000 --patience 20 --time-limit 40000 --metric accuracy
```

The predictor is trained on `split_0` (`split_0/automl_search`); the other seeds reuse its architecture, and the accuracy of
the AutoGluon model itself is evaluated only on `split_0`.

## 3. Feature selection (input of Experiment 5)

```bash
python scripts/feature_ranking.py --dataset-directory <data>/balanced_binary/split_0 --predictor-directory <data>/balanced_binary/split_0/automl_search --algorithm custom_sbe --cpus 6 --time-limit 40000 --metric accuracy --recursive
python scripts/select_features.py --ranking_dir <data>/balanced_binary/split_0/automl_search/feature_ranking_custom_sbe --tolerance 1.0
```

The ranking removes one feature at a time (sequential backward elimination) and records the accuracy on the fine-tuning
split. `select_features.py` keeps the smallest subset whose accuracy is within `--tolerance` percentage points of the accuracy
with all the features, the same rule for every dataset, and writes `important_features.txt` and `feature_selection.json`.

## 4. Experiments 1-8 over 5 seeds

```bash
export OMP_NUM_THREADS=6      # QAT results depend on the number of PyTorch threads; every log records it
for N in 1 2 3 4 6 7 8; do
  python scripts/run_multiseed.py --experiment $N --predictor_path <data>/balanced_binary/split_0/automl_search --data_dir_template "<data>/balanced_binary/split_{seed}" --output_dir <data>/balanced_binary/compressed_models --epochs 100 --batch_size 256 --seeds 0 1 2 3 4
done
python scripts/run_multiseed.py --experiment 5 --important_features_file <data>/balanced_binary/split_0/automl_search/feature_ranking_custom_sbe/important_features.txt --predictor_path <data>/balanced_binary/split_0/automl_search --data_dir_template "<data>/balanced_binary/split_{seed}" --output_dir <data>/balanced_binary/compressed_models --epochs 100 --batch_size 256 --seeds 0 1 2 3 4
```

A single run (one seed, one split) is `scripts/autogluon_to_pytorch_qat.py` with the flags of the table, `--data_dir
<data>/balanced_binary/split_<S>`, `--seed <S>` and `--experiment_number <run>`.

## 5. Baselines and analyses

```bash
for S in 0 1 2 3 4; do
  python scripts/baselines_classical_ml.py --data_dir <data>/balanced_binary/split_$S --output_dir <data>/balanced_binary/baselines/split_$S
done
python scripts/run_analyses.py --data_root <data>/balanced_binary --seeds 0 1 2 3 4 --stages ptq equivalence noise resolution tradeoff engines
python scripts/run_analyses.py --data_root <data>/balanced_binary --stages sweep
```

`run_analyses.py` runs, with the arguments that tie each analysis to the runs above, and writes under
`<data>/balanced_binary/analyses/`:

| stage | analysis | script |
|---|---|---|
| `ptq` | static post-training quantization of the FP32 model of each seed, for the experiments of `--ptq_experiments` (default Experiment 1; `--ptq_experiments 3 4 6 7 8` quantizes each one with the input transformation it was trained with) | `post_training_quantization.py` |
| `equivalence` | prediction agreement between the AutoGluon model and the FP32 replica of Experiment 3 (the same Min-Max input), seed 0 | `verify_functional_equivalence.py` |
| `noise` | Experiment 4, seed 0, under Gaussian input noise | `robustness_noise_eval.py` |
| `resolution` | resolution that survives 8-bit input quantization, per input representation | `input_resolution_diagnostic.py` |
| `tradeoff` | size and single-thread latency of the seed-0 model of Experiments 1-5 | `build_tradeoff_table.py` |
| `engines` | the checkpoints of Experiments 1-5 re-evaluated with the `x86`, `onednn` and `qnnpack` engines | `evaluate_checkpoint_metrics.py` |
| `sweep` | Experiment 4 for rescaling factors from 1e2 to 1e8 (seed 0, `split_0`) | `scale_factor_sweep.py` |

## 6. Tables

The tables of the paper are built from the files above. Both table scripts take the datasets as `NAME=ROOT[,MODELS_DIR]`
(`ROOT` = `<data>/<dataset>/balanced_binary`; `MODELS_DIR` = `compressed_models`, or `compressed_models_bnfold` for the BatchNorm-folded runs):

```bash
python scripts/build_metrics_table.py --dataset CICIDS2017_improved=<data>/CICIDS2017_improved/balanced_binary \
    UNSW_NB15=<data>/UNSW_NB15/balanced_binary,compressed_models_bnfold TON_IoT=<data>/TON_IoT/balanced_binary --output_dir <tables>
python scripts/build_preprocessing_table.py --dataset <the same datasets> --output_dir <tables>
```

* `build_metrics_table.py`: recall, FPR and F1 of the FP32 and INT8 models of every experiment (mean +/- std over the seeds), the precision they imply
  at other attack prevalences, and the classical baselines, from `multiseed_exp<N>_results.csv` and `baselines/split_<seed>/`. The FPR of a baseline
  whose CSV has no confusion counts is derived with the real share of malicious flows of its test set (`split_info.json`).
* `build_preprocessing_table.py` (Experiments 3, 4, 6, 7 and 8): the same metrics per input representation, the number of seeds whose INT8 model
  loses more than 2 points, and the paired comparisons of Experiment 4 with each alternative (seed by seed, since seed *i* is trained and tested on
  `split_<i>`): mean difference with its 95 % interval, median, seeds ahead, paired t-test, exact Wilcoxon test and Holm-adjusted p.
* The trade-off table comes from `analyses/tradeoff_table_*.csv` (its AutoGluon accuracy is that of a run with the original input), the AutoGluon/replica
  agreement from `analyses/functional_equivalence_exp<N>.csv`, and the on-device evaluation from the seed-0 checkpoints of Experiment 4
  (`examples/IoT_test/m5stack_ids_eval/`).
