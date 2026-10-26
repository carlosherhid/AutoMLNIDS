# Running the pipeline on a SLURM GPU cluster

Two job scripts:

* [`setup_env.sbatch`](setup_env.sbatch) builds a **persistent** virtual environment once. The pipeline submits many
  small jobs (multi-seed reruns, sweeps, baselines), so reinstalling AutoGluon/PyTorch inside every job would be
  wasteful.
* [`run_python.sbatch`](run_python.sbatch) activates that environment and runs any script of the pipeline with the
  arguments given at submission time. The same wrapper covers every stage (dataset creation, AutoML search, QAT
  experiments, analysis scripts).

Both scripts must keep Unix (LF) line endings; a CRLF shows up as `bad interpreter` or `$'\r': command not found`
(`dos2unix cluster/*.sbatch` fixes it). Job stdout/stderr go to `automl-nids-<jobid>.out/.err` in the directory the job
was submitted from; the pipeline outputs (checkpoints, logs, CSVs) are written wherever `--output_dir`/`--data_dir`
point.

## 1. Project layout on the cluster

Everything lives under one project directory, `~/automl_nids` by default (any other path works if it is passed to
both scripts with `sbatch --export=PROJECT_DIR=...`):

```
~/automl_nids/
    requirements.txt      <- copy of the root requirements.txt
    code/                 <- this repository (without .git / .venv)
    data/
        CICIDS2017/balanced_binary/   split CSVs, automl_search/ predictor, compressed_models/
        CICIoT2023/balanced_binary/
```

Copy it from your machine, for example:

```bash
ssh <user>@<cluster-host> "mkdir -p ~/automl_nids/data"
scp requirements.txt <user>@<cluster-host>:~/automl_nids/requirements.txt
rsync -a --exclude .git --exclude .venv ./ <user>@<cluster-host>:~/automl_nids/code/
rsync -a examples/CICIDS2017/balanced_binary/ <user>@<cluster-host>:~/automl_nids/data/CICIDS2017/balanced_binary/
```

`requirements.txt` pins the software stack of the paper (Python 3.11, AutoGluon 1.6.2, PyTorch 2.4.1, scikit-learn 1.4.2)
to the versions of the cluster environment that produced the results.

## 2. Build the environment (once)

```bash
cd ~/automl_nids
sbatch code/cluster/setup_env.sbatch
squeue -u $USER
cat automl-nids-setup-<jobid>.out      # look for "CUDA available: True"
```

If it prints `CUDA available: False`, the partition or driver is not what the job expects and the training jobs would
silently fall back to the CPU.

## 3. Smoke test (useful after changing the AutoGluon version)

The scripts read undocumented AutoGluon internals (`predictor._trainer.model_best`, `predictor._trainer.load_model()`,
the `.model` attribute of the loaded model) that are not part of its public API and can change between releases; they
work with the version pinned in `requirements.txt`. A two-minute search checks them before a long run:

```bash
sbatch --job-name=smoke-test code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/automl_model_search.py search \
  --directory ~/automl_nids/data/CICIoT2023/balanced_binary \
  --cpus 4 --gpus 1 --attempts 1 --epochs 2 --patience 1 --time-limit 60 --metric accuracy

sbatch --job-name=smoke-dump code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/automl_model_search.py dump \
  --directory ~/automl_nids/data/CICIoT2023/balanced_binary/automl_search
```

Both jobs must finish without a Python traceback in their `.err` file. Run it on a dataset that does not yet have the
`automl_search/` you want to keep: the real search that follows renames the throwaway predictor with a timestamp
instead of deleting it.

## 4. Submit the pipeline

```bash
# AutoML search (CICIoT2023):
sbatch --job-name=ciciot-automl code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/automl_model_search.py search \
  --directory ~/automl_nids/data/CICIoT2023/balanced_binary \
  --cpus 12 --gpus 1 --attempts 1000 --epochs 1000 --patience 20 --time-limit 40000 --metric accuracy

# One QAT experiment:
sbatch --job-name=ciciot-exp1 code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/autogluon_to_pytorch_qat.py \
  --predictor_path ~/automl_nids/data/CICIoT2023/balanced_binary/automl_search \
  --data_dir ~/automl_nids/data/CICIoT2023/balanced_binary \
  --output_dir ~/automl_nids/data/CICIoT2023/balanced_binary/compressed_models \
  --epochs 100 --batch_size 256 --experiment_number 1 --normal --qat \
  --disable_early_stopping --disable_lr_scheduler

# The same configuration over 5 seeds (one job that loops internally):
sbatch --job-name=multiseed-exp1 code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/run_multiseed.py \
  --predictor_path ~/automl_nids/data/CICIDS2017/balanced_binary/automl_search \
  --data_dir ~/automl_nids/data/CICIDS2017/balanced_binary \
  --output_dir ~/automl_nids/data/CICIDS2017/balanced_binary/compressed_models \
  --experiment_number_base 1 --seeds 0 1 2 3 4 \
  --extra_args "--normal --qat --disable_early_stopping --disable_lr_scheduler"
```

The analysis scripts (`scale_factor_sweep.py`, `baselines_classical_ml.py`, `post_training_quantization.py`,
`robustness_noise_eval.py`, `build_tradeoff_table.py`, `verify_functional_equivalence.py`, ...) use the same
`sbatch code/cluster/run_python.sbatch code/examples/CICIDS2017/scripts/<script>.py <args>` pattern. `dataset_creator.py`
and `feature_ranking.py` do not need the GPU: add `--gres= --gpus=0` (and, for example, `--mem=16G`) to the `sbatch`
command, or use a CPU partition.

## 5. Monitor / cancel

```bash
squeue -u $USER
tail -f automl-nids-<jobid>.out
scancel <jobid>
```

The CICIDS2017 predictor (`automl_search/`) was created earlier with AutoGluon 1.1.1 on Python 3.10; it is loaded
unchanged under the pinned stack (see `examples/CICIDS2017/scripts/utils/common.py`).
