# Running the pipeline on a SLURM GPU cluster

Four job scripts:

* [`setup_env.sbatch`](setup_env.sbatch) builds a **persistent** virtual environment once. The pipeline submits many
  small jobs (multi-seed reruns, sweeps, baselines), so reinstalling AutoGluon/PyTorch inside every job would be
  wasteful.
* [`run_python.sbatch`](run_python.sbatch) activates that environment and runs any script of the pipeline **that does not
  use the GPU** with the arguments given at submission time (dataset creation, feature ranking and selection, classical
  baselines, `describe_automl_model.py`, the analysis scripts). It sits in a CPU partition and reserves no GPU.
* [`run_automl.sbatch`](run_automl.sbatch) runs the AutoML search of one split (AutoGluon trains PyTorch networks on the
  GPU, with the search budget of the paper written in the script).
* [`run_experiment.sbatch`](run_experiment.sbatch) runs the stages that train on the GPU after the search: the numbered QAT
  experiments over the seeds and the rescaling-factor sweep.

The two GPU scripts name the Python script they run, and stop with an error if PyTorch does not see the GPU the job
reserved (the training scripts would otherwise fall back to the CPU, whose results are not comparable). A cluster
scheduler that reviews the submitted scripts can then see that the GPU is used, which it cannot with a generic wrapper
that only runs `python "$@"`.

All the scripts must keep Unix (LF) line endings; a CRLF shows up as `bad interpreter` or `$'\r': command not found`
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
        <DATASET>/original/                              raw CSVs (CICIDS2017_improved, UNSW_NB15, CICIoT2023, TON_IoT)
        <DATASET>/balanced_binary/split_<seed>/          train/validation/test/finetune CSVs of each split
        <DATASET>/balanced_binary/split_0/automl_search/ AutoGluon predictor (trained on split_0)
        <DATASET>/balanced_binary/compressed_models/     checkpoints and logs of the experiments
```

`dataset_creator.py` recognizes the dataset from the name of the data directory (`CICIDS2017_improved`, `UNSW_NB15`,
`TON_IoT`, `CICIoT2023`, ...), so keep those folder names.

Copy it from your machine, for example:

```bash
ssh <user>@<cluster-host> "mkdir -p ~/automl_nids/data"
scp requirements.txt <user>@<cluster-host>:~/automl_nids/requirements.txt
rsync -a --exclude .git --exclude .venv ./ <user>@<cluster-host>:~/automl_nids/code/
rsync -a <raw CSVs of one dataset>/ <user>@<cluster-host>:~/automl_nids/data/<DATASET>/original/
```

`requirements.txt` pins the software stack of the paper (Python 3.11, AutoGluon 1.6.2, PyTorch 2.4.1, scikit-learn 1.4.2)
to the versions of the cluster environment that produced the results.

## 2. Build the environment (once)

```bash
cd ~/automl_nids
sbatch code/cluster/setup_env.sbatch
squeue -u $USER
cat automl-nids-setup-<jobid>.out      # look for "PyTorch version", "Patched" and "AutoGluon import OK"
```

The setup job only installs packages: it runs on a CPU partition and reserves no GPU, so it prints `CUDA available: False`.
The GPU is checked by the jobs that use it (`=== GPU CHECK ===` in their output), for instance in the smoke test below.

### Python for the CPU nodes

A venv points at the interpreter it was built with. If that interpreter is a system Python installed only on some nodes (here, the GPU nodes),
jobs that need no GPU and run on a CPU partition cannot activate the environment. The jobs that need no GPU (splits, baselines, ranking, analyses) must
not reserve one, so the environment has to work on every node. One way: install a standalone Python 3.11 on the shared file system and build the venv with it.

```bash
# the installer's UV_UNMANAGED_INSTALL puts the binary in that folder and leaves the shell configuration alone
curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL=$PROJECT_DIR/tools sh
UV_PYTHON_INSTALL_DIR=$PROJECT_DIR/tools/pythons $PROJECT_DIR/tools/uv python install 3.11
export PYTHON311=$(ls -d $PROJECT_DIR/tools/pythons/cpython-3.11*/bin/python3.11 | head -1)
mv $PROJECT_DIR/.venv $PROJECT_DIR/.venv_old        # if there is one; kept in case you want to go back
sbatch --export=ALL cluster/setup_env.sbatch        # CPU partition: builds .venv with $PYTHON311 and installs requirements.txt
```

`setup_env.sbatch` refuses to build the environment with any other Python version, and does not touch an existing venv whose interpreter is
missing on the node it runs on.

## 3. Smoke test (useful after changing the AutoGluon version)

The scripts read undocumented AutoGluon internals (`predictor._trainer.model_best`, `predictor._trainer.load_model()`,
the `.model` attribute of the loaded model) that are not part of its public API and can change between releases; they
work with the version pinned in `requirements.txt`. A two-minute search checks them before a long run:

```bash
sbatch --job-name=smoke-test --time=00:30:00 --cpus-per-task=4 --mem=16G code/cluster/run_automl.sbatch \
  ~/automl_nids/smoke/split_0 --attempts 1 --epochs 2 --patience 1 --time-limit 60

sbatch --job-name=smoke-dump code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/automl_model_search.py dump \
  --directory ~/automl_nids/smoke/split_0/automl_search
```

Both jobs must finish without a Python traceback in their `.err` file. Run it on a throwaway copy of one split
(`cp -r .../balanced_binary/split_0 ~/automl_nids/smoke/split_0`), so that no toy `automl_search/` is left in the real data.

## 4. Submit the pipeline

```bash
# Splits: one balanced train/validation/test/finetune split per seed, written to ../balanced_binary/split_<seed>/
sbatch --job-name=unsw-split --cpus-per-task=2 --mem=8G code/cluster/run_python.sbatch \
  code/examples/CICIDS2017/scripts/dataset_creator.py \
  --directory ~/automl_nids/data/UNSW_NB15/original --seeds 0,1,2,3,4 --cpus 2 --keep-attack-type

# AutoML search on split_0 (GPU; the budget of the paper is in the script, see run_automl.sbatch):
sbatch --job-name=unsw-automl code/cluster/run_automl.sbatch ~/automl_nids/data/UNSW_NB15/balanced_binary/split_0

# One numbered experiment over 5 seeds, seed i on split_i (one job that loops internally, GPU). QAT results depend on the number
# of PyTorch threads, so fix it (the log records it) and give the job one CPU per thread:
sbatch --job-name=unsw-exp1 --cpus-per-task=6 --export=ALL,OMP_NUM_THREADS=6 code/cluster/run_experiment.sbatch experiment \
  --predictor_path ~/automl_nids/data/UNSW_NB15/balanced_binary/split_0/automl_search \
  --data_dir_template "$HOME/automl_nids/data/UNSW_NB15/balanced_binary/split_{seed}" \
  --output_dir ~/automl_nids/data/UNSW_NB15/balanced_binary/compressed_models \
  --experiment 1 --seeds 0 1 2 3 4
```

`--experiment N` selects one of the numbered experiments of `examples/CICIDS2017/scripts/utils/experiments.py` (1-5: the
experiments of the paper; 6-8: the alternative input representations).

The AutoGluon predictor is trained on `split_0`, so its own accuracy is evaluated only there (`run_multiseed.py` adds
`--skip_autogluon_eval` for the other seeds).

[`submit_experiments.sh`](submit_experiments.sh) submits, for one dataset and chained with dependencies, everything that
follows the AutoML search: the feature ranking and selection of Experiment 5, Experiments 1-8 over the seeds (one job per
experiment, `OMP_NUM_THREADS` fixed), the classical baselines of every split, the analyses that need the trained models
(`run_analyses.py`) and the rescaling-factor sweep:

```bash
bash code/cluster/submit_experiments.sh UNSW_NB15 --after <automl job id>
bash code/cluster/submit_experiments.sh UNSW_NB15 --dry-run      # only print the sbatch commands
```

When the AutoML model has BatchNorm layers and the replica without them does not train (a deep network, for instance the 21-layer model
chosen for UNSW-NB15), run the same experiments with the BatchNorm kept during the FP32 training and **folded** into the next `Linear` before the
quantization (in inference a BatchNorm is a per-feature affine map, so it is absorbed into the weights and bias of the `Linear` that follows it and
the network keeps only the layer types of any other replica); the QAT starts from the folded weights:

```bash
bash code/cluster/submit_experiments.sh UNSW_NB15 --keep-batchnorm       # Experiments 1-8 with the same numbers, in compressed_models_bnfold/ and analyses_bnfold/
```

`run_multiseed.py --keep_batchnorm` and `run_analyses.py --models_dir compressed_models_bnfold --out_dir analyses_bnfold --keep_batchnorm` are what
that option calls. Check `Folded N BatchNorm layers ... accuracy X% before, X% after` in every log: both numbers must match.

Any other script of the pipeline (those that do not train on the GPU) uses
`sbatch code/cluster/run_python.sbatch code/examples/CICIDS2017/scripts/<script>.py <args>`; add `--mem=...`,
`--cpus-per-task=...` or `--time=...` to the `sbatch` command when the defaults of the script (6 CPUs, 16 GB, 2 hours) are
not enough.

## 5. Monitor / cancel

```bash
squeue -u $USER
tail -f automl-nids-<jobid>.out
scancel <jobid>
```

## 6. Recording machine and resource use

To keep a record of where and how every stage ran, submit the jobs with `TELEMETRY_DIR` set:

```bash
sbatch --export=ALL,TELEMETRY_DIR=$HOME/automl_nids/telemetry --job-name=exp6 code/cluster/run_python.sbatch <script.py> <args>
```

The three runners (`run_python.sbatch`, `run_automl.sbatch`, `run_experiment.sbatch`) then run the script through [`run_stage.py`](../examples/CICIDS2017/scripts/run_stage.py), which writes in
`$TELEMETRY_DIR`:

* `machine.json` (once): CPU model and cores, RAM, GPUs and driver, operating system, Python and package versions, PyTorch
  threads and quantization engine, and the SLURM allocation (partition, node list, CPUs, memory, GPUs);
* `<job name>-<job id>.json` per stage: start and end time, wall-clock time, exit code, CPU time, peak resident memory, mean CPU
  use, and per GPU the mean utilization, peak memory, mean power and an estimate of the energy;
* `stages.csv`: one row per stage with the main figures, ready to be turned into a table.

Without `TELEMETRY_DIR` the wrapper is not used and nothing changes. The power and energy figures are estimates from
`nvidia-smi` samples (once per second), and the GPU memory of the table is that of the whole device; the memory of the process
itself is in the JSON. For the exact accounting of SLURM, run `sacct -j <jobid> --format=JobID,Elapsed,TotalCPU,MaxRSS,AllocTRES%60`
after the job ends. The inference latency of the trained models is measured on the same machine by
`build_tradeoff_table.py`.
