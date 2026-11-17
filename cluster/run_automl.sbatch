#!/usr/bin/env bash
#SBATCH --job-name=automl-search
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=automl-nids-%j.out
#SBATCH --error=automl-nids-%j.err

# AutoML search on ONE split. The search is AutoGluon's hyperparameter optimisation of neural networks (NeuralNetTorch, i.e.
# PyTorch models), which are trained on the GPU: automl_model_search.py gives num_gpus to predictor.fit(). The budget of the
# paper (the same for every dataset) is written below: 1000 attempts, at most 1000 epochs with patience 20, 40 000 s of search
# (11.1 h) and the accuracy as metric; the time limit of this job leaves about three hours of margin for loading the data and
# saving the predictor.
#
# Usage (from the project directory, ~/automl_nids):
#   sbatch --job-name=unsw-automl code/cluster/run_automl.sbatch ~/automl_nids/data/UNSW_NB15/balanced_binary/split_0
#
# Anything after the directory is passed to `automl_model_search.py search` and, for a repeated option, replaces the value
# written here (smoke test):
#   sbatch --time=00:30:00 --cpus-per-task=4 --mem=16G code/cluster/run_automl.sbatch <split dir> --attempts 1 --epochs 2 --patience 1 --time-limit 60
#
# Jobs that need no GPU (splits, ranking, baselines, analyses...) use run_python.sbatch, which reserves none.

set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: sbatch run_automl.sbatch <split directory> [extra options of automl_model_search.py search]" >&2
  exit 1
fi
SPLIT_DIR="$1"; shift

echo "=== JOB INFO ==="
date -u
hostname
echo "JOBID=$SLURM_JOB_ID"
echo "USER=$USER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Split: $SPLIT_DIR"
echo

PROJECT_DIR="${PROJECT_DIR:-$HOME/automl_nids}"
VENV_PATH="$PROJECT_DIR/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "ERROR: venv not found at $VENV_PATH - run 'sbatch cluster/setup_env.sbatch' first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"
python --version

echo "=== GPU CHECK ==="
nvidia-smi || true
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('ERROR: this job reserved a GPU but PyTorch cannot see it; stopping instead of searching on the CPU')
print('PyTorch sees the GPU:', torch.cuda.get_device_name(0))
PY

cd "$PROJECT_DIR"
CMD=(code/examples/CICIDS2017/scripts/automl_model_search.py search --directory "$SPLIT_DIR"
     --cpus "${SLURM_CPUS_PER_TASK:-12}" --gpus 1
     --attempts 1000 --epochs 1000 --patience 20 --time-limit 40000 --metric accuracy "$@")
echo "=== RUNNING: python ${CMD[*]} ==="
if [[ -n "${TELEMETRY_DIR:-}" ]]; then
  # Same command, plus a record of the machine and of the CPU, memory and GPU use of this job in $TELEMETRY_DIR
  # (see "Recording machine and resource use" in cluster/README.md).
  python code/examples/CICIDS2017/scripts/run_stage.py --name "${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}" \
    --out_dir "$TELEMETRY_DIR" -- python "${CMD[@]}"
else
  python "${CMD[@]}"
fi

deactivate || true
echo "=== DONE ==="
