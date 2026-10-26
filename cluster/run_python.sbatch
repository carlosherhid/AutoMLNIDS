#!/usr/bin/env bash
#SBATCH --job-name=automl-nids
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=automl-nids-%j.out
#SBATCH --error=automl-nids-%j.err

# Generic runner: activates the persistent venv (built once by setup_env.sbatch)
# and runs whatever python script + args you pass at submission time. One
# script covers every stage of the pipeline - dataset creation, AutoML search,
# QAT experiments, the analysis scripts, everything.
#
# Usage:
#   sbatch cluster/run_python.sbatch code/examples/CICIDS2017/scripts/dataset_creator.py --directory ./original ...
#   sbatch --job-name=exp6 cluster/run_python.sbatch code/examples/CICIDS2017/scripts/autogluon_to_pytorch_qat.py --predictor_path ...
#
# Override any #SBATCH directive per-submission on the sbatch command line
# itself, e.g. a lighter/CPU-only/shorter job:
#   sbatch --mem=16G --time=02:00:00 --cpus-per-task=6 cluster/run_python.sbatch code/.../dataset_creator.py ...

set -euo pipefail

echo "=== JOB INFO ==="
date -u
hostname
echo "JOBID=$SLURM_JOB_ID"
echo "USER=$USER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: $*"
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

if [[ $# -eq 0 ]]; then
  echo "ERROR: no command given. Usage: sbatch run_python.sbatch <script.py> [args...]" >&2
  exit 1
fi

cd "$PROJECT_DIR"
echo "=== RUNNING: python $* ==="
python "$@"

deactivate || true
echo "=== DONE ==="
