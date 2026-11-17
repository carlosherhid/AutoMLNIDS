#!/usr/bin/env bash
#SBATCH --job-name=automl-nids-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=automl-nids-%j.out
#SBATCH --error=automl-nids-%j.err

# Generic runner for the stages that do NOT use the GPU: activates the persistent venv (built once by setup_env.sbatch)
# and runs whatever python script + args you pass at submission time: dataset creation (splits), feature ranking and
# selection, classical baselines, describe_automl_model.py, the analysis scripts (PTQ, equivalence, noise, trade-off...).
# It reserves no GPU and sits in a CPU partition.
#
# The stages that train models on the GPU have their own scripts, which say what they run and check that the GPU is visible:
#   run_automl.sbatch       AutoML search (AutoGluon, NeuralNetTorch on the GPU)
#   run_experiment.sbatch   QAT experiments and the rescaling-factor sweep
#
# Usage:
#   sbatch cluster/run_python.sbatch code/examples/CICIDS2017/scripts/dataset_creator.py --directory ./original ...
#
# Override any #SBATCH directive per-submission on the sbatch command line itself, e.g. more CPUs, memory or time:
#   sbatch --mem=32G --time=1-00:00:00 --cpus-per-task=6 cluster/run_python.sbatch code/.../feature_ranking.py ...

set -euo pipefail

echo "=== JOB INFO ==="
date -u
hostname
echo "JOBID=$SLURM_JOB_ID"
echo "USER=$USER"
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

if [[ $# -eq 0 ]]; then
  echo "ERROR: no command given. Usage: sbatch run_python.sbatch <script.py> [args...]" >&2
  exit 1
fi

cd "$PROJECT_DIR"
echo "=== RUNNING: python $* ==="
if [[ -n "${TELEMETRY_DIR:-}" ]]; then
  # Same command, plus a record of the machine and of the CPU and memory use of this job in $TELEMETRY_DIR
  # (see "Recording machine and resource use" in cluster/README.md).
  python code/examples/CICIDS2017/scripts/run_stage.py --name "${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}" \
    --out_dir "$TELEMETRY_DIR" -- python "$@"
else
  python "$@"
fi

deactivate || true
echo "=== DONE ==="
