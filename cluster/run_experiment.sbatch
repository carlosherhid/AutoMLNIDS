#!/usr/bin/env bash
#SBATCH --job-name=qat-experiment
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=automl-nids-%j.out
#SBATCH --error=automl-nids-%j.err

# The two stages that train models on the GPU after the AutoML search:
#   experiment   run_multiseed.py: one numbered experiment over the seeds. Each seed trains the FP32 replica and the QAT model
#                with autogluon_to_pytorch_qat.py, which moves the model and the batches to the GPU (device = cuda); only the
#                INT8 conversion and the quantized evaluation run on the CPU.
#   sweep        run_analyses.py --stages sweep: Experiment 4 for a range of rescaling factors, the same training on the GPU.
# The CPU cores are the threads of PyTorch (OMP_NUM_THREADS, which submit_experiments.sh fixes): QAT results depend on that number.
#
# Usage (from the project directory, ~/automl_nids); submit_experiments.sh builds these commands:
#   sbatch --job-name=unsw-exp4 --export=ALL,OMP_NUM_THREADS=6 code/cluster/run_experiment.sbatch experiment \
#     --experiment 4 --predictor_path <split_0/automl_search> --data_dir_template "<balanced_binary>/split_{seed}" \
#     --output_dir <compressed_models> --epochs 100 --batch_size 256 --seeds 0 1 2 3 4
#   sbatch --job-name=unsw-sweep --export=ALL,OMP_NUM_THREADS=6 code/cluster/run_experiment.sbatch sweep \
#     --data_root <balanced_binary> --sweep_epochs 100
#
# Jobs that need no GPU (splits, ranking, baselines, analyses...) use run_python.sbatch, which reserves none.

set -euo pipefail

STAGE="${1:-}"
case "$STAGE" in
  experiment|sweep) shift;;
  *) echo "usage: sbatch run_experiment.sbatch experiment <run_multiseed.py options> | sweep <run_analyses.py options>" >&2; exit 1;;
esac
SCRIPTS=code/examples/CICIDS2017/scripts
if [[ "$STAGE" == experiment ]]; then
  CMD=("$SCRIPTS/run_multiseed.py" "$@")
else
  CMD=("$SCRIPTS/run_analyses.py" --stages sweep "$@")
fi

echo "=== JOB INFO ==="
date -u
hostname
echo "JOBID=$SLURM_JOB_ID"
echo "USER=$USER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Stage: $STAGE"
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
# autogluon_to_pytorch_qat.py falls back to the CPU when it does not see a GPU, and the results of the CPU and of the GPU are
# not comparable: stop here instead of silently training on the CPU while holding a GPU.
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('ERROR: this job reserved a GPU but PyTorch cannot see it; stopping instead of training on the CPU')
print('PyTorch sees the GPU:', torch.cuda.get_device_name(0))
PY

cd "$PROJECT_DIR"
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
