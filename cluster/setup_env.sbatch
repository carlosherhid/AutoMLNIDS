#!/usr/bin/env bash
#SBATCH --job-name=automl-nids-setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=automl-nids-setup-%j.out
#SBATCH --error=automl-nids-setup-%j.err

# One-time job: builds a PERSISTENT venv (not per-job scratch, unlike the
# classification-leaf example this is adapted from) so every later job just
# activates it instead of reinstalling autogluon/torch/etc. every time - that
# install is heavy (many GB of transitive deps) and we submit many small jobs
# (multi-seed reruns, sweeps...), not one long one.
#
# Run once: sbatch cluster/setup_env.sbatch
# Re-run any time you change requirements.txt (it just reinstalls into the
# same venv).

set -euo pipefail

echo "=== JOB INFO ==="
date -u
hostname
echo "JOBID=$SLURM_JOB_ID"
echo "USER=$USER"
echo

# Fixed, persistent project location - same path every job expects. Override
# with `sbatch --export=PROJECT_DIR=/some/other/path cluster/setup_env.sbatch`
# if the project is not under $HOME/automl_nids.
PROJECT_DIR="${PROJECT_DIR:-$HOME/automl_nids}"
VENV_PATH="$PROJECT_DIR/.venv"
echo "PROJECT_DIR=$PROJECT_DIR"

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
  echo "ERROR: $PROJECT_DIR/requirements.txt not found - rsync the project there first (see cluster/README.md)." >&2
  exit 1
fi

echo "=== FINDING SYSTEM PYTHON (3.11) ==="
# requirements.txt pins the software stack of the paper, which ran on Python 3.11.
PYBIN=""
for p in /usr/local/bin/python3.11 /usr/bin/python3.11 /bin/python3.11 python3.11; do
  if command -v "$p" >/dev/null 2>&1; then
    PYBIN="$(command -v "$p")"
    break
  fi
done

if [[ -z "$PYBIN" ]]; then
  echo "WARNING: Python 3.11 (the version of the paper's software stack) not found on this node;" >&2
  echo "falling back to the system's python3. Check whether 'module load python/3.11' (or similar) is available." >&2
  PYBIN="$(command -v python3 || true)"
fi

if [[ -z "$PYBIN" ]]; then
  echo "ERROR: no system python3 found at all." >&2
  exit 1
fi
echo "Using PYBIN=$PYBIN"
"$PYBIN" --version

echo "=== CREATING PERSISTENT VENV (rebuilt if missing or on the wrong Python version) ==="
NEED_NEW_VENV=1
if [[ -x "$VENV_PATH/bin/python" ]]; then
  EXISTING_VER="$("$VENV_PATH/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
  WANTED_VER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [[ "$EXISTING_VER" == "$WANTED_VER" ]]; then
    NEED_NEW_VENV=0
  else
    echo "Existing venv is Python $EXISTING_VER, want $WANTED_VER - rebuilding from scratch."
    rm -rf "$VENV_PATH"
  fi
fi
if [[ "$NEED_NEW_VENV" == "1" ]]; then
  "$PYBIN" -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python --version

pip install --upgrade pip setuptools wheel
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== APPLYING AUTOGLUON auto_trainer.py PATCH ==="
# Without this, automl_model_search.py crashes with
# "TypeError: unhashable type: 'slice'" the moment fit() logs hyperparameters
# (NN_TORCH's value is a dict, not a list, but the installed file always does
# hyperparameters[k][:3]). Confirmed byte-identical to our patched version except
# for that one line, on AutoGluon 0.8.2, 1.0.0 and 1.6.2 - see
# examples/CICIDS2017/scripts/patches/README.md. Located dynamically (not a
# hardcoded lib/pythonX.Y/ path) so this keeps working whichever Python version
# ended up being used above.
PATCH_SRC="$PROJECT_DIR/code/examples/CICIDS2017/scripts/patches/autogluon_auto_trainer.py"
if [[ -f "$PATCH_SRC" ]]; then
  TRAINER_FILE="$(python -c 'import autogluon.tabular.trainer.auto_trainer as m; print(m.__file__)' 2>/dev/null || echo "")"
  if [[ -n "$TRAINER_FILE" && -f "$TRAINER_FILE" ]]; then
    cp "$PATCH_SRC" "$TRAINER_FILE"
    echo "Patched $TRAINER_FILE"
  else
    echo "WARNING: could not locate the installed auto_trainer.py to patch - automl_model_search.py may crash with 'TypeError: unhashable type: slice'. Apply the patch manually (see examples/CICIDS2017/scripts/patches/README.md)." >&2
  fi
else
  echo "WARNING: patch source $PATCH_SRC not found (code/ not uploaded?) - skipping." >&2
fi

echo "=== GPU CHECK ==="
nvidia-smi || true

python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available to PyTorch in this venv - training will fall back to CPU.")
PY

python - <<'PY'
try:
    from autogluon.tabular import TabularPredictor
    print("AutoGluon import OK")
except Exception as e:
    print(f"WARNING: AutoGluon import failed: {e}")
PY

deactivate || true
echo "=== DONE - venv ready at $VENV_PATH ==="
