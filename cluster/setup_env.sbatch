#!/usr/bin/env bash
#SBATCH --job-name=automl-nids-setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=automl-nids-setup-%j.out
#SBATCH --error=automl-nids-setup-%j.err

# One-time job: builds a PERSISTENT venv (not per-job scratch, unlike the
# classification-leaf example this is adapted from) so every later job just
# activates it instead of reinstalling autogluon/torch/etc. every time - that
# install is heavy (many GB of transitive deps) and we submit many small jobs
# (multi-seed reruns, sweeps...), not one long one.
#
# This job only installs packages, so it runs on a CPU partition and reserves no GPU (the GPU is checked by the jobs that use it).
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
# PYTHON311 (optional) is the path of a Python 3.11 that every node can run, e.g. one installed on the shared file system
# (see "Python for the CPU nodes" in cluster/README.md); it is tried first.
for p in "${PYTHON311:-}" /usr/local/bin/python3.11 /usr/bin/python3.11 /bin/python3.11 python3.11; do
  if [[ -n "$p" ]] && command -v "$p" >/dev/null 2>&1; then
    PYBIN="$(command -v "$p")"
    break
  fi
done

if [[ -z "$PYBIN" && -x "$VENV_PATH/bin/python" ]]; then
  # never rebuild a working venv with a different Python just because this node lacks python3.11
  echo "WARNING: Python 3.11 not found on this node; keeping the existing venv at $VENV_PATH and reinstalling the requirements into it." >&2
  PYBIN="$VENV_PATH/bin/python"
fi

if [[ -z "$PYBIN" ]]; then
  # the pinned stack (torch 2.4.1...) has no wheels for newer Pythons: do not build an environment with another version
  echo "ERROR: Python 3.11 (the version of the paper's software stack) not found on $(hostname). Set PYTHON311=/path/to/python3.11 (a Python every node can run)." >&2
  exit 1
fi

if [[ -z "$PYBIN" ]]; then
  echo "ERROR: no system python3 found at all." >&2
  exit 1
fi
echo "Using PYBIN=$PYBIN"
"$PYBIN" --version

if [[ -e "$VENV_PATH" && ! -x "$VENV_PATH/bin/python" ]]; then
  # a venv points at the interpreter it was built with; on a node that lacks it the link is broken, and recreating the venv here
  # with another Python would damage it
  echo "ERROR: $VENV_PATH exists but its Python is not usable on $(hostname) (see $VENV_PATH/pyvenv.cfg). Run this job on a node that has that interpreter. The venv was not touched." >&2
  exit 1
fi

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

echo "=== FIXING AUTOGLUON auto_trainer.py (one expression) ==="
# AutoGluon 1.6.2 already guards the expression that logs hyperparameters[k][:3] (NN_TORCH's value is a dict, not a list), so on the
# pinned version this step prints "already patched" and changes nothing. It stays as a safety net for AutoGluon versions without that
# guard, where automl_model_search.py would stop with "TypeError: unhashable type: 'slice'" the moment fit() logs the hyperparameters.
# The fix edits only that expression (patches/apply_autogluon_patch.py), because the whole file differs between AutoGluon versions: a
# copy taken from another version makes the import of AutoGluon fail. It is applied on a pristine copy of the package, so a file left
# by an earlier patch is discarded.
AG_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("autogluon.tabular"))')"
pip install --quiet --force-reinstall --no-deps "autogluon.tabular==$AG_VERSION"
PATCH_SCRIPT="$PROJECT_DIR/code/examples/CICIDS2017/scripts/patches/apply_autogluon_patch.py"
if [[ -f "$PATCH_SCRIPT" ]]; then
  python "$PATCH_SCRIPT"
else
  echo "WARNING: $PATCH_SCRIPT not found (code/ not uploaded?) - automl_model_search.py may crash with 'TypeError: unhashable type: slice'." >&2
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
    print("CUDA not visible in this job (expected: it runs on a CPU partition without a GPU). The jobs that use the GPU print the device they get.")
PY

python - <<'PY'
try:
    from autogluon.tabular import TabularPredictor
    print("AutoGluon import OK")
except Exception as e:
    print(f"ERROR: AutoGluon import failed: {e}")
    raise SystemExit(1)
PY

deactivate || true
echo "=== DONE - venv ready at $VENV_PATH ==="
