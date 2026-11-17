#!/usr/bin/env bash
# Submits everything that follows the AutoML search of ONE dataset, as separate SLURM jobs chained with dependencies, through
# run_python.sbatch (no GPU) and run_experiment.sbatch (GPU). Run it from the project directory (~/automl_nids) on the login node.
#
#   bash code/cluster/submit_experiments.sh <DATASET> --after <AutoML job id> [options]
#
# <DATASET> is the folder name under $PROJECT_DIR/data (CICIDS2017_improved, UNSW_NB15, CICIoT2023, TON_IoT). It must contain
# balanced_binary/split_<seed>/ (dataset_creator.py --seeds ...) and balanced_binary/split_0/automl_search/ (the AutoML search
# of split_0), or the jobs given to --after must create them.
#
# What is submitted (names of the experiments: examples/CICIDS2017/scripts/utils/experiments.py):
#   <tag>-rank, <tag>-select   feature ranking by sequential backward elimination on split_0, and the list of kept features
#   <tag>-exp1 ... <tag>-exp8  Experiments 1-5 of the paper and the three alternative input representations (6-8), each over
#                              the seeds, seed i trained and tested on split_i (run_multiseed.py --experiment N)
#   <tag>-base-s<i>            Random Forest, Logistic Regression and XGBoost on split_i
#   <tag>-analyses             PTQ, AutoGluon/replica agreement, noise, input resolution, size and latency, quantized engines
#   <tag>-sweep                Experiment 4 for a range of rescaling factors
#
# Options:
#   --after JOBID[:JOBID...]   jobs that must finish OK first (normally the AutoML search); the checks of the splits and of the
#                              predictor are then skipped, because those jobs create them
#   --experiments "1 2 3 4 5 6 7 8"   which experiments to run (Experiment 5 needs the ranking and selection jobs)
#   --seeds "0 1 2 3 4"        seeds; seed i uses split_i and training seed i
#   --epochs 100               epochs of every training run (also of the rescaling-factor sweep); small values are for smoke tests
#   --sweep-factors "1e2 ..."  factors of the sweep (default: 1e2 ... 1e8)
#   --tolerance 1.0            Experiment 5 keeps the smallest feature subset within this many accuracy points (fine-tuning split)
#                              of the accuracy with all the features (select_features.py)
#   --threads N                torch/OpenMP threads of every job that trains or measures a model (default 6). QAT results
#                              depend on this number, so keep it identical for every dataset and partition
#   --run-name NAME            write the runs to compressed_models_NAME (and name the jobs <tag>-NAME-exp<N>) instead of compressed_models, and
#                              submit only the experiments: the same experiments can then be repeated with other threads or seeds without
#                              overwriting anything. For the paired collapse study (analyze_collapse_pairs.py):
#                                for t in 4 6 12; do
#                                  bash code/cluster/submit_experiments.sh <DATASET> --after <AutoML job> --experiments "3 4" \
#                                    --seeds "0 1 2 3 4 5 6 7 8 9" --threads $t --run-name t$t
#                                done
#   --keep-batchnorm           run every experiment with the BatchNorm layers of the AutoML network kept during the FP32 training and folded into the
#                              next Linear before the quantization (autogluon_to_pytorch_qat.py --keep_batchnorm). For a model whose BatchNorm-free
#                              replica does not train (a deep network with BatchNorm). Same experiment numbers 1-8: the runs go to
#                              compressed_models_bnfold, the analyses to analyses_bnfold, the jobs are named <tag>-bnfold-exp<N>. The ranking, the
#                              baselines and the sweep are still submitted: add --no-ranking --no-baselines when the dataset already has them
#   --sbatch-args "ARGS"       extra sbatch arguments for every job, e.g. "--partition=<gpu partition> --cpus-per-task=6 --mem=16G --time=2-00:00:00"
#   --no-baselines | --no-analyses | --no-sweep | --no-ranking   leave that part out (--no-ranking: Experiment 5 uses an
#                              existing $BASE/split_0/automl_search/feature_ranking_custom_sbe/important_features.txt)
#   --dry-run                  print the sbatch commands without submitting
#
# Jobs without GPU (ranking, selection, baselines, analyses) go through run_python.sbatch to the partition $CPU_PARTITION (default: cpu)
# without a GPU; the experiments and the rescaling-factor sweep train on the GPU and go through run_experiment.sbatch (GPU partition,
# one CPU per thread of --threads, 16 GB).
#
# Every submitted job id is appended to $PROJECT_DIR/submitted_jobs.tsv (job id, name, time) so that `sacct -j` can report them.
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
fi
DATASET="$1"; shift

PROJECT_DIR="${PROJECT_DIR:-$HOME/automl_nids}"
EXPERIMENTS="1 2 3 4 5 6 7 8"
SEEDS="0 1 2 3 4"
EPOCHS=100
SWEEP_FACTORS=""
AFTER=""
TOLERANCE=1.0
THREADS=6
RUN_NAME=""
SBATCH_ARGS=""
BASELINES=1; ANALYSES=1; SWEEP=1; RANKING=1
KEEP_BN=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments) EXPERIMENTS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --sweep-factors) SWEEP_FACTORS="$2"; shift 2;;
    --after) AFTER="$2"; shift 2;;
    --tolerance) TOLERANCE="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --sbatch-args) SBATCH_ARGS="$2"; shift 2;;
    --no-baselines) BASELINES=0; shift;;
    --no-analyses) ANALYSES=0; shift;;
    --no-sweep) SWEEP=0; shift;;
    --no-ranking) RANKING=0; shift;;
    --keep-batchnorm) KEEP_BN=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

BASE="$PROJECT_DIR/data/$DATASET/balanced_binary"
# the BatchNorm-folded protocol writes to its own folders (a --run-name picks another folder name for the runs)
FOLDER_NAME="$RUN_NAME"
ANALYSES_DIR=analyses
BN_OPTS=()                 # options of the analysis and sweep jobs for the BatchNorm-folded protocol
if [[ $KEEP_BN -eq 1 ]]; then
  [[ -z "$FOLDER_NAME" ]] && FOLDER_NAME=bnfold
  ANALYSES_DIR=analyses_bnfold
  BN_OPTS=(--models_dir "compressed_models_$FOLDER_NAME" --out_dir "$ANALYSES_DIR" --keep_batchnorm)
fi
OUT_DIR="$BASE/compressed_models${FOLDER_NAME:+_$FOLDER_NAME}"
if [[ -n "$RUN_NAME" ]]; then BASELINES=0; ANALYSES=0; SWEEP=0; RANKING=0; fi
PREDICTOR="$BASE/split_0/automl_search"
RANKING_DIR="$PREDICTOR/feature_ranking_custom_sbe"
FEATURES="$RANKING_DIR/important_features.txt"
SCRIPTS="code/examples/CICIDS2017/scripts"          # relative to $PROJECT_DIR, where run_python.sbatch runs the script
RUNNER="$PROJECT_DIR/code/cluster/run_python.sbatch"                  # jobs without GPU
GPU_RUNNER="$PROJECT_DIR/code/cluster/run_experiment.sbatch"           # experiments and sweep: they train on the GPU
TELEMETRY_DIR="${TELEMETRY_DIR:-$PROJECT_DIR/telemetry}"
case "$DATASET" in
  CICIDS2017_improved) TAG=cic17;;
  UNSW_NB15) TAG=unsw;;
  CICIoT2023) TAG=ciciot;;
  TON_IoT) TAG=ton;;
  *_smoke) TAG=smoke;;      # throwaway copy of a dataset used to test the whole chain
  *) TAG="$DATASET";;
esac
ATAG="$TAG"; [[ $KEEP_BN -eq 1 ]] && ATAG="$TAG-bnfold"      # prefix of the analysis and sweep jobs
# jobs that train or measure models run with a fixed number of threads, and record machine and resource use
ENV_TRAIN="ALL,OMP_NUM_THREADS=$THREADS,TELEMETRY_DIR=$TELEMETRY_DIR"
ENV_PLAIN="ALL,TELEMETRY_DIR=$TELEMETRY_DIR"
# jobs that do not use the GPU (ranking, baselines, analyses...) go to a CPU partition and reserve no GPU
CPU_PARTITION="${CPU_PARTITION:-cpu}"
NO_GPU=(--partition="$CPU_PARTITION" --gres= --gpus=0)

# With --after the splits and the predictor may not exist yet (the jobs we wait for create them): nothing to check then.
if [[ $DRY_RUN -eq 0 && -z "$AFTER" ]]; then
  [[ -d "$BASE/split_0" ]] || { echo "missing $BASE/split_0 (run dataset_creator.py first)" >&2; exit 1; }
  [[ -d "$PREDICTOR" ]] || { echo "missing $PREDICTOR (run the AutoML search first)" >&2; exit 1; }
  for s in $SEEDS; do
    [[ -f "$BASE/split_$s/test.csv" ]] || { echo "missing $BASE/split_$s/test.csv (create the splits with --seeds including $s)" >&2; exit 1; }
  done
fi

submit() {   # submit <job name> <dependency job ids or ""> <sbatch options...> -- <script args...>; prints the job id
  local name="$1" deps="$2"; shift 2
  local opts=()
  while [[ "$1" != "--" ]]; do opts+=("$1"); shift; done
  shift
  local dep=()
  [[ -n "$deps" ]] && dep=(--dependency="afterok:$deps")
  # shellcheck disable=SC2206
  local cmd=(sbatch --parsable --job-name="$name" ${dep[@]+"${dep[@]}"} "${opts[@]}" $SBATCH_ARGS "$RUNNER" "$@")
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '%q ' "${cmd[@]}" >&2; echo >&2
    echo "DRYRUN-$name"
    return
  fi
  local id
  id="$("${cmd[@]}")"
  printf '%s\t%s\t%s\n' "$id" "$name" "$(date -Is)" >> "$PROJECT_DIR/submitted_jobs.tsv"
  echo "$id"
}
submit_gpu() { local RUNNER="$GPU_RUNNER"; submit "$@"; }      # same as submit, through the GPU runner (bash scoping: submit reads this RUNNER)
join() { local IFS=:; echo "$*"; }
wants() { [[ " $EXPERIMENTS " == *" $1 "* ]]; }

# --- feature selection for Experiment 5
SELECT_JOB=""
if wants 5; then
  if [[ $RANKING -eq 1 ]]; then
    RANK_JOB="$(submit "$TAG-rank" "$AFTER" "${NO_GPU[@]}" --cpus-per-task=6 --mem=32G --time=1-00:00:00 --export="$ENV_PLAIN" -- \
        "$SCRIPTS/feature_ranking.py" --dataset-directory "$BASE/split_0" --predictor-directory "$PREDICTOR" \
        --algorithm custom_sbe --cpus 6 --time-limit 40000 --metric accuracy --recursive)"
    echo "$TAG-rank -> job $RANK_JOB"
    SELECT_JOB="$(submit "$TAG-select" "$RANK_JOB" "${NO_GPU[@]}" --cpus-per-task=2 --mem=8G --time=00:30:00 --export="$ENV_PLAIN" -- \
        "$SCRIPTS/select_features.py" --ranking_dir "$RANKING_DIR" --tolerance "$TOLERANCE")"
    echo "$TAG-select -> job $SELECT_JOB"
  elif [[ $DRY_RUN -eq 0 && -z "$AFTER" && ! -f "$FEATURES" ]]; then
    echo "Experiment 5 needs $FEATURES (run the ranking and select_features.py, or drop --no-ranking)" >&2; exit 1
  fi
fi

# --- the experiments
EXP_JOBS=()
for n in $EXPERIMENTS; do
  deps="$AFTER"
  extra=()
  [[ $KEEP_BN -eq 1 ]] && extra+=(--keep_batchnorm)
  if [[ "$n" == 5 ]]; then
    extra+=(--important_features_file "$FEATURES")
    [[ -n "$SELECT_JOB" ]] && deps="$(join ${AFTER:+"$AFTER"} "$SELECT_JOB")"
  fi
  id="$(submit_gpu "$TAG${FOLDER_NAME:+-$FOLDER_NAME}-exp$n" "$deps" --cpus-per-task="$THREADS" --export="$ENV_TRAIN" -- \
        experiment --experiment "$n" ${extra[@]+"${extra[@]}"} --predictor_path "$PREDICTOR" \
        --data_dir_template "$BASE/split_{seed}" --output_dir "$OUT_DIR" \
        --epochs "$EPOCHS" --batch_size 256 --seeds $SEEDS)"
  echo "$TAG${FOLDER_NAME:+-$FOLDER_NAME}-exp$n -> job $id"
  EXP_JOBS+=("$id")
done

# --- classical baselines of every split
if [[ $BASELINES -eq 1 ]]; then
  for s in $SEEDS; do
    id="$(submit "$TAG-base-s$s" "$AFTER" "${NO_GPU[@]}" --cpus-per-task=6 --mem=16G --export="$ENV_PLAIN" -- \
          "$SCRIPTS/baselines_classical_ml.py" --data_dir "$BASE/split_$s" --output_dir "$BASE/baselines/split_$s")"
    echo "$TAG-base-s$s -> job $id"
  done
fi

# --- analyses that need the trained models (after all the experiments of this run)
if [[ $ANALYSES -eq 1 && ${#EXP_JOBS[@]} -gt 0 ]]; then
  stages="ptq equivalence noise resolution tradeoff"
  wants 5 && stages="$stages engines"      # the engine comparison also re-evaluates Experiment 5
  id="$(submit "$ATAG-analyses" "$(join "${EXP_JOBS[@]}")" "${NO_GPU[@]}" --cpus-per-task="$THREADS" --mem=32G --export="$ENV_TRAIN" -- \
        "$SCRIPTS/run_analyses.py" --data_root "$BASE" --seeds $SEEDS --stages $stages ${BN_OPTS[@]+"${BN_OPTS[@]}"})"
  echo "$ATAG-analyses -> job $id"
fi
if [[ $SWEEP -eq 1 ]]; then
  sweep_args=(--sweep_epochs "$EPOCHS")
  [[ -n "$SWEEP_FACTORS" ]] && sweep_args+=(--sweep_factors $SWEEP_FACTORS)
  id="$(submit_gpu "$ATAG-sweep" "$AFTER" --cpus-per-task="$THREADS" --export="$ENV_TRAIN" -- \
        sweep --data_root "$BASE" "${sweep_args[@]}" ${BN_OPTS[@]+"${BN_OPTS[@]}"})"
  echo "$ATAG-sweep -> job $id"
fi
