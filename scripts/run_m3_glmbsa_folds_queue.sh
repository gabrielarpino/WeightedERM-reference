#!/usr/bin/env bash
set -Eeuo pipefail

repo=/scratch/ga442/gab-code-env/papers/WeightedERM-reference
queue_dir="$repo/results/notebook_queue_20260903_m3_glmbsa_2-5-10_full"
status_file="$queue_dir/status.txt"
log_file="$queue_dir/run.log"
conda=/scratch/ga442/miniforge3/bin/conda

mkdir -p "$queue_dir"
cd "$repo"
exec >>"$log_file" 2>&1

status() {
    printf '%s\n' "$1" >"$status_file"
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1"
}

fail() {
    exit_code=$?
    status "failed (exit $exit_code)"
    exit "$exit_code"
}
trap fail ERR

run_notebook() {
    env -u WERM_SMOKE \
        WERM_N_JOBS=36 \
        JAX_PLATFORM_NAME=cpu \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$conda" run -n weighted-erm jupyter nbconvert \
        --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 "$1"
}

require_checkpoint_count() {
    checkpoint_dir=$1
    expected=$2
    count=$(find "$checkpoint_dir" -maxdepth 1 -name 'setting_*_trial_*.pkl' | wc -l)
    test "$count" -eq "$expected"
}

status "running_delta3_alpha05_then_alpha03_then_alpha01_full_serial"
run_notebook M3_upsample_redo2.ipynb

require_checkpoint_count \
    results/checkpoints/M3_upsample_d_redo_full_cv_delta3_full_seed42_p200_L4_schema12 800
require_checkpoint_count \
    results/checkpoints/M3_upsample_d_redo_full_cv_delta3_alpha0.3_full_seed42_p200_L4_schema12 800
require_checkpoint_count \
    results/checkpoints/M3_upsample_d_redo_full_cv_delta3_alpha0.1_full_seed42_p200_L4_schema12 800
test -s results/M3d_redo_delta3_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42_100trials.pkl
test -s results/M3d_redo_delta3_alpha03_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42_100trials.pkl
test -s results/M3d_redo_delta3_alpha01_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42_100trials.pkl

status "running_delta6_full_after_all_delta3_experiments"
run_notebook M3_upsample_redo_delta6.ipynb

require_checkpoint_count \
    results/checkpoints/M3_upsample_d_redo_full_cv_delta6_full_seed42_p200_L4_schema12 800
test -s results/M3d_redo_delta6_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42_100trials.pkl

status "complete"
