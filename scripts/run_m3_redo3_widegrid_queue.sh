#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="/scratch/ga442/gab-code-env/papers/WeightedERM-reference"
queue_dir="$repo_dir/results/notebook_queue_20260904_m3_redo3_widegrid"
status_file="$queue_dir/status.txt"
log_file="$queue_dir/M3_upsample_redo3.log"

mkdir -p "$queue_dir"
cd "$repo_dir"

on_error() {
    code=$?
    printf 'failed exit_code=%s finished_at=%s\n' "$code" "$(date --iso-8601=seconds)" > "$status_file"
    exit "$code"
}
trap on_error ERR

printf 'running pid=%s started_at=%s notebook=M3_upsample_redo3.ipynb trials=100 jobs=36\n' "$$" "$(date --iso-8601=seconds)" > "$status_file"

env \
    PYTHONPATH=. \
    WERM_N_JOBS=36 \
    JAX_PLATFORM_NAME=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /scratch/ga442/miniforge3/bin/conda run --no-capture-output -n weighted-erm \
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        M3_upsample_redo3.ipynb >> "$log_file" 2>&1

checkpoint_dirs=(
    "results/checkpoints/M3_upsample_d_redo_full_cv_delta3_full_seed42_p200_L4_schema13"
    "results/checkpoints/M3_upsample_d_redo_full_cv_delta3_alpha0.3_full_seed42_p200_L4_schema13"
    "results/checkpoints/M3_upsample_d_redo_full_cv_delta3_alpha0.1_full_seed42_p200_L4_schema13"
    "results/checkpoints/M3_upsample_d_redo_full_cv_delta6_full_seed42_p200_L4_schema13"
)
artifacts=(
    "results/M3d_redo_delta3_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl"
    "results/M3d_redo_delta3_alpha03_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl"
    "results/M3d_redo_delta3_alpha01_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl"
    "results/M3d_redo_delta6_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl"
)

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    checkpoint_count=$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'setting_*_trial_*.pkl' | wc -l)
    if [[ "$checkpoint_count" -ne 800 ]]; then
        printf 'Expected 800 checkpoints in %s, found %s\n' "$checkpoint_dir" "$checkpoint_count" >&2
        exit 1
    fi
done

for artifact in "${artifacts[@]}"; do
    test -s "$artifact"
done

printf 'complete finished_at=%s notebook=M3_upsample_redo3.ipynb trials=100 jobs=36\n' "$(date --iso-8601=seconds)" > "$status_file"
