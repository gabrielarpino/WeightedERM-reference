#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="/scratch/ga442/gab-code-env/papers/WeightedERM-reference"
queue_dir="$repo_dir/results/notebook_queue_20260908_m3_redo3_delta6_alphas"
status_file="$queue_dir/status.txt"
log_file="$queue_dir/run.log"

mkdir -p "$queue_dir"
cd "$repo_dir"

on_error() {
    code=$?
    printf 'failed exit_code=%s finished_at=%s\n' "$code" "$(date --iso-8601=seconds)" > "$status_file"
    exit "$code"
}
trap on_error ERR

printf 'running pid=%s started_at=%s order=alpha0.1,alpha0.3 trials=100 jobs=36\n' \
    "$$" "$(date --iso-8601=seconds)" > "$status_file"

/scratch/ga442/miniforge3/bin/conda run --no-capture-output -n weighted-erm \
    env PYTHONPATH=. WERM_N_JOBS=36 JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python scripts/run_m3_redo3_delta6_extensions.py >> "$log_file" 2>&1

printf 'complete finished_at=%s order=alpha0.1,alpha0.3 trials=100 jobs=36 outputs_embedded=true\n' \
    "$(date --iso-8601=seconds)" > "$status_file"
