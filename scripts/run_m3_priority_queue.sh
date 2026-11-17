#!/usr/bin/env bash
set -Eeuo pipefail

repo=/scratch/ga442/gab-code-env/papers/WeightedERM-reference
queue_dir="$repo/results/notebook_queue_20260903_m3_priority_glmbsa2fold"
status_file="$queue_dir/status.txt"
log_file="$queue_dir/run.log"
# Keep the execution copy in the repository root.  Nbconvert uses the
# notebook's directory as its working directory, which is required so the
# notebook imports this checkout and writes to this checkout's results/ tree.
priority_nb="$repo/.M3_upsample_redo2_base_priority.ipynb"
python=/scratch/ga442/miniforge3/envs/weighted-erm/bin/python
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

status "running_priority_delta3_alpha05_100trials"
run_notebook "$priority_nb"

priority_checkpoint="$repo/results/checkpoints/M3_upsample_d_redo_full_cv_delta3_full_seed42_p200_L4_schema11"
priority_count=$(find "$priority_checkpoint" -maxdepth 1 -name 'setting_*_trial_*.pkl' | wc -l)
test "$priority_count" -eq 800
test -s "$repo/results/M3d_redo_delta3_glmbsa2fold_eleven_methods_folds2-5-10_seed42_100trials.pkl"

status "priority_complete_merging_plots"
"$python" scripts/m3_priority_notebook.py merge-priority \
    M3_upsample_redo2.ipynb "$priority_nb"

# The corrected alpha smoke checkpoints already completed before the queue was
# reprioritized. Delta 6 was the only pending smoke test.
status "priority_complete_running_delta6_smoke"
WERM_SMOKE=1 \
WERM_N_JOBS=2 \
JAX_PLATFORM_NAME=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$conda" run -n weighted-erm jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 M3_upsample_redo_delta6.ipynb

status "running_alpha03_then_alpha01_full_after_priority"
run_notebook M3_upsample_redo2.ipynb

status "running_delta6_full_after_priority"
run_notebook M3_upsample_redo_delta6.ipynb

status "complete"
