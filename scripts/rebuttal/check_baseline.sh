#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv-phase1/bin/python ]]; then
  echo "Missing .venv-phase1; run: bash scripts/rebuttal/setup_phase1.sh" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log_path="rebuttal/phase1/results/test_output.txt"
mkdir -p "$(dirname "$log_path")"

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
  .venv-phase1/bin/python scripts/rebuttal/baseline_smoke.py \
  --requested-workers "${WERM_PHASE1_WORKERS:-1}" 2>&1 \
  | tee "$log_path"
