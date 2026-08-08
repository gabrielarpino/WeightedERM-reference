#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

log_path="rebuttal/phase1/results/test_output.txt"
mkdir -p "$(dirname "$log_path")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x .venv-phase1/bin/python ]]; then
  echo "Missing .venv-phase1; run: bash scripts/rebuttal/setup_phase1.sh" | tee "$log_path"
  echo "PHASE1 CHECK: INCOMPLETE"
  echo "OPEN FIRST: rebuttal/phase1/README.md"
  echo "TEST LOG: rebuttal/phase1/results/test_output.txt"
  exit 2
fi

baseline_status=0
pytest_status=0
audit_status=0

bash scripts/rebuttal/check_baseline.sh
baseline_status=$?

.venv-phase1/bin/python -m pytest -q tests/test_phase1_dp.py tests/test_dp_default.py 2>&1 \
  | tee -a "$log_path"
pytest_status=${PIPESTATUS[0]}

.venv-phase1/bin/python scripts/rebuttal/phase1_audit.py \
  --requested-workers "${WERM_PHASE1_WORKERS:-1}" 2>&1 \
  | tee -a "$log_path"
audit_status=${PIPESTATUS[0]}

if [[ $pytest_status -ne 0 || $audit_status -ne 0 ]]; then
  phase1_status="FAIL"
elif [[ $baseline_status -ne 0 ]]; then
  phase1_status="INCOMPLETE"
else
  phase1_status="PASS"
fi

echo "PHASE1 CHECK: $phase1_status"
echo "OPEN FIRST: rebuttal/phase1/README.md"
echo "TEST LOG: rebuttal/phase1/results/test_output.txt"

if [[ $baseline_status -ne 0 || $pytest_status -ne 0 || $audit_status -ne 0 ]]; then
  exit 1
fi
