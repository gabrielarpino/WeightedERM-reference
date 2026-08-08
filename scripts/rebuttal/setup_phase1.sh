#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

uv venv --python 3.10 .venv-phase1
uv pip sync \
  --python .venv-phase1/bin/python \
  rebuttal/phase1/requirements.lock

{
  .venv-phase1/bin/python --version
  uv pip freeze --python .venv-phase1/bin/python
} > rebuttal/phase1/.run.log

echo "Environment ready: .venv-phase1"
echo "Run log: rebuttal/phase1/.run.log"
