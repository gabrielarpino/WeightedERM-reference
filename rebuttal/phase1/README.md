# Phase 1 reproducibility setup

This first stacked change adds only the uv environment and reduced baseline
checks. It does not execute or rewrite the research notebooks.

```bash
bash scripts/rebuttal/setup_phase1.sh
bash scripts/rebuttal/check_baseline.sh
```

On this server, missing historical artifacts are checked read-only under
`/scratch/ga442/Weighted-ERM`. An equivalent source tree can be selected with:

```bash
export WERM_HISTORICAL_ROOT=/path/to/Weighted-ERM
```

The setup and checks never write to or modify that historical tree.

The environment is created in ignored directory `.venv-phase1`. Python and
package versions from the local run are written to ignored file
`rebuttal/phase1/.run.log`.

To refresh the compiled lock deliberately:

```bash
uv pip compile \
  --python-version 3.10 \
  --output-file rebuttal/phase1/requirements.lock \
  rebuttal/phase1/requirements.in
```

The latest readable check log is saved at `results/test_output.txt`.

See [baseline.md](baseline.md) for the five workflow classifications.
