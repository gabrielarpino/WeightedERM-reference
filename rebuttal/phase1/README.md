# Phase 1 reproducibility setup

This first stacked change adds only the uv environment and reduced baseline
checks. It does not execute or rewrite the research notebooks.

```bash
bash scripts/rebuttal/setup_phase1.sh
bash scripts/rebuttal/check_baseline.sh
```

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
