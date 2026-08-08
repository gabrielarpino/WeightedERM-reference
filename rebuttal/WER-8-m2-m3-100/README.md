# WER-8: author-run M2/M3 notebooks

Status: **NOTEBOOKS READY; EXPERIMENTS NOT RUN**.

This PR provides two unexecuted Jupyter notebooks derived from the final
manuscript-producing cells in the original comparison notebooks:

- [`M2_upsample.ipynb`](M2_upsample.ipynb), from cell 27 of
  `M2 Comparison Experiments.ipynb`;
- [`M3_upsample.ipynb`](M3_upsample.ipynb), from cell 13 of
  `M3 Comparison Experiments.ipynb`.

The source notebooks in this repository are byte-for-byte identical to their
counterparts in the read-only `/scratch/ga442/Weighted-ERM` reference tree. The
reference tree was consulted only as a read-only source and was not modified.

## Deliberately small changes

The scientific configuration, data generation, competitors, tuning, metrics,
and plotting code are preserved from the identified original cells. The new
notebooks change only the execution wrapper needed for this rerun:

1. `num_trials` is 100 instead of 30.
2. WERM calls use the accepted production route, which dispatches partition
   selection to exact DP. The legacy `brute_force=False` argument is retained
   because it is the original notebook API spelling; after WER-7 it no longer
   selects greedy search.
3. The outer worker count is capped at 32 and CPU affinity, and inner numerical
   library threads are set to one.
4. Logs and result pickles are written below the ignored local directory
   `rebuttal/WER-8-m2-m3-100/artifacts/`, so historical repository outputs are
   not overwritten.

No experiment, competitor, plotting cell, or notebook output was executed by
Codex for this revision. The notebooks intentionally contain no result claims.

## Run manually

From the repository root, start Jupyter through the confirmed working Conda
environment:

```bash
conda run -n weighted-erm jupyter lab
```

Open and run **one notebook at a time**. Before execution, confirm that the
kernel reports `/scratch/ga442/miniforge3/envs/weighted-erm/bin/python` in the
environment-check cell. This activation is required so `rpy2` uses the Conda R
installation containing `charcoal`, `changepoints`, `inferchange`, `moseg`,
and `myglm2`, rather than the incompatible system R installation.

The notebooks print their effective worker count before constructing the task
grid. Generated logs and pickles remain local under `artifacts/` and are ignored
by Git.
