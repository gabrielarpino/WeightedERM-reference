# Experiment Babysitting Handoff

Last updated: 2026-08-14 15:16 BST.

## Read this first

An M1d run is **active and must not be restarted, killed, or duplicated**. It is a
detached process (`PPID=1`, no controlling TTY) and survives any agent session ending.
Your job is read-only monitoring, then validation, then launching M1e.

Never replace `run_charcoal` with `run_charcoal_updated`, and do not alter any method or
parameter, without explicit user authorization.

## Sequencing

Run the exact experiment cells from `M1_upsample.ipynb` sequentially, in isolated terminal
Python processes, never in the user's Jupyter kernel:

1. M1d: heading cell 33, experiment code cell **34** (`M1d alpha 1.0 (non-sparse, Gaussian noise)`) — **in progress**.
2. M1e: heading cell 35, experiment code cell **36** (`M1e alpha 0.05 (very sparse, Gaussian noise)`) — not started.

Do not start M1e until M1d has exited successfully and its pickle plus fresh logs have been validated.

## Active M1d: adopt; do not restart or terminate

`PID 2775295`

`/scratch/ga442/miniforge3/envs/weighted-erm/bin/python /tmp/run_monitored_notebook_cell.py M1_upsample.ipynb 34 /tmp/m1d_v2_workers.log`

Parent log: `/tmp/m1d_v2_parent.log`
Worker log: `/tmp/m1d_v2_workers.log`

This is not the user's Jupyter kernel. Do not launch another M1d process. The
monitoring-only runner at `/tmp/run_monitored_notebook_cell.py` reads the exact cell
source, changes Joblib to `return_as="generator"`, adds aggregation progress, and
redirects worker stdout/stderr. It preserves result ordering and experiment calculations.

These are fresh `m1d_v2` logs. Do not mix prior aborted-run logs into diagnostics.

### Snapshot at handoff (2026-08-14 15:16 BST)

- Completed/yielded: **498/900 (55.3%)**; submitted: 540/900.
- Elapsed: **39h 58m** (process start ~23:18 on 2026-08-12).
- Children: 38 — 36 compute workers plus 2 idle Joblib manager threads at 0% CPU.
- Aggregate CPU: ~3593% (pool saturated).
- Caught charcoal failures: **210** instances of
  `Error in cpreg_stats[i] <- tmp$stats[tmp$cp]: replacement has length zero`.
  This count has been frozen since before 2026-08-13 15:45 — no new method-level failures.
- Tracebacks: 0. Socket errors: 0.
- No M1d result pickle yet.

Note: an earlier version of this document recorded ~22h56m elapsed at 2026-08-13 13:19,
which back-dated the start incorrectly. `ps` elapsed time is authoritative; the PID,
command line, and logs have been continuous throughout.

## Behavior you will see — and what is actually wrong

**Flat progress counters are normal.** Under `return_as="generator"`, nothing yields until
an in-flight trial finishes. Observed pauses of 70 min to ~2h 14m at trials 400, 433, 472,
and 498 all resolved on their own, then released several trials at once. tqdm's remaining-time
estimate swings wildly (35h to 183h) as a direct artifact of this batch-yield pattern — treat
it as noise, not a projection.

**Trial durations are growing sharply.** Early in the run individual trials took ~5h; at
handoff three workers had each been on a single trial for **26h+**. This is the tail of the
delta sweep, not a fault. The practical consequence is that the remaining 402 trials will
take considerably longer than any tqdm figure suggests. If the wall-clock matters for a
deadline, raise it with the user rather than changing anything.

**A stall is only real if liveness degrades.** When the counter is flat, check:

```
ls -la --time-style=+%H:%M:%S /tmp/m1d_v2_workers.log     # mtime should be ~now
ps --ppid 2775295 -o %cpu= | awk '{s+=$1} END {print s}'  # should be ~3500-3600
```

Escalate only if the worker log stops growing, aggregate CPU collapses, tracebacks or
socket errors appear, or the process exits.

## Monitoring

Sparse read-only checks every 30-45 minutes; extend the gap for slower later trials.
Report completed/900 and percent, elapsed time, warning/error counts, caught method
failures, and process/worker health.

A ready-made poller lives at `/tmp/watch_m1d.sh`. It emits one status line per 35 minutes
and exits when the process dies or the pickle appears. Re-arm it as a persistent background
monitor at the start of your session:

`bash /tmp/watch_m1d.sh`

(The previous agent's monitor task ended with its session. This does not affect PID 2775295.)

Individual read-only checks:

`ps -p 2775295 -o pid,etime,stat,%cpu,%mem,cmd`

`tr '\r' '\n' < /tmp/m1d_v2_parent.log | rg 'Completed simulation trials:|Running simulation trials:' | tail -n 4`

`ps --ppid 2775295 -o stat= | sort | uniq -c`

`rg -n -i 'traceback|warning|socket error|connection reset|broken pipe|nan|replacement has length zero' /tmp/m1d_v2_parent.log /tmp/m1d_v2_workers.log | tail -n 80`

Do not `cat` the worker log — it is 6+ MB and mostly carriage-return progress bars. Filter it.

## Expected output paths

M1d:

`results/M1_upsample_d_alpha10_cp025060_charcoal_McScan_WERM-unp_DCDP_DPDU_MOSEG_results_p200_L4_delta123456789_noisecale0.316_seed46.pkl`

M1e:

`results/M1_upsample_e_alpha005_cp025060_McScan_WERM-unp_DCDP_DPDU_MOSEG_results_p200_L4_delta123456789_noisecale0.316_seed46.pkl`

## Validation after each completed run

Using the weighted-erm Python environment, load the pickle and verify:

- `params['num_trials'] == 100` and nine delta values.
- Expected methods: M1d has `charcoal`, `McScan`, `WERM-unp`, `DCDP`, `DPDU`, `MOSEG`;
  M1e has `McScan`, `WERM-unp`, `DCDP`, `DPDU`, `MOSEG`.
- All result arrays have shape `(9, 100)`.
- Per-delta successful-trial counts, NaNs/nonfinite values, and changepoint-list lengths.
- Every infinite Hausdorff score corresponds exactly to a legitimate zero-changepoint
  detection; report mismatches.
- Fresh parent and worker log scan for warnings/errors/tracebacks/socket errors,
  distinguishing caught per-method failures from uncaught process failures. The ~210
  charcoal `replacement has length zero` errors are caught per-method failures and are
  expected; confirm they are reflected as failed charcoal trials rather than silent successes.

Do not begin M1e if M1d lacks its pickle, exited nonzero, or validation finds a material
inconsistency; report that condition to the user.

## M1e launch (only after M1d validation)

Use fresh log names and a persistent terminal session:

`PYTHONPATH=/scratch/ga442/gab-code-env/papers/WeightedERM-reference R_HOME=/scratch/ga442/miniforge3/envs/weighted-erm/lib/R OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 /scratch/ga442/miniforge3/envs/weighted-erm/bin/python /tmp/run_monitored_notebook_cell.py M1_upsample.ipynb 36 /tmp/m1e_alpha005_workers_20260813.log > /tmp/m1e_alpha005_parent_20260813.log 2>&1`

Then monitor M1e sparsely with its own fresh logs and perform the same complete validation
before reporting both output paths and concise diagnostics.
