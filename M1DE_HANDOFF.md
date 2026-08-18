# Handoff: M1d / M1e experiment babysitting (M1_upsample.ipynb)

Written 2026-08-13 ~04:25 local. Copy this whole file into a new LLM session.

---

## 1. Objective

Run and monitor two experiment cells from `M1_upsample.ipynb`, **strictly sequentially**,
in a separate terminal Python process (never in the user's live Jupyter kernel):

1. **M1d** — the code cell immediately below the markdown heading
   `# M1d alpha 1.0 (non-sparse, Gaussian noise)`. **Currently cell index 34**
   (markdown heading is cell 33). *Verified 2026-08-12.*
2. **M1e** — the code cell immediately below `# M1e alpha 0.05 (very sparse, Gaussian noise)`.
   **Currently cell index 36** (markdown heading is cell 35). *Verified 2026-08-12.*

Do **not** start M1e until M1d has finished *and* its pickle has been validated.

Re-verify the indices before use:

```bash
cd /scratch/ga442/gab-code-env/papers/WeightedERM-reference
/scratch/ga442/miniforge3/envs/weighted-erm/bin/python -c "
import json
nb=json.load(open('M1_upsample.ipynb'))
for i,c in enumerate(nb['cells']):
    print(i, c['cell_type'], repr(''.join(c['source']).strip().split(chr(10))[0][:90]))
"
```

---

## 2. Hard constraints from the user

- Run the **exact** experiment code from the notebook cells. Do **not** modify the
  experiment, substitute algorithms, remove methods, or change parameters to work
  around failures. Report material problems instead.
- Do **not** replace `run_charcoal` with `run_charcoal_updated` unless the user
  explicitly authorizes it.
- Do **not** touch or interfere with the user's running Jupyter kernel
  (PID 917230, kernel `9afd898f-...`). Separate terminal process only.
- Monitor at reasonably spaced intervals (~every few minutes to an hour is fine).
  Do **not** poll frequently/repetitively — it wastes tokens.
- Use fresh log filenames so old data is not appended/counted.
- Persist until both experiments complete and validate, unless a systemic problem
  requires the user's decision.

---

## 3. Environment

```
PYTHONPATH=/scratch/ga442/gab-code-env/papers/WeightedERM-reference
R_HOME=/scratch/ga442/miniforge3/envs/weighted-erm/lib/R
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
MPLBACKEND=Agg
PYTHONUNBUFFERED=1
```

Python: `/scratch/ga442/miniforge3/envs/weighted-erm/bin/python`
Working dir: `/scratch/ga442/gab-code-env/papers/WeightedERM-reference`
Git branch: `rebuttal/WER-8-m2-m3-100`
Machine: 48 cores, 251 GB RAM. Run uses `n_jobs=36`; steady-state load average ~37,
each worker ~1.6 GB RSS (~58 GB total), ~190 GB RAM still available.

---

## 4. The monitoring runner

`/tmp/run_monitored_notebook_cell.py` (pre-existing, inspected and confirmed
monitoring-only). Usage:

```
python /tmp/run_monitored_notebook_cell.py <notebook> <cell_index> <worker_log_path>
```

It makes exactly three edits to the cell source, none of which change the experiment:

1. `Parallel(n_jobs=n_jobs, verbose=0)(` → adds `return_as="generator"`.
   Results are still yielded **in submission order**, so aggregation and array
   indexing are unchanged.
2. Wraps the aggregation loop's iterable in a `tqdm(..., desc="Completed simulation
   trials", total=len(tasks))`.
3. Monkeypatches `run_single_trial` to redirect worker stdout/stderr into the
   worker log file.

Important consequence: the **"Completed simulation trials"** counter reflects
*in-order yielded* trials, so it lags actual work — a single slow trial blocks the
counter while trials behind it finish. The **"Running simulation trials"** counter is
the dispatch side and stays ~48–72 tasks ahead (joblib `pre_dispatch='2*n_jobs'`).
Neither counter stalling on its own means the job is stuck; check CPU/load instead.

---

## 5. Confirmed cell parameters (read from the notebook, not assumed)

Both cells:

- `p = 300` (**not** 200 — the user's expected filenames said p200; the actual cell
  sets `p = 300`, so output filenames contain `p300`)
- `L = 4`
- `delta_list = np.arange(1.0, 10.0, 1.0)` → 9 values (1..9)
- `num_trials = 100` → **900 tasks**
- `fixed_noise_scale = np.sqrt(0.1)` → formats as `0.316`
- `fractional_chgpt_locations = np.array([0.25, 0.60])`
- `n_jobs = 36`
- `random_seed = 46`
- Result arrays: `haus_<m>`, `size_<m>`, `runtime_<m>`, each `np.full((9, 100), np.nan)`

M1d (cell 34): `α = 1.0`; six methods
`charcoal, McScan, WERM-unp, DCDP, DPDU, MOSEG`

M1e (cell 36): `α = 0.05`; five methods (charcoal is commented out in the cell)
`McScan, WERM-unp, DCDP, DPDU, MOSEG`

Per-method exception handling inside `run_single_trial` prints
`Trial {j+1} for delta={delta:.1f}: {method_name} failed with error: {e}`
and sets that method's `runtime/haus/size` to NaN and `chgpts_<m>` to an empty array,
letting the remaining methods continue.

### Expected output paths

M1d:
```
results/M1_upsample_d_alpha10_cp025060_charcoal_McScan_WERM-unp_DCDP_DPDU_MOSEG_results_p300_L4_delta123456789_noisecale0.316_seed46.pkl
```
M1e:
```
results/M1_upsample_e_alpha005_cp025060_McScan_WERM-unp_DCDP_DPDU_MOSEG_results_p300_L4_delta123456789_noisecale0.316_seed46.pkl
```

**Caveat worth knowing:** in both cells the `pickle.dump` happens *after* several
plotting calls at the end of the cell. If a plotting call raises, the pickle is never
written. Watch the parent log for a traceback after the progress bar reaches 900/900.

---

## 6. Current state (as of 2026-08-13 04:24)

**M1d is RUNNING.** Launched 2026-08-12 23:17.

- Parent PID: **2775295**
- Launch command actually used:

```bash
cd /scratch/ga442/gab-code-env/papers/WeightedERM-reference
PYTHONPATH=/scratch/ga442/gab-code-env/papers/WeightedERM-reference \
R_HOME=/scratch/ga442/miniforge3/envs/weighted-erm/lib/R \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
nohup /scratch/ga442/miniforge3/envs/weighted-erm/bin/python \
  /tmp/run_monitored_notebook_cell.py \
  M1_upsample.ipynb 34 /tmp/m1d_v2_workers.log \
  > /tmp/m1d_v2_parent.log 2>&1 &
```

- Parent log:  `/tmp/m1d_v2_parent.log`
- Worker log:  `/tmp/m1d_v2_workers.log`
- **Ignore** the older logs from the aborted attempt: `/tmp/m1d_alpha10_*.log`,
  `/tmp/m1_upsample_cell18_*.log`.

### Progress history (yielded / 900)

| time  | completed | elapsed |
|-------|-----------|---------|
| 23:26 | 36        | 0h05m   |
| 23:41 | 90        | 0h15m   |
| 00:06 | 112       | 0h46m   |
| 00:51 | 173       | 1h30m   |
| 01:36 | 199       | 2h01m   |
| 02:35 | 204       | 3h09m   |
| 03:34 | 237       | 4h13m   |
| 04:24 | **254**   | **4h58m** |

Dispatch counter at 04:24: 324/900. Throughput falls as δ grows because n = δ·p
(n = 300 at δ=1 up to n = 2700 at δ=9), so the bulk of the wall-clock is still ahead.
**Expect many more hours.** tqdm's ETA is unreliable here; do not quote it as fact.

### Health at 04:24

- 36 loky workers alive, all ~100% CPU; load average ~37; ~190 GB RAM free.
- Tracebacks: 0. Socket errors: 0. Broken pipes: 0. MemoryError: 0. Killed: 0.
- Python-side NaN warnings: 0.

### Caught method failures so far (charcoal only)

| delta | charcoal failures |
|-------|-------------------|
| 1.0   | 100 (all trials)  |
| 2.0   | 19                |
| 3.0   | 2                 |

Underlying R errors seen (all inside charcoal):

- `Error in cpreg_stats[i] <- tmp$stats[tmp$cp] : replacement has length zero` (×200 log lines)
- `Error in ret_refined$cps[i] <- tmp$cp + before.mid :` (×8)
- `Error in colSums(A[, -n] * as.vector(Z)) :` (×4)

Three benign R warnings also appeared:
`In Gout_min * 3/4 + Gout_max/4 : longer object length is not a multiple of shorter object length`

This charcoal-failure pattern is consistent with the user's earlier aborted attempt
(charcoal failed all 100 δ=1 trials and 6 δ=2 trials by ~188 yielded trials). The user
is aware and does **not** want it worked around. Keep reporting counts; do not fix.

No other method (McScan, WERM-unp, DCDP, DPDU, MOSEG) has failed.

---

## 7. Monitoring recipe

A status script already exists at `/tmp/status_m1.sh`:

```bash
bash /tmp/status_m1.sh /tmp/m1d_v2_parent.log /tmp/m1d_v2_workers.log
```

It prints: process counts, both tqdm lines, non-tqdm parent output, worker log size,
caught failures grouped by delta+method, keyword counts
(Traceback / socket / Broken / nan / NaN / Warning / "Error in " / MemoryError / Killed),
and the top distinct R error messages.

Its contents:

```bash
#!/bin/bash
# usage: status_m1.sh <parent.log> <workers.log>
P="$1"; W="$2"
echo "=== $(date +%H:%M:%S) ==="
echo "-- procs: parent=$(pgrep -fc run_monitored_notebook_cell) loky_workers=$(pgrep -fc 'loky' ) --"
echo "-- last progress line --"
tr '\r' '\n' < "$P" | grep -E "Completed simulation trials" | tail -1
tr '\r' '\n' < "$P" | grep -E "Running simulation trials" | tail -1
echo "-- parent non-tqdm output (last 15) --"
tr '\r' '\n' < "$P" | grep -vE "simulation trials:|^\s*$|^\[A" | tail -15
echo "-- worker log size: $(wc -l < "$W") lines --"
echo "-- caught failures by method x delta --"
grep -oE "delta=[0-9.]+: [A-Za-z-]+ failed" "$W" 2>/dev/null | sort | uniq -c | sort -rn | head -20
echo "-- other error/warning keyword counts --"
for k in Traceback "socket" "Broken" "nan" "NaN" "Warning" "Error in " "MemoryError" "Killed"; do
  c=$(grep -c "$k" "$W" 2>/dev/null); echo "  $k: $c"
done
echo "-- distinct error messages (top 10) --"
grep -oE "Error in [^:]*: .*" "$W" 2>/dev/null | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -10
```

Supplementary health check:

```bash
ps -eo pid,pcpu,etimes,rss,comm --sort=-pcpu | head -12
free -g; uptime
```

Poll roughly hourly. Report to the user: completed/total + percentage, elapsed time,
warning/error counts, caught failures grouped by method and delta, and whether the
process/workers still look healthy.

---

## 8. Detecting completion

M1d is done when either:

- `Completed simulation trials: 100%|...| 900/900` appears in the parent log, then
  plotting output, then the pickle appears in `results/`; or
- the parent process exits (`pgrep -f run_monitored_notebook_cell` returns nothing).

Then check for a traceback in the parent log and confirm the pickle exists:

```bash
ls -la results/M1_upsample_d_alpha10_*p300*seed46.pkl
tr '\r' '\n' < /tmp/m1d_v2_parent.log | grep -vE "simulation trials:" | tail -40
```

---

## 9. Validation checklist for each pickle (run before starting the next experiment)

Load the pickle and verify:

1. Params: `num_trials == 100`, `delta_list` has 9 values (1..9), `p == 300`, `L == 4`,
   `fixed_noise_scale == sqrt(0.1)`, `random_seed == 46`,
   `fractional_chgpt_locations == [0.25, 0.60]`.
2. Methods present: M1d → `charcoal, McScan, WERM-unp, DCDP, DPDU, MOSEG`;
   M1e → `McScan, WERM-unp, DCDP, DPDU, MOSEG`.
3. Every `haus_<m>`, `size_<m>`, `runtime_<m>` array has shape `(9, 100)`.
4. Per-delta successful-trial counts (`results['successful_trials'][method]`) —
   compare against NaN counts per row; they should be complementary
   (successes + NaNs == 100 per delta per method).
5. Count NaNs and other nonfinite values; report per method per delta.
6. Changepoint-list lengths (`chgpts_<m>` entries) — sanity-check against `size_<m>`.
7. **Infinite Hausdorff scores must correspond exactly to legitimate zero-changepoint
   detections.** i.e. every `haus_<m>[i,j] == inf` should have `size_<m>[i,j] == 0`,
   and vice versa. Any mismatch is a real finding — report it.
8. Final scan of both logs for warnings/errors/tracebacks.

A suggested validation script skeleton (adapt key names to what the pickle actually
contains — inspect its top-level keys first):

```python
import pickle, numpy as np
path = "results/M1_upsample_d_alpha10_cp025060_charcoal_McScan_WERM-unp_DCDP_DPDU_MOSEG_results_p300_L4_delta123456789_noisecale0.316_seed46.pkl"
d = pickle.load(open(path, "rb"))
print(sorted(d.keys()))
```

---

## 10. Launching M1e (only after M1d validates)

```bash
cd /scratch/ga442/gab-code-env/papers/WeightedERM-reference
PYTHONPATH=/scratch/ga442/gab-code-env/papers/WeightedERM-reference \
R_HOME=/scratch/ga442/miniforge3/envs/weighted-erm/lib/R \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
nohup /scratch/ga442/miniforge3/envs/weighted-erm/bin/python \
  /tmp/run_monitored_notebook_cell.py \
  M1_upsample.ipynb 36 /tmp/m1e_v1_workers.log \
  > /tmp/m1e_v1_parent.log 2>&1 &
```

M1e has no charcoal, so it should be faster and should show no R errors at all.

---

## 11. If you need to restart M1d from scratch

There is **no checkpointing** — the pickle is written only at the very end, so a kill
loses all progress. To stop cleanly:

```bash
pkill -f run_monitored_notebook_cell   # then confirm loky workers died:
pgrep -fc loky
```

Then relaunch with **fresh log filenames** (e.g. `/tmp/m1d_v3_*.log`) using the
command in section 6.

---

## 12. Final deliverable to the user

For each of M1d and M1e, report:

- the exact output pickle path,
- completed/total trials and total wall-clock,
- caught method failures grouped by method and delta,
- NaN / nonfinite counts,
- the infinite-Hausdorff ↔ zero-changepoint consistency result,
- any warnings/errors/tracebacks found in the final log scan.
