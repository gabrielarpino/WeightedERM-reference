# WER-5 pre-change baseline

Baseline date: 2026-08-07
Base commit: `b4759864180a8d87dd0fd6ae346bba248eb4ceca`
Branch: `copilot/rebuttalwer-5-baseline-env`

No estimator, partition search, simulation design, or manuscript result was
changed. No full simulation or plotting workflow was run, so no comparison
plots were generated. The only notebook edits replace active 36- or 45-worker
settings with 32 in the five scoped workflows.

## Status

| Workflow | Status | Cheapest attempted check | Historical parameters found | Blocker or limit |
| --- | --- | --- | --- | --- |
| M1 | **BLOCKED** | Core WERM and KDE imports, then `import WERM.comparison` | Common blocks use seed 42, `p=200`, `L=3/4`, 15 or 30 trials, and several delta/noise grids | Stops at missing R package `charcoal`; later imports also require `changepoints`, `inferchange`, `moseg`, and custom `myglm2`. Expected `results/*.pkl` inputs are not committed. |
| M2 | **BLOCKED** | Core WERM and KDE imports, then `import WERM.comparison` | Seed 42, `p=200`, `L=4`, 30 trials, delta grids beginning at 0.5, noise scale `sqrt(0.1)` | Same R-package blocker at `charcoal`; expected plotting pickles are not committed. |
| M3 | **BLOCKED** | Core WERM import, then `import WERM.comparison` | Seed 42, `p=200`, `L=2/4`, 15 or 30 trials, delta/kappa grids, nominal noise `sqrt(0.6)` | Same R-package blocker at `charcoal`; logistic comparisons also require custom `myglm2`. Expected plotting pickles are not committed. |
| Myocardial | **SMOKE-PASSED** | Fetched UCI dataset 579 and imported `WERM.posterior` | Seed 42; fetched features `(1700, 111)` and targets `(1700, 12)` | The expensive fixed-point/posterior jobs were not run. `results/myocardial_infarction_posterior_exact_one_chgpt.pkl` is absent, so plotting cells depending on it remain blocked. |
| FRED-MD | **BLOCKED** | Loaded committed pickle and RData through rpy2 | Notebook sets `L=5`; pickle has `X (773, 114)`, `y (773,)`; the data-preparation R script uses seed 12345 only for CI sections | Data load passes, then the workflow stops at missing R package `inferchange`. The notebook retains a machine path `/scratch/ga442/Weighted-ERM/FRED-MD data/fredmd_data.pkl` with a local fallback. The “3-chgpt” filenames versus `L=5` are not reinterpreted here. |

`SMOKE-PASSED` means only the stated import/data boundary passed. It is not a
reproduction. None of the five workflows is classified `REPRODUCED`.

## Environment

`.python-version` selects Python 3.10 and `uv.lock` locks the core and
`notebooks` extra. This machine supplied CPython 3.10.20. The historical
`setup.py` requires exactly 3.10.10, but uv 0.11.8 could not locate or
provision that patch release. The uv metadata therefore uses the narrow
compatible range `>=3.10,<3.11`; this is an environment compatibility choice,
not a claim that 3.10.20 reproduces the original results.

`[tool.uv] package = false` is intentional. This remains a notebook-centered
research repository: uv creates the environment from `pyproject.toml`, and
Python imports `WERM` from the checkout without converting the project or
overriding the existing `setup.py` metadata.

Exact setup and core verification:

```bash
uv sync --frozen
uv run --frozen python -c "import WERM; import WERM.estimators"

# Required only for notebook dependencies. ABI mode avoids a Python.h build
# failure observed on this machine.
RPY2_CFFI_MODE=ABI uv sync --frozen --extra notebooks
```

The `notebooks` extra includes the Python dependencies directly imported by
the scoped notebooks, including `kde-diffusion`, `KDEpy`, `rpy2`,
`pandas`, `seaborn`, and `ucimlrepo`. uv cannot supply the historical
custom R packages.

## Resource bounds

Before any future notebook execution, run one workflow at a time in the
foreground and set:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Active worker values above the WER-5 limit were changed mechanically to 32:
three M1 assignments, four M2 assignments, four M3 assignments, and two
myocardial `Parallel(...)` calls. No run in this baseline used those workers;
all checks were single foreground import/data-load commands.

## Missing inputs and ambiguities

- R 4.6.1 is present, but `charcoal`, `changepoints`, `inferchange`,
  `moseg`, `myglm2`, `fbi`, `readr`, `pracma`, `devtools`, and
  `fnets` are absent. Several are GitHub/custom packages with no version or
  commit recorded in the repository, so they were not installed or guessed.
- Default rpy2 compilation failed because `Python.h` is absent. The locked
  package installed successfully in ABI mode; both attempts are logged.
- The FRED-MD CSV, RData, and pickle are committed and readable. Its historical
  absolute pickle path is machine-specific; the existing local fallback was
  used and the notebook was not rewritten.
- No `.pkl` experiment result is committed under `results/`. M1/M2/M3
  plotting cells name many such files, and myocardial plotting names the
  missing posterior pickle above. Existing PDF/PGF/NPZ artifacts do not prove
  that the current checkout reproduces their generating runs.
- M1/M2/M3 contain multiple experiment blocks and parameter grids rather than
  one canonical command. No block was selected or reduced on the author's
  behalf.

## Commands and logs

All attempted execution checks used the thread environment above. Key commands:

```bash
uv lock --python 3.10
uv sync --frozen
RPY2_CFFI_MODE=ABI uv sync --frozen --extra notebooks
uv run --frozen --no-sync python -c "import WERM; import WERM.estimators"
uv run --frozen --extra notebooks --no-sync python -c "import WERM.comparison"
uv run --frozen --extra notebooks --no-sync python -c "from KDEpy import FFTKDE; from kde_diffusion import kde1d"
```

Readable logs are in `rebuttal/phase1/logs/`:

- `00_uv_sync.log`: failed editable-build attempt caused by conflicting legacy
  and PEP 621 metadata.
- `01_uv_sync_retry.log`: successful frozen core sync in uv environment-only
  mode.
- `02_uv_sync_notebooks.log`: default rpy2 build failure (`Python.h`
  missing).
- `03_uv_sync_notebooks_abi.log` and `06_uv_sync_kde_dependency.log`:
  successful notebook-extra syncs in ABI mode.
- `04_uv_exact_python_check.log`: unavailable exact Python 3.10.10 check.
- `05_r_package_inventory.log`: R-package availability.
- `10_core_import.log` and `11_kde_import.log`: passing core/KDE imports.
- `20_m1_import.log`, `21_m2_import.log`, `22_m3_import.log`: comparison
  import blockers.
- `23_myocardial_smoke.log`: passing UCI fetch/import smoke check.
- `24_fred_md_smoke.log`: preserved first check-harness error.
- `25_fred_md_smoke_retry.log`: corrected FRED-MD load and
  `inferchange` blocker.
- `30_notebook_integrity_and_worker_cap.log`: valid notebook JSON and no
  active scoped worker value above 32.
- `31_thread_cap.log`: thread-limit environment evidence.
- `32_final_core_sync.log`: final frozen core-only sync.
- `33_readme_core_command.log`: passing exact core import command documented
  in the README.
