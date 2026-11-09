"""Create the WERM/AMP comparison notebook without launching experiments."""
from pathlib import Path
import copy
import sys

import nbformat as nbf


def build(destination=None, spacing_roots=None, t20_roots=None):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination) if destination else root / "WERM_AMP_comparison.ipynb"
    previous = nbf.read(destination, as_version=4) if destination.exists() else None
    if spacing_roots is None:
        spacing_roots = (previous.metadata.get('werm_amp_spacing_roots', {})
                         if previous is not None else {})
    spacing_roots = {str(key): str(value) for key, value in spacing_roots.items()}
    if not set(spacing_roots).issubset({'20', '10'}):
        raise ValueError('Spacing roots must be keyed by denominator 20 or 10.')
    if t20_roots is None:
        t20_roots = (previous.metadata.get('werm_amp_t20_roots', {})
                     if previous is not None else {})
    t20_roots = {str(key): str(value) for key, value in t20_roots.items()}
    if not set(t20_roots).issubset({'20', '10'}):
        raise ValueError('T=20 roots must be keyed by denominator 20 or 10.')
    # Notebook JSON sorts metadata keys; retain the original cell's study order
    # so rebuilding does not discard outputs for a reordered dictionary literal.
    t20_roots = {key: t20_roots[key] for key in ('20', '10') if key in t20_roots}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(r"""# WERM and AMP: change points with Gaussian regression signals

> **WERM+P correction:** Earlier WERM+P curves are retained as archival results
> and are superseded. The compact unknown-count and relocated sections at the
> end use the corrected single upper-bound fit and half-squared search loss
> with $\gamma=6$.

Compare **WERM+CV, WERM+P and AMP** in the linear model as the sampling ratio
$\delta=n/p$ increases, under three covariate covariances. All methods receive
the same $(X,y)$ within each trial. **The covariates are not whitened.**

The original $\Delta_n=n/40$ comparison is retained as an **archival reference**:
its unknown-count AMP results used superseded prefix-label handling. Two separate
sections below repeat both count settings with **$\Delta_n=n/20$** and
**$\Delta_n=n/10$**. These two repeats share the same data and random seeds;
their unknown-count truth is $(0.25,0.50)n$, whereas the original reference
uses $(0.20,0.50)n$. The repeats use the corrected AMP label support described
below. Original unknown-count results should not be treated as results of that
corrected implementation.

A final section repeats the two corrected larger-gap studies with **20 AMP
iterations**, using the same datasets, initializations and Monte Carlo seeds.
Their WERM fits are reused unchanged, so the new comparison isolates the AMP
iteration budget. The earlier sections retain their $T=10$ outputs.

The main comparison uses an unknown change-point count. A supplementary
fixed-count comparison isolates localization. The working dimension is
$p=200$. The full design targets **100 paired trials per ratio, covariance and comparison**:

| $\delta$ | 1.5 | 3.5 | 7.5 | 11.5 |
|---|---:|---:|---:|---:|
| $n$ | 300 | 700 | 1500 | 2300 |

| Comparison | True fractional locations | Working segment count |
|---|---|---|
| Original $n/40$ unknown count | $(0.20,0.50)$ | $K\in\{1,2,3,4\}$ |
| New $n/20$ and $n/10$ unknown count | $(0.25,0.50)$ | $K\in\{1,2,3,4\}$ |
| Known count, all separations | $(0.25,0.45,0.70)$ | exactly $K=4$ |

The question is whether the ordinary Gaussian AMP procedure performs differently
under isotropic and correlated designs relative to WERM. No ranking is imposed
on the results; every scheduled trial and numerical failure is retained.
This notebook creates local outputs only and does not modify the manuscript."""),
        md(r"""## Data and covariance conventions

For each regime $r$, draw the coefficient entries independently as
$\beta_j^{(r)}\sim N(0,\delta)$. Independently draw
$$x_i\sim N(0,\Sigma/n),\qquad
\varepsilon_i\sim N(0,0.1),\qquad
y_i=x_i^\top\beta^{(r_i)}+\varepsilon_i.$$
There is no intercept or coefficient sparsity. The independent Gaussian
coefficients replace M2's sparse signal differences; its dimension, noise
variance, ratio sweep, count-selection conventions and plotting style are
retained. The coefficient scaling keeps the expected signal variance equal to
one because all three covariance matrices have trace $p$.

| Covariance | Definition and source |
|---|---|
| Identity | $\Sigma=I_p$. |
| AR(1) | $\Sigma_{jk}=0.4^{|j-k|}$, matching `M1_upsample_final.ipynb` and the robust linear-model experiment in the paper. The dense Gaussian/McScan appendix experiment also uses this covariance. |
| M3 random correlation | Generate $Q$ by QR decomposition of a standard Gaussian $p\times p$ matrix and independent $d_j\sim\chi^2_{10}$. Set $S=Q^\top\operatorname{diag}(d)Q$ and $\Sigma_{jk}=S_{jk}/\sqrt{S_{jj}S_{kk}}$. This is exactly `sample_corr(p, df=10, seed=42)` from `M3_upsample_final.ipynb`. |

The M3 covariance draw is fixed across trials. Form the design as
$X=Z\operatorname{chol}(\Sigma)^\top/\sqrt n$, with independent standard Gaussian
entries in $Z$. Every method is given this same unwhitened design and unstandardized
response. AMP is given the matched Gaussian coefficient prior and noise variance;
its ordinary denoiser is not replaced by a covariance-aware or whitened method.
**AMP is not supplied $\Sigma$, its eigenvalues, or a covariance estimate.**
Its Gaussian coefficient prior is $\delta I_L$ in every covariance setting, and
its state updates retain the isotropic-design calibration. Across covariance
settings, only the data-generating mechanism changes; AMP receives the resulting
raw $X,y$ and the same prior, noise level, location constraints and iteration count.
True locations and realized coefficient vectors are used only to generate data
and evaluate performance, not to fit the estimators."""),
        md(r"""## Estimators and admissible partitions: archival $n/40$ reference

This section records the original run. Its unknown-count AMP adapter restricted
the working labels to $(0,\ldots,K-1)$ for each $K$. That restriction differed
from the supplied AMP method and is corrected in the two new studies below.
The original figures and their saved source specification are retained for
traceability; the current implementation is not used to rerun this archive.

At fitting sample size $m$, use the M2 separation
$$\Delta_m=\max\{\lceil m/50\rceil,\lfloor m/40\rfloor,1\}.$$
Consecutive internal boundaries must be at least $\Delta_m$ apart; the two end
segments must be nonempty. Boundary indices are zero-based slice boundaries:
a boundary $c$ starts the next segment at observation `c`.

The location prior is uniform over the admissible partitions conditional on
$K$. **Its finite-sample marginal weights are computed from exact combinatorial
probabilities**, and the same partition convention is used throughout. This
replaces the canonical notebooks' interpolation from a 50-point reference grid;
it avoids giving WERM and AMP different approximations to their location prior.
The original AMP configuration generator also imposes the minimum gap on the
end segments. The local adapter instead uses the shared M2 support above, so
the methods search the same admissible boundaries.

- **WERM+CV:** for each $K=1,\ldots,4$, perform the original exhaustive,
  interlaced 10-fold procedure. Fit fixed-count WERM with unpenalized squared
  loss on each training fold, map its boundaries to the original indices,
  and refit ordinary least squares within the resulting segments using only
  training observations. Sum **absolute held-out prediction errors** across
  the folds. Select the smallest-scoring $K$ and fit fixed-count WERM again
  on all observations. Squared loss is used for every regression fit; absolute
  loss is used only for held-out scoring. Recompute the gap and prior weights
  at each training sample size.
- **WERM+P:** fit the same full-data fixed-count WERM candidates and select
  $$\widehat K=\arg\min_{K\in\{1,2,3,4\}}
  \left\{\sum_i\big(y_i-\widehat\theta_{i,\widehat r_i(K)}^{(K)}\big)^2
       +6\log\binom{n-1}{K-1}\right\}.$$
  This is the **squared**, not half-squared, search objective and the frozen
  $\gamma=6$ convention of the latest M2 final comparison. There is no coefficient
  penalty. The weights are conditional on $K$; this is the fixed-count candidate
  version of WERM+P. If WERM+CV and WERM+P choose the same $K$, their final
  locations coincide.
- **AMP:** use the linear-model implementation in `../AMP_chgpt_reg` with its
  usual Gaussian denoiser, the matched independent Gaussian coefficient prior,
  known noise standard deviation $\sqrt{0.1}$, and **10 iterations**. Use the
  uniform prior on $K=1,\ldots,4$ and the same conditional location prior.
  Select the joint MAP count and boundaries with an exact dynamic program.
  Output-channel state expectations use **16,384 Monte Carlo draws per
  iteration**, with the discrete regime labels averaged exactly. Gaussian
  input-channel state updates use their closed-form expressions. AMP is supplied
  the population Gaussian coefficient prior and noise variance; WERM does not
  require these inputs. No true count is supplied in the unknown-count comparison.

The local AMP adapter corrects two implementation details in the upstream
simulation path: simulated noise uses the supplied noise standard deviation,
and posterior evaluation uses the state-evolution parameters aligned with the
final AMP iterate. It does not whiten the design or change the Gaussian denoiser.

For the known-count comparison, both methods use exactly four segments. WERM
uses its fixed-count fit and AMP conditions its posterior on $K=4$; neither
cross-validation nor a count penalty is needed. Some short least-squares refits
are underdetermined at these ratios; they use the numerical least-squares solver
used by M2. This is an empirical comparison, without an SE overlay or a claim
that all segment refits satisfy an overdetermined-theory condition."""),
        code("""# Load the pinned archival run; do not prepare a new run with current sources.
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import json
from pathlib import Path
from IPython.display import Image, display
from tools import werm_amp_experiment as experiment

RESULTS_DIR = os.environ.get('WERM_AMP_RESULTS_DIR')
ROOT = (Path(RESULTS_DIR).resolve() if RESULTS_DIR else
        experiment.ROOT / 'results/WERM_AMP_comparison/8490596ee1d3')
if RESULTS_DIR:
    print('OVERRIDE: displaying the saved run at WERM_AMP_RESULTS_DIR.')
    print('A pilot or partial run is not the full 100-trial design described above.')
print('Results and checkpoints:', ROOT)
print(json.dumps(experiment.status(ROOT), indent=2, default=str))
"""),
        md("""## Loading the archival reference

The next cell remains disabled. The original run is pinned to its saved result
directory, and its sources differ from the corrected implementation used below.
Its source-hash guard will reject an attempt to append current-implementation
results to that archive. Use the separate larger-gap controls below for the
new studies. `WERM_AMP_JOBS` controls workers when those studies are launched.
Refreshing this notebook does not start experiments by default.
The saved specification and trial checkpoints determine which results are loaded;
results from different designs are not pooled. `WERM_AMP_RESULTS_DIR` can point to
a separate pilot directory; its actual saved settings are displayed below.
Before starting or resuming a run, the next cell checks that the local experiment
and upstream AMP source files match the hashes stored with that run."""),
        code("""RUN_EXPERIMENT = False
JOBS = int(os.environ.get('WERM_AMP_JOBS', '4'))
if RUN_EXPERIMENT:
    import hashlib
    saved_spec = json.loads((ROOT / 'specification.json').read_text())
    repository = Path(experiment.__file__).resolve().parents[1]
    source_mismatches = []
    for base, key in ((repository, 'sources_sha256'),
                      (repository.parent / 'AMP_chgpt_reg', 'amp_sources_sha256')):
        expected_sources = saved_spec.get(key)
        if not isinstance(expected_sources, dict) or not expected_sources:
            raise RuntimeError(f'Saved run has no valid {key}; refusing to resume.')
        for relative_path, expected_hash in expected_sources.items():
            path = base / relative_path
            actual_hash = (hashlib.sha256(path.read_bytes()).hexdigest()
                           if path.is_file() else None)
            if actual_hash != expected_hash:
                source_mismatches.append(str(path))
    if source_mismatches:
        raise RuntimeError('Source files differ from the saved run; refusing to '
                           'mix results. Prepare a new run or restore the recorded '
                           'sources. Mismatches: ' + ', '.join(source_mismatches))
    experiment.run(ROOT, jobs=JOBS)
else:
    print('No trials launched. Set RUN_EXPERIMENT=True above to run or resume.')
print(json.dumps(experiment.status(ROOT), indent=2, default=str))
"""),
        md(r"""## Current snapshot and diagnostics

The following cells display all currently completed checkpoints, including a
partial run. Sample sizes and successful, empty and failed fits are reported
explicitly; incomplete trial counts are not presented as a completed experiment.

The location metric is Hausdorff distance divided by $n$. It is undefined for
an empty estimated set, so the location curve is conditional on nonempty,
successful fits. Valid empty estimates contribute zero to the count curve.
Numerical failures are distinct from empty estimates and are reported in the
diagnostics. These rates should be read together with the location and count
curves. Histograms contain all estimated boundaries from successful fits and
therefore give more boundary observations to trials selecting more changes.

Line plots follow the final M2 conventions: means and first/third-quartile
variation, not confidence intervals. Histograms mark true locations. The
plotting code saves PNG and PDF versions under this run's figure directory."""),
        code("""CURRENT_STATUS = experiment.status(ROOT)
print(json.dumps(CURRENT_STATUS, indent=2, default=str))
PAYLOAD = experiment.load_results(ROOT)
SUMMARY = experiment.summarize(PAYLOAD)
print('Saved specification for this snapshot:')
display(PAYLOAD['specification'])
print('Completed paired trial records:', len(PAYLOAD['records']))
display(SUMMARY.round(5))
if not SUMMARY.empty:
    SUMMARY.to_csv(ROOT / 'notebook_summary.csv', index=False)
"""),
        md(r"""## Original $n/40$ reference: unknown number of change points

True changes are at $(0.20,0.50)n$. Each covariance has its own Hausdorff,
selected-count and location-histogram panels for **WERM+CV, WERM+P and AMP**.
The count reference is two; each procedure may select zero to three changes.
The complete ratio range and every available trial are retained."""),
        code("""if PAYLOAD['records']:
    UNKNOWN_FIGURES = experiment.plot_study(
        PAYLOAD, study='unknown', output_dir=ROOT / 'figures' / 'unknown')
    for path in UNKNOWN_FIGURES:
        display(Image(filename=str(path)))
else:
    print('No completed trial snapshots yet. Refresh this cell after the run starts.')
"""),
        md(r"""## Original $n/40$ reference: known number of change points

True changes are at $(0.25,0.45,0.70)n$, with exactly four working segments.
Compare fixed-count **WERM** and **AMP** to isolate their location estimates.
The count panel checks that successful fits return exactly three boundaries;
it is not a count-selection result. The Hausdorff curves and histograms use
the same conventions as above."""),
        code("""if PAYLOAD['records']:
    KNOWN_FIGURES = experiment.plot_study(
        PAYLOAD, study='known', output_dir=ROOT / 'figures' / 'known')
    for path in KNOWN_FIGURES:
        display(Image(filename=str(path)))
else:
    print('No completed trial snapshots yet.')
"""),
        md(r"""## Original $n/40$ reference: AMP Monte Carlo resolution check

A separate numerical check uses **12 prechosen trial-0 datasets**: both
comparisons, all three covariances, and $\delta\in\{3.5,11.5\}$. It compares
the main run's 16,384 Monte Carlo draws per iteration with 65,536 draws, keeping
the data, initialization seed, Gaussian denoiser, iteration count and partition
prior fixed. The main experiment remains the 16,384-draw run; higher-resolution
outputs are not substituted into its curves.

In this check, all selected counts agree and nine of the twelve estimated
boundary sets agree exactly. Two unknown-count datasets show substantial
location changes, and one known-count dataset changes more modestly. This
small check therefore does **not** establish numerical stability of the AMP
location estimates. The table below retains both resolutions, including their
statuses and normalized Hausdorff errors. This cell only reads the saved
diagnostic; it does not rerun AMP."""),
        code("""RESOLUTION_FILE = ROOT / 'diagnostics' / 'amp_mc_resolution.csv'
if RESOLUTION_FILE.exists():
    import pandas as pd
    RESOLUTION = pd.read_csv(RESOLUTION_FILE)
    indexed = RESOLUTION.set_index(['study', 'covariance', 'delta', 'trial'])
    resolution_tables = {}
    for draws in (16384, 65536):
        names = [f'cp_{draws}', f'haus_{draws}',
                 f'count_{draws}', f'status_{draws}']
        resolution_tables[f'{draws:,} draws'] = indexed[names].rename(columns={
            f'cp_{draws}': 'boundaries', f'haus_{draws}': 'Hausdorff / n',
            f'count_{draws}': 'count', f'status_{draws}': 'status'})
    display(pd.concat(resolution_tables, axis=1).round(5))
    same_counts = RESOLUTION['count_16384'].eq(RESOLUTION['count_65536'])
    same_locations = RESOLUTION['cp_16384'].eq(RESOLUTION['cp_65536'])
    print(f'Identical counts: {same_counts.sum()}/{len(RESOLUTION)}; '
          f'identical boundary sets: {same_locations.sum()}/{len(RESOLUTION)}.')
    print('Largest absolute Hausdorff changes:',
          RESOLUTION['haus_difference'].abs().nlargest(3).round(5).tolist())
else:
    print('No saved resolution check for this run; no additional fits launched.')
"""),
        md(r"""## Reading the archival $n/40$ comparison

These observations describe the original saved outputs, including its
superseded unknown-count AMP label restriction. They are not conclusions about
the corrected AMP implementation used in the new studies below.

The known-count experiment shows a clear covariance contrast at the larger
sampling ratios: AMP localizes more accurately under the identity covariance,
whereas WERM localizes more accurately under the AR(1) and M3 covariances.
The unknown-count comparison is mixed. AMP is not uniformly better in the
isotropic case, and both WERM procedures overestimate the true two changes
by selecting three at the larger ratios. Their small Hausdorff errors should
therefore be read together with the count results. The Monte Carlo resolution
check also shows that individual AMP locations can depend on numerical
resolution; it does not justify a general stability claim.

The table below is computed from the loaded results at the largest sampling
ratio. `mean_haus` conditions on a successful, nonempty estimate. `mean_count`
includes valid empty estimates as zero and excludes failed fits.
`correct_count_rate` and `empty_rate` use all completed trials as their
denominator; failed fits are not correct-count estimates or valid empty
estimates. The success and failure columns make these denominators explicit.
The known-count correct-count rate checks the required output size, rather
than evaluating count selection."""),
        code("""if not SUMMARY.empty:
    largest_delta = max(PAYLOAD['specification']['deltas'])
    endpoint_columns = ['study', 'covariance', 'method', 'trials', 'successes',
        'failure_rate', 'mean_haus', 'mean_count', 'correct_count_rate', 'empty_rate']
    ENDPOINT_SUMMARY = SUMMARY.loc[
        SUMMARY['delta'].eq(largest_delta), endpoint_columns].copy()
    print(f'Summary at delta={largest_delta:g}.')
    if not CURRENT_STATUS['complete']:
        print('PARTIAL SNAPSHOT: the interpretation above concerns the full design; '
              'inspect the completed-trial counts shown here.')
    display(ENDPOINT_SUMMARY.round(5))
else:
    print('No completed trials to summarize.')
"""),
        md("""## Reproducibility and interpretation

`tools/werm_amp_experiment.py` defines the data generator, covariance matrices,
shared exact prior, estimators, resumable execution and plotting. Its saved
specification records the settings used for this run. The local AMP adapter
imports the implementation from `../AMP_chgpt_reg`; that repository must be
available beside `WeightedERM-reference`.

Compare all three covariance settings, together with empty/failure frequencies
and available trial counts. Finite-sample changes in AMP performance under
correlation are observations from this experiment, not a predetermined outcome
or a theorem. The mild AR(1) and M3 covariance choices are matched to existing
paper experiments and are not selected after observing the methods' ranking."""),
    ]
    reproducibility = cells.pop()
    cells.extend([
        md(r"""## Correcting AMP's unknown-count label support

The original local adapter restricted a $K$-segment AMP path to working labels
$(0,\ldots,K-1)$. This concentrated the no-change prior on label zero and did
not match the configuration support in `AMP_chgpt_reg`. The two new studies use
that supplied method's label convention:

- For $K=1$ (no change), any one of the $L$ working labels may remain constant.
- For $K\ge2$, the first segment has label zero and subsequent labels form an
  increasing subset of $\{1,\ldots,L-1\}$.

The prior is uniform on $K\in\{1,\ldots,L\}$, then uniform on the permitted
label sequences and admissible locations conditional on $K$. Thus the number
of label choices is $a_1=L$ and $a_K=\binom{L-1}{K-1}$ for $K\ge2$. Both the
AMP marginal probabilities and its MAP search use this same support; the latter
includes the corresponding $\log a_K$ prior term. The location spacing remains
the shared M2 convention specified below. In the known-count case $K=L=4$,
the label sequence is uniquely $(0,1,2,3)$.

This corrects the prior construction and the admissible label paths in the
posterior search. The Gaussian denoiser, state recursion, **10 iterations**,
and absence of any covariance input or whitening remain unchanged. The saved
validation tables below, when present, are read without launching extra fits."""),
        code("""LABEL_VALIDATION_DIR = (experiment.ROOT / 'results' /
                        'WERM_AMP_comparison' / 'amp_label_support_validation')
validation_files = sorted(LABEL_VALIDATION_DIR.glob('*.csv'))
if validation_files:
    import pandas as pd
    for validation_file in validation_files:
        print(validation_file.name)
        display(pd.read_csv(validation_file).round(6))
else:
    print('No saved label-support validation tables at', LABEL_VALIDATION_DIR)
"""),
        md(r"""## Repeats with larger minimum separation

The following two studies use $\Delta_n=n/20$ and $\Delta_n=n/10$, with
**unknown-count true locations $(0.25,0.50)n$ in both**. Each repeats both the
unknown-count comparison and the known four-segment comparison, with all three
covariance matrices, all four sampling ratios, and 100 paired trials per
setting. The known-count locations remain $(0.25,0.45,0.70)n$.

The two repeats share identical datasets and AMP initialization/Monte Carlo
seeds; only the gap and its associated location prior differ between them.
Relative to the original $n/40$ reference, the unknown-count study changes
the separation, the first true location (from $0.20n$ to $0.25n$), and the
AMP label-support handling corrected above. Its comparison with that reference
therefore does not isolate separation alone. Coefficient vectors, latent design
draws, noise draws and AMP seeds
retain their original seeding convention.

For denominator $q\in\{20,10\}$ and a training sample of size $m$, the integer
gap is $\Delta_m=\max\{\lceil m/50\rceil,\lfloor m/q\rfloor,1\}$.
Thus the full-data gaps are exactly $n/q$ for the sample sizes used here;
fold-local gaps follow the same rounding convention as the original CV code.
The exact prior marginals are recomputed for each gap and training sample size.
The end segments remain nonempty, as in the reference experiment.

AMP still uses **10 iterations**, 16,384 Monte Carlo draws per iteration,
the same Gaussian denoiser, and no covariance input or whitening. WERM+P keeps
$\gamma=6$, and WERM+CV keeps the same 10-fold procedure. The figures below
use their own saved results; the original $n/40$ conclusions and numerical
resolution check are not assumed to apply to these larger-gap studies."""),
        code("""# Load the two larger-gap runs; no fits are launched by default.
SPACING_ROOT_HINTS = __ROOTS__
SPACING_ROOTS, SPACING_PAYLOADS, SPACING_SUMMARIES = {}, {}, {}
RUN_SPACING_EXPERIMENTS = False

for denominator in (20, 10):
    hint = os.environ.get(f'WERM_AMP_RESULTS_DIR_GAP{denominator}')
    hint = hint or SPACING_ROOT_HINTS.get(str(denominator))
    if hint:
        run_root = Path(hint)
        if not run_root.is_absolute():
            run_root = experiment.ROOT / run_root
    else:
        run_root = experiment.prepare({'separation_fraction': 1 / denominator,
                                       'unknown_truth': [.25, .50]})
    saved = json.loads((run_root / 'specification.json').read_text())
    if saved['separation_fraction'] != 1 / denominator:
        raise ValueError(f'Incorrect separation in {run_root}')
    if saved['unknown_truth'] != [.25, .50]:
        raise ValueError(f'Incorrect unknown-count truth in {run_root}')
    if 'uniform source label subsets:' not in saved.get('amp_configuration_prior', ''):
        raise ValueError(f'Run predates the AMP label-support correction: {run_root}')
    if RUN_SPACING_EXPERIMENTS:
        experiment.run(run_root, jobs=JOBS)  # verifies source hashes before fitting
    SPACING_ROOTS[denominator] = run_root
    SPACING_PAYLOADS[denominator] = experiment.load_results(run_root)
    SPACING_SUMMARIES[denominator] = experiment.summarize(SPACING_PAYLOADS[denominator])
    print(f'Delta=n/{denominator}: {run_root}')
    print(json.dumps(experiment.status(run_root), indent=2))
if not RUN_SPACING_EXPERIMENTS:
    print('Larger-gap sections only load snapshots; no fits launched.')
""".replace('__ROOTS__', repr(spacing_roots))),
    ])
    for denominator in (20, 10):
        cells.extend([
            md(r"""## Repeat with $\Delta_n=n/__DIVISOR__$

This section displays the separate $n/__DIVISOR__$ run. Means and quartiles,
conditional Hausdorff errors, empty estimates and failed fits retain the
definitions above. The summary includes every completed trial and identifies
partial snapshots explicitly.""".replace('__DIVISOR__', str(denominator))),
            code("""spacing_root = SPACING_ROOTS[__DIVISOR__]
spacing_payload = SPACING_PAYLOADS[__DIVISOR__]
spacing_summary = SPACING_SUMMARIES[__DIVISOR__]
spacing_status = experiment.status(spacing_root)
print('Delta=n/__DIVISOR__:', json.dumps(spacing_status, indent=2))
print('Saved settings:')
settings_to_show = ['p', 'trials', 'deltas', 'covariances', 'unknown_truth',
    'known_truth', 'max_segments', 'separation_fraction', 'minimum_gap_fraction',
    'endpoint_minimum', 'folds', 'gamma', 'noise_std', 'amp_iterations',
    'amp_mc_samples', 'amp_configuration_prior', 'whitening', 'response_standardization']
display({key: spacing_payload['specification'][key] for key in settings_to_show})
display(spacing_summary.round(5))
if not spacing_status['complete']:
    print('PARTIAL SNAPSHOT: inspect the trial counts before interpreting the curves.')
if not spacing_summary.empty:
    spacing_summary.to_csv(spacing_root / 'notebook_summary.csv', index=False)
""".replace('__DIVISOR__', str(denominator))),
            md(r"""### $\Delta_n=n/__DIVISOR__$: unknown number of change points

True locations are $(0.25,0.50)n$, and the search permits zero to three
changes. Compare **WERM+CV, WERM+P and AMP** on the same datasets. Each covariance
is shown with its Hausdorff/count curves and the four sampling-ratio histogram
rows; the count reference is two.""".replace('__DIVISOR__', str(denominator))),
            code("""payload = SPACING_PAYLOADS[__DIVISOR__]
if payload['records']:
    for path in experiment.plot_study(payload, study='unknown',
            output_dir=SPACING_ROOTS[__DIVISOR__] / 'figures' / 'unknown'):
        print('Delta=n/__DIVISOR__ — unknown count:', path.name)
        display(Image(filename=str(path)))
else:
    print('No completed n/__DIVISOR__ trials yet; refresh after checkpoints are available.')
""".replace('__DIVISOR__', str(denominator))),
            md(r"""### $\Delta_n=n/__DIVISOR__$: known number of change points

True locations remain $(0.25,0.45,0.70)n$. Both **WERM** and **AMP** must return
exactly three boundaries ($L=4$). These curves assess localization without CV
or count selection; the count panel verifies the required output size.""".replace('__DIVISOR__', str(denominator))),
            code("""payload = SPACING_PAYLOADS[__DIVISOR__]
if payload['records']:
    for path in experiment.plot_study(payload, study='known',
            output_dir=SPACING_ROOTS[__DIVISOR__] / 'figures' / 'known'):
        print('Delta=n/__DIVISOR__ — known count:', path.name)
        display(Image(filename=str(path)))
else:
    print('No completed n/__DIVISOR__ trials yet.')
""".replace('__DIVISOR__', str(denominator))),
        ])
    cells.extend([
        md(r"""## Larger-gap comparison at the largest sampling ratio

The table below is generated from the two loaded repeat studies. It retains
both localization and count diagnostics, rather than selecting a preferred
separation or covariance after observing the results. The original $n/40$
result remains available above. `mean_haus` conditions on successful nonempty
fits; empty and failure rates must be read with it."""),
        code("""import pandas as pd
larger_gap_tables = []
for denominator, summary in SPACING_SUMMARIES.items():
    if not summary.empty:
        largest = max(SPACING_PAYLOADS[denominator]['specification']['deltas'])
        endpoint = summary.loc[summary['delta'].eq(largest),
            ['study', 'covariance', 'method', 'trials', 'successes', 'failure_rate',
             'mean_haus', 'mean_count', 'correct_count_rate', 'empty_rate']].copy()
        endpoint.insert(0, 'separation', f'n/{denominator}')
        larger_gap_tables.append(endpoint)
if larger_gap_tables:
    display(pd.concat(larger_gap_tables, ignore_index=True).round(5))
else:
    print('No larger-gap trials to summarize yet.')
"""),
    ])
    cells.extend([
        md(r"""## AMP with $T=20$ iterations

Repeat both corrected larger-gap studies with **20 AMP iterations**. The
Gaussian denoiser, label prior/search, 16,384 Monte Carlo draws per iteration,
and covariance-unaware state calibration are unchanged. AMP is still supplied
neither $\Sigma$ nor an estimate of it, and no whitening is performed.

Each $T=20$ fit uses the same dataset, initialization seed and Monte Carlo
stream as its $T=10$ counterpart. The first ten iterations therefore follow the
same trajectory; the new estimate is taken after the prescribed twentieth
iteration, without selecting an intermediate iterate. **WERM results are
copied from the matching $T=10$ trial**, since its estimators do not depend on
the AMP iteration budget.

Both gaps, all three covariance matrices, all four sampling ratios, and both
count scenarios are retained, with 100 paired trials per setting. The unknown
truth remains $(0.25,0.50)n$ and the known-count truth remains
$(0.25,0.45,0.70)n$. The following cells load saved results only; they do not
launch new fits. Empty estimates and failed fits retain the explicit treatment
used above. The figures use all available trials and do not assume that more
iterations improve performance."""),
        code("""# Read-only by default; set RUN_AMP20=True to run or resume the dedicated driver.
from tools import werm_amp_iterations as iteration_experiment

RUN_AMP20 = False
T20_ROOT_HINTS = __T20_ROOTS__
T20_ROOTS, T20_PAYLOADS, T20_SUMMARIES = {}, {}, {}
prepared_iteration_roots = None
for denominator in (20, 10):
    hint = os.environ.get(f'WERM_AMP_T20_RESULTS_DIR_GAP{denominator}')
    hint = hint or T20_ROOT_HINTS.get(str(denominator))
    if hint:
        run_root = Path(hint)
        if not run_root.is_absolute():
            run_root = experiment.ROOT / run_root
    else:
        if prepared_iteration_roots is None:
            prepared_iteration_roots = iteration_experiment.prepare_iterations()
        run_root = prepared_iteration_roots[denominator]
    saved = json.loads((run_root / 'specification.json').read_text())
    baseline = SPACING_PAYLOADS[denominator]['specification']
    if baseline['amp_iterations'] != 10 or saved['amp_iterations'] != 20:
        raise ValueError(f'Expected paired T=10/T=20 runs for gap n/{denominator}')
    compared_t10 = {key: value for key, value in baseline.items() if key != 'amp_iterations'}
    compared_t20 = {key: value for key, value in saved.items() if key != 'amp_iterations'}
    if compared_t10 != compared_t20:
        raise ValueError('T=20 settings or algorithm sources differ beyond iteration count.')
    T20_ROOTS[denominator] = run_root
if RUN_AMP20:
    iteration_experiment.run_iterations(T20_ROOTS, jobs=JOBS)
for denominator, run_root in T20_ROOTS.items():
    T20_PAYLOADS[denominator] = experiment.load_results(run_root)
    T20_SUMMARIES[denominator] = experiment.summarize(T20_PAYLOADS[denominator])
    print(f'T=20, Delta=n/{denominator}: {run_root}')
    print(json.dumps(experiment.status(run_root), indent=2))
if not RUN_AMP20:
    print('No additional fits launched by these notebook cells.')
""".replace('__T20_ROOTS__', repr(t20_roots))),
    ])
    for denominator in (20, 10):
        cells.extend([
            md(r"""### $T=20$, $\Delta_n=n/__DIVISOR__$

The summary below records the actual trial counts and success/empty/failure
rates for this run. The following figures compare **WERM+CV, WERM+P and
AMP ($T=20$)** for unknown count, and **WERM and AMP ($T=20$)** for known
$L=4$. WERM curves use the unchanged paired $T=10$ results.""".replace('__DIVISOR__', str(denominator))),
            code("""t20_root = T20_ROOTS[__DIVISOR__]
t20_payload = T20_PAYLOADS[__DIVISOR__]
t20_summary = T20_SUMMARIES[__DIVISOR__]
t20_status = experiment.status(t20_root)
print('T=20, Delta=n/__DIVISOR__:', json.dumps(t20_status, indent=2))
display({key: t20_payload['specification'][key] for key in settings_to_show})
display(t20_summary.round(5))
if not t20_status['complete']:
    print('PARTIAL SNAPSHOT: inspect trial counts before comparing iteration budgets.')
if not t20_summary.empty:
    t20_summary.to_csv(t20_root / 'notebook_summary.csv', index=False)
""".replace('__DIVISOR__', str(denominator))),
            md(r"""#### $T=20$, $\Delta_n=n/__DIVISOR__$: unknown count

True changes are at $(0.25,0.50)n$. The metric panels and histograms use the
same definitions and styling as the preceding $T=10$ comparison.""".replace('__DIVISOR__', str(denominator))),
            code("""payload = T20_PAYLOADS[__DIVISOR__]
if payload['records']:
    for path in experiment.plot_study(payload, study='unknown',
            output_dir=T20_ROOTS[__DIVISOR__] / 'figures' / 'unknown'):
        print('T=20, Delta=n/__DIVISOR__ — unknown count:', path.name)
        display(Image(filename=str(path)))
else:
    print('No completed T=20, n/__DIVISOR__ trials yet.')
""".replace('__DIVISOR__', str(denominator))),
            md(r"""#### $T=20$, $\Delta_n=n/__DIVISOR__$: known count

Both methods return exactly three boundaries ($L=4$), with true locations
$(0.25,0.45,0.70)n$. This comparison isolates localization.""".replace('__DIVISOR__', str(denominator))),
            code("""payload = T20_PAYLOADS[__DIVISOR__]
if payload['records']:
    for path in experiment.plot_study(payload, study='known',
            output_dir=T20_ROOTS[__DIVISOR__] / 'figures' / 'known'):
        print('T=20, Delta=n/__DIVISOR__ — known count:', path.name)
        display(Image(filename=str(path)))
else:
    print('No completed T=20, n/__DIVISOR__ trials yet.')
""".replace('__DIVISOR__', str(denominator))),
        ])
    cells.extend([
        md(r"""### Direct comparison of AMP iteration budgets

The direct plots compare $T=10$ and $T=20$ across the four sampling ratios;
each figure contains all three covariance settings. The following table gives
the corresponding values at the largest sampling ratio. Each comparison uses
the same completed $T=20$ datasets at both budgets, including in partial
snapshots. Localization means condition on successful, nonempty estimates;
empty and failure rates are reported alongside them. Counts include valid
empty estimates as zero. The paired Hausdorff difference shown separately uses
only trials with successful nonempty estimates at both budgets. These
comparisons contain AMP only, because the WERM results were reused without
modification."""),
        code("""for denominator in (20, 10):
    before, after = SPACING_PAYLOADS[denominator], T20_PAYLOADS[denominator]
    if not before['records'] or not after['records']:
        print(f'No paired iteration-budget plots yet for Delta=n/{denominator}.')
        continue
    for study in ('unknown', 'known'):
        for path in iteration_experiment.plot_amp_iteration_comparison(
                before, after, study=study,
                output_dir=T20_ROOTS[denominator] / 'figures' / 'iterations'):
            print(f'Delta=n/{denominator}, {study}: AMP T=10 versus T=20')
            display(Image(filename=str(path)))
"""),
        code("""iteration_tables, paired_difference_tables = [], []
for denominator in (20, 10):
    paired = iteration_experiment.paired_summary(
        SPACING_PAYLOADS[denominator], T20_PAYLOADS[denominator])
    if paired.empty:
        continue
    largest = max(T20_PAYLOADS[denominator]['specification']['deltas'])
    endpoint_paired = paired.loc[paired['delta'].eq(largest)].copy()
    metric_names = ['successes', 'nonempty_successes', 'failure_rate', 'mean_haus',
                    'mean_count', 'correct_count_rate', 'empty_rate']
    for iterations in (10, 20):
        names = [f'{key}_T{iterations}' for key in metric_names]
        endpoint = endpoint_paired[['study', 'covariance', 'trials', *names]].copy()
        endpoint = endpoint.rename(columns={f'{key}_T{iterations}': key for key in metric_names})
        endpoint.insert(0, 'AMP iterations', iterations)
        endpoint.insert(0, 'separation', f'n/{denominator}')
        iteration_tables.append(endpoint)
    differences = endpoint_paired[['study', 'covariance', 'trials', 'n_common_nonempty',
        'mean_paired_haus_T20_minus_T10', 'changed_locations_rate_common_successful']].copy()
    differences.insert(0, 'separation', f'n/{denominator}')
    paired_difference_tables.append(differences)
if iteration_tables:
    display(pd.concat(iteration_tables, ignore_index=True).round(5))
    print('Paired Hausdorff differences: negative values favor T=20; '
          'only the common nonempty subset contributes.')
    display(pd.concat(paired_difference_tables, ignore_index=True).round(5))
else:
    print('No AMP iteration-comparison results available yet.')
"""),
        md(r"""## Compact known-count figures: $\Delta_n=n/10$, $T=10$

Use the saved 100 paired trials with $p=200$, known $L=4$ and true locations
$(0.25,0.45,0.70)n$. The three separate Hausdorff panels have a common vertical
scale and follow the canonical M1--M3 plotting style; points are means and
bars span the first and third quartiles. The histogram rows are
$\delta\in\{7.5,11.5\}$, with WERM and AMP columns paired within each covariance.
Black dashed lines mark the true locations. Histograms use the canonical
12-bin density and Gaussian KDE conventions.

This cell only exports figures from the completed $T=10$ run. PDF and PNG
versions are saved in its `figures/compact_known` directory."""),
        code("""from IPython.display import Image, display
from tools.werm_amp_compact_plots import export_compact_known

COMPACT_KNOWN_FIGURES = export_compact_known()
for path in COMPACT_KNOWN_FIGURES:
    display(Image(filename=str(path), width=1100 if path == COMPACT_KNOWN_FIGURES[-1] else 480))
"""),
        md(r"""## Compact unknown-count figures: $\Delta_n=n/10$, $T=10$

Use the corrected 100 paired trials with $p=200$, true locations $(0.25,0.50)n$
and an upper bound of $L=4$ segments. Compare **WERM+CV** (10 folds),
**WERM+P** ($\gamma=6$) and **AMP** (10 iterations).

WERM+P fits the $L$ weighted least-squares regressions **once**, using occupancy
weights from a uniform prior on $K=1,\ldots,L$ and uniform admissible locations
conditional on $K$. It then searches using the first $K$ columns of that same
$L$-column fit, minimizing
$$\frac12\sum_{i=1}^n(y_i-\widehat\theta_{i,\psi_i})^2
  +6\log\binom{n-1}{K-1},\qquad 1\le K\le L.$$
Coefficients are not refitted for each candidate $K$. This is the upper-bound
search-penalty convention of `E_L_upper_bound.ipynb`; WERM+CV retains its
candidate-specific fits and absolute held-out loss.

Three Hausdorff panels and three count panels follow the canonical M1--M3
style, with a common scale
within each metric. Points are means and bars span the first and third
quartiles; the dotted count reference marks the true two change points.

Hausdorff summaries use successful, nonempty estimates. A sampling ratio with
no successful nonempty estimates has no Hausdorff point. Counts include valid
empty estimates as zero; failure and empty-estimate counts are retained in the
saved summary. The histogram has two rows ($\delta=7.5,11.5$) and nine columns:
WERM+CV, WERM+P and AMP within each covariance. It pools estimated locations
and uses the same 12-bin density and Gaussian KDE conventions as above, with
black dashed lines at the true locations.

The corrected run is saved separately under
`results/WERM_AMP_comparison/7b64ccb1c8c6`, with PDF and PNG figures in
`figures/compact_unknown`. The next cell reads completed results by default;
enable `RUN_CORRECTED_EXPERIMENT` to resume or reproduce this run."""),
        code("""import json
from IPython.display import Image, display
from tools import werm_amp_experiment as experiment
from tools.werm_amp_compact_unknown_plots import export_compact_unknown

CORRECTED_ROOT = experiment.ROOT / 'results/WERM_AMP_comparison/7b64ccb1c8c6'
RUN_CORRECTED_EXPERIMENT = False
if RUN_CORRECTED_EXPERIMENT:
    saved_spec = json.loads((CORRECTED_ROOT / 'specification.json').read_text())
    if experiment.prepare(saved_spec) != CORRECTED_ROOT:
        raise RuntimeError('Current sources or versions differ from the saved run.')
    experiment.run(CORRECTED_ROOT, jobs=16)
if not experiment.status(CORRECTED_ROOT)['complete']:
    raise RuntimeError('Corrected study is incomplete; resume it before exporting all 100 trials.')

COMPACT_UNKNOWN_FIGURES = export_compact_unknown(
    CORRECTED_ROOT, expected_truth=(.25, .50))
for path in COMPACT_UNKNOWN_FIGURES:
    display(Image(filename=str(path), width=1450 if path == COMPACT_UNKNOWN_FIGURES[-1] else 480))
"""),
        md(r"""## Relocated unknown-count comparison: true locations $(0.45,0.75)n$

Repeat the corrected unknown-count experiment with true change points at
$(0.45,0.75)n$. All other settings remain fixed: $p=200$,
$\delta\in\{1.5,3.5,7.5,11.5\}$, $\Delta_n=n/10$, upper bound $L=4$,
100 paired trials per setting, 10-fold WERM+CV, WERM+P with $\gamma=6$,
and AMP with $T=10$. WERM+P again fits the upper-bound mixture-weighted
regressions once and searches the first $K$ columns of that same $L$-column
fit using half-squared loss plus $6\log\binom{n-1}{K-1}$; there is no
candidate-wise coefficient refit. The three covariances,
coefficient/noise distributions, prior weights and AMP settings are unchanged
from the corrected comparison above; no method receives whitened
data or the true locations. This rerun uses the same latent draws and AMP
seeds as the preceding experiment, with the response regenerated for the new
locations.

The three Hausdorff panels, three count panels and two-row histogram retain
the preceding plotting conventions. Histogram rows are $\delta=7.5,11.5$;
black dashed lines now mark $0.45$ and $0.75$. Hausdorff summaries condition
on successful nonempty estimates; count summaries include empty estimates
as zero. A sampling ratio with no successful nonempty estimates has no
Hausdorff point. Full diagnostics are retained in the saved summary and audit.

Results and PDF/PNG figures are saved separately under
`results/WERM_AMP_comparison/448fa5a988cc`. The next cell reads this completed
run by default; enable `RUN_RELOCATED_EXPERIMENT` to resume or reproduce it."""),
        code("""import json
from IPython.display import Image, display
from tools import werm_amp_experiment as experiment
from tools.werm_amp_compact_unknown_plots import export_compact_unknown

RELOCATED_ROOT = experiment.ROOT / 'results/WERM_AMP_comparison/448fa5a988cc'
RUN_RELOCATED_EXPERIMENT = False
if RUN_RELOCATED_EXPERIMENT:
    saved_spec = json.loads((RELOCATED_ROOT / 'specification.json').read_text())
    if experiment.prepare(saved_spec) != RELOCATED_ROOT:
        raise RuntimeError('Current sources or versions differ from the saved run.')
    experiment.run(RELOCATED_ROOT, jobs=16)
if not experiment.status(RELOCATED_ROOT)['complete']:
    raise RuntimeError('Relocated study is incomplete; resume it before exporting all 100 trials.')

RELOCATED_FIGURES = export_compact_unknown(
    RELOCATED_ROOT, expected_truth=(.45, .75))
for path in RELOCATED_FIGURES:
    display(Image(filename=str(path), width=1450 if path == RELOCATED_FIGURES[-1] else 480))
"""),
        md(r"""## Five-column unknown-count histogram

Use the same completed $\Delta_n=n/10$, $T=10$ experiment with true locations
$(0.45,0.75)n$. Rows show $\delta=7.5$ and $11.5$. Columns show **WERM+CV
(AR(1)), WERM+P (AR(1)), AMP isotropic, AMP AR(1), and AMP random orthogonal**,
in that order. Each panel uses all 100 trials from its indicated design and the
same histogram/KDE conventions as above. This cell only renders saved estimates."""),
        code("""from IPython.display import Image, display
from tools import werm_amp_experiment as experiment
from tools.werm_amp_compact_unknown_plots import export_five_column_unknown_histogram

FIVE_COLUMN_HISTOGRAM_ROOT = experiment.ROOT / 'results/WERM_AMP_comparison/448fa5a988cc'
FIVE_COLUMN_HISTOGRAM = export_five_column_unknown_histogram(
    FIVE_COLUMN_HISTOGRAM_ROOT, expected_truth=(.45, .75))
display(Image(filename=str(FIVE_COLUMN_HISTOGRAM), width=1150))
"""),
        reproducibility,
    ])
    # Preserve the original reference outputs and stable cell IDs until the
    # caller deliberately executes the new notebook snapshot.
    if previous is not None:
        old_by_source = {(cell.cell_type, cell.source): cell for cell in previous.cells}
        preserved_ids = {old_by_source[cell.cell_type, cell.source].id
                         for cell in cells if (cell.cell_type, cell.source) in old_by_source}
        for index, cell in enumerate(cells):
            old = old_by_source.get((cell.cell_type, cell.source))
            if old is not None:
                cells[index] = copy.deepcopy(old)
            elif index < len(previous.cells) and previous.cells[index].id not in preserved_ids:
                cell.id = previous.cells[index].id
    for cell in cells:
        if cell.cell_type == 'code' and any(marker in cell.source for marker in (
                'UNKNOWN_FIGURES = experiment.plot_study(',
                'KNOWN_FIGURES = experiment.plot_study(',
                'payload = SPACING_PAYLOADS[')) and 'experiment.plot_study(' in cell.source:
            tags = list(cell.metadata.get('tags', []))
            for tag in ('baseline-plots', 'existing-plots', 'skip-execution'):
                if tag not in tags:
                    tags.append(tag)
            cell.metadata['tags'] = tags
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            'kernelspec': {
                'display_name': 'Python (weighted-erm)',
                'language': 'python',
                'name': 'weighted-erm',
            },
            'language_info': (copy.deepcopy(previous.metadata['language_info'])
                if previous is not None and 'language_info' in previous.metadata
                else {'name': 'python', 'version': sys.version.split()[0]}),
            'werm_amp_spacing_roots': spacing_roots,
            'werm_amp_t20_roots': t20_roots,
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, destination)
    return destination


if __name__ == '__main__':
    print(build())
