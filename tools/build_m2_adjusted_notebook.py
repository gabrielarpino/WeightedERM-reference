"""Build the two-part adjusted-WERM notebook without rerunning existing studies."""
from pathlib import Path
import nbformat as nbf


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r'''# M2: WERM versus WERM adjusted

Two paired experiments, each with **100 trials at each** $\delta=n/p\in\{1.5,3.5,7.5,11.5\}$,
$p=100$. Only WERM and WERM adjusted are compared.

The original M2 data generator is retained: `SparseDiffSignal(var_1=8,
sigma_w=sqrt(400*delta), alpha=0.3, L=4)`, independent Gaussian covariates
$X_{ij}\sim N(0,1/n)$, and Gaussian noise with standard deviation $\sqrt{0.1}$.
The prior rescales perturbed coefficients to preserve marginal variance, exactly as in M2.
Within each trial both methods receive identical data, marginal weights, and CV folds.

- **Unknown count:** true changes at $(0.3,0.55)n$; candidates have 1–4 segments.
  Each method independently selects its count with the original M2 interlaced 10-fold CV.
- **Known count:** exactly three changes at $(0.25,0.45,0.7)n$; fixed $L=4$, no CV.

All fits use unpenalized squared loss, no intercept, and no partition penalty.
Minimum internal-boundary separation follows M2's helper:
$\max(\lceil n_{\mathrm{fit}}/50\rceil,\lfloor n_{\mathrm{fit}}/40\rfloor,1)$.
The native DP does not impose that separation on the two end segments.
The existing experiments and their saved results are not changed.'''),
    md(r'''## Adjustment checked against the paper and myocardial notebook

`Myocardial Infarction Experiment - final_final.ipynb`, case A, uses
`WERM.signal_strength.empirical_adjustment`; its adjusted fit is named `V`.
This implements `eq:theta-deb-defn` and `eq:b-hat-eqn` in
`../WERM-paper/JRSSB-revision-rebuttal/JRSSB_revision.tex`:
$$\theta^{\mathrm{adj}}_{i\ell}=\hat\theta_{i\ell}
+\hat b_\ell\pi_i^{(\ell)}M'(\hat\theta_{i\ell},y_i),\qquad
\frac1n\sum_i\frac1{1+\hat b_\ell\pi_i^{(\ell)}M''(\hat\theta_{i\ell},y_i)}=1-\frac pn.$$
For this experiment $M(u,y)=(u-y)^2$, hence $M'=2(u-y)$ and $M''=2$.
The **plus sign is intentional**. The same weighted coefficient fits are used;
the adjusted method substitutes $\Theta^{\mathrm{adj}}$ for $X\hat B$ only in
the squared-loss partition search. It does not fit a new coefficient matrix.

In CV the adjustment uses only training responses and the fold-local $n/p$.
Each method's training partition is mapped back to original indices, followed
by ordinary segment coefficient refits and **absolute held-out prediction error**,
exactly as in the original M2 helper. Held-out responses never enter the adjustment.
The final search is repeated on all observations at the independently selected count.

These are empirical comparisons, not claims that the adjustment improves localization.
Finite-root failures and failed CV candidates are retained in diagnostics.
The scalar $b$ equations suffice; no full $(\Lambda,K)$ system is needed here.'''),
    code('''# Run from the WeightedERM-reference directory; no earlier notebook is needed.
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle
from IPython.display import display
from tools.m2_adjusted_experiment import (
    run_experiment, summarize, DELTAS, METHODS, FRACTIONS)
from tools import m2_adjusted_relocated as relocated

DIRECTORY = relocated.PARENT
SPEC = json.loads((DIRECTORY/'specification.json').read_text())
WEIGHTS = pickle.loads((DIRECTORY/'weights.pkl').read_bytes())
UNKNOWN_DIRECTORY, UNKNOWN_SPEC, UNKNOWN_WEIGHTS = relocated.prepare()
# Load the unchanged known-count experiment for partial reruns of this notebook.
known = pickle.loads((DIRECTORY/'known.pkl').read_bytes())
known_summary = summarize(known)
JOBS = int(os.environ.get('WERM_N_JOBS','12'))
print('Unknown-count checkpoints and results:', UNKNOWN_DIRECTORY)
print('Unchanged known-count results:', DIRECTORY)
print('Native marginal curves are frozen and shared by both methods.')'''),
    md('''## 1. Unknown number of change points — separate 10-fold CV

This cell reuses completed, provenance-checked trial checkpoints. It computes missing
trials if necessary. Localization summaries exclude empty estimates and failed fits;
the table reports both explicitly. Correct-count frequencies use successful fits.'''),
    code('''unknown = relocated.run(UNKNOWN_DIRECTORY, UNKNOWN_SPEC, UNKNOWN_WEIGHTS, jobs=JOBS)
unknown_summary = summarize(unknown)
unknown_summary.to_csv(UNKNOWN_DIRECTORY/'unknown_summary.csv',index=False)
display(unknown_summary.round(3))'''),
    md('''## Publication plotting setup

Use the final M1/M2/M3 conventions: TeX/Computer Modern, 5×4-inch line canvases,
18 pt axis labels and 14 pt ticks; means with absolute distances to the first and
third quartiles, **not confidence intervals**. Compact boxes use the shared
paper-sized style, 1.5-IQR whiskers and faint deterministic trial jitter.
Histograms use 12 data-range bins, five-fold bandwidth-selected Gaussian KDE,
truth-position ticks, and the canonical 0–18 density range (clipped peaks reported).
WERM is black; WERM adjusted is blue. Runtime is a diagnostic including each method's
own fits, CV, adjustment and final search; common marginal preparation is excluded.'''),
    code('''import tools.consistent_plotting as style
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
style.OUTPUT_DIR = DIRECTORY/'figures'
style.SOURCE_HASHES.clear()
style.FIGURE_AUDIT.clear()
LABELS = (r'\\textbf{WERM}', r'\\textbf{WERM adjusted}')
COLORS = ('black','#4477AA')

def groups_for(payload, name, metric):
    return [[r['methods'][name][metric] for r in payload['records']
             if r['setting']==i and r['methods'][name]['error'] is None]
            for i in range(len(DELTAS))]

def plot_comparison(mode):
    result_directory = UNKNOWN_DIRECTORY if mode=='unknown' else DIRECTORY
    style.OUTPUT_DIR = result_directory/'figures'
    style.FIGURE_AUDIT.clear()
    payload = style.load_saved(result_directory/f'{mode}.pkl')
    metrics = [('haus',r'Hausdorff dist. / $n$'), ('runtime','Runtime (s)')]
    if mode=='unknown': metrics.insert(1,('size','No. change points'))
    for metric,ylabel in metrics:
        with plt.rc_context(style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            for j,name in enumerate(METHODS):
                style.error_series(ax,DELTAS,groups_for(payload,name,metric),
                    LABELS[j],COLORS[j],('D','o')[j],('-','--')[j],scale=1)
            ax.set(xlabel=r'$\\delta=n/p$',ylabel=ylabel,xticks=DELTAS)
            if metric=='size':
                ax.axhline(2,color='gray',alpha=.8,lw=2.5)
                ax.set_yticks([0,1,2,3])
            if metric=='runtime': ax.legend(loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();style.export_plot(fig,f'{mode}_{metric}_line')
        if metric=='haus':
            with plt.rc_context(style.PAPER_STYLE):
                fig,ax=style.paper_panel();rng=np.random.default_rng(42)
                for j,name in enumerate(METHODS):
                    for i,values in enumerate(groups_for(payload,name,metric)):
                        style.box_group(ax,values,3*i+j,COLORS[j],rng)
                ax.set(xticks=np.arange(4)*3+.5,xticklabels=[str(d) for d in DELTAS],
                       xlabel=r'$\\delta=n/p$',ylabel=ylabel)
                ax.legend(handles=[Patch(facecolor=c,alpha=.28,label=l) for c,l in zip(COLORS,LABELS)],
                          fontsize=7,loc='best')
                style.export_plot(fig,f'{mode}_{metric}_box')
    hist=[]
    for i in range(4):
        row=[]
        for name in METHODS:
            pieces=[r['methods'][name]['cp']/r['n'] for r in payload['records']
                    if r['setting']==i and r['methods'][name]['error'] is None]
            row.append(np.concatenate(pieces) if pieces else np.array([]))
        hist.append(row)
    style.histogram_grid(hist,LABELS,COLORS,payload['spec']['fractions'][mode],f'{mode}_histograms',
        row_labels=[rf'$\\delta={d:g}$' for d in DELTAS],density_label_x=.005)
    style.verify_sources()
'''),
    code("plot_comparison('unknown')"),
    md('''## 2. Known number of change points — fixed L=4, no CV

Three true changes at fractions 0.25, 0.45 and 0.7. Each method searches over
partitions with exactly four segments. This isolates the adjustment's effect on
locations without count selection.'''),
    code('''known = run_experiment('known', DIRECTORY, SPEC, WEIGHTS, jobs=JOBS)
known_summary = summarize(known)
known_summary.to_csv(DIRECTORY/'known_summary.csv',index=False)
display(known_summary.round(3))
assert all(f['size']==3 for r in known['records'] for f in r['methods'].values()
           if f['error'] is None)'''),
    code("plot_comparison('known')"),
    md('''## Paired localization differences and numerical diagnostics

Differences are **adjusted minus standard** normalized Hausdorff error; positive
values mean worse localization with adjustment. Unknown-count differences are
conditional on **both** methods returning a nonempty estimate. This restriction
is separate from each method's own conditional means above. Counts of excluded
pairs are reported. The known-count comparison includes every successful pair.'''),
    code('''paired=[]
for mode,payload in [('unknown',unknown),('known',known)]:
    for i,delta in enumerate(DELTAS):
        records=[r for r in payload['records'] if r['setting']==i]
        differences=np.array([r['methods'][METHODS[1]]['haus']-r['methods'][METHODS[0]]['haus']
            for r in records if all(np.isfinite(r['methods'][m]['haus']) for m in METHODS)])
        paired.append(dict(experiment=mode,delta=delta,paired_nonempty=len(differences),
            excluded_pairs=len(records)-len(differences),mean_adjusted_minus_standard=differences.mean(),
            median_adjusted_minus_standard=np.median(differences),
            adjusted_better=int(np.sum(differences < -1e-12)),
            tied=int(np.sum(np.abs(differences)<=1e-12)),
            adjusted_worse=int(np.sum(differences > 1e-12))))
paired_table=pd.DataFrame(paired)
paired_table.to_csv(UNKNOWN_DIRECTORY/'paired_localization.csv',index=False)
display(paired_table.round(4))

boundary_rows=[]
for i,delta in enumerate(DELTAS):
    for name in METHODS:
        errors=np.array([(r['methods'][name]['cp']-r['truth'])/r['n']
            for r in known['records'] if r['setting']==i and r['methods'][name]['error'] is None])
        for j,location in enumerate(FRACTIONS['known']):
            boundary_rows.append(dict(delta=delta,method=name,true_fraction=location,
                mean_signed_error=errors[:,j].mean(),mean_absolute_error=np.abs(errors[:,j]).mean()))
boundary_table=pd.DataFrame(boundary_rows)
boundary_table.to_csv(UNKNOWN_DIRECTORY/'known_boundary_errors.csv',index=False)
display(boundary_table.round(4))
failures=[dict(experiment=mode,delta=DELTAS[r['setting']],trial=r['trial'],method=m,error=f['error'])
          for mode,payload in [('unknown',unknown),('known',known)] for r in payload['records']
          for m,f in r['methods'].items() if f['error'] is not None]
print('Failed method runs:',len(failures))
print('Maximum b-equation residual:',max(unknown_summary.max_b_residual.max(),known_summary.max_b_residual.max()))
if failures: display(pd.DataFrame(failures))
style.verify_sources()'''),
    md('''## Reproducibility and verification

Implementation: `tools/m2_adjusted_experiment.py`. Tests:
`tests/test_m2_adjusted_experiment.py` verify the fast additive-loss partition search
against the original exact DP (including ties), standard CV scores/counts/locations
against the original M2 helper with identical weights, and the paper's adjustment
against an analytically solvable case. Per-trial artifacts retain data hashes,
CV scores, selected counts, b values/residuals, failures, estimates and timings.
`specification.json` records parameters and source hashes; plotting checks that
saved input artifacts remain unchanged. No original M2 or myocardial result is overwritten.

The unknown-count rerun uses `tools/m2_adjusted_relocated.py` with true fractions
$(0.30,0.55)$. It reuses the original coefficient, covariate and noise draws,
frozen weights, and estimator/CV implementation. Its artifacts are saved separately
under `results/M2_adjusted_comparison/unknown_030_055_*`. The known-count experiment
and its displayed outputs remain those of the original run in `ecd9d5c2529d`.''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'python3','display_name':'Python (weighted-erm)','language':'python'}})
    path = Path('M2_adjusted_comparison.ipynb')
    if path.exists():
        existing = nbf.read(path, as_version=4)
        for i, cell in enumerate(notebook.cells):
            if i < len(existing.cells) and cell.source == existing.cells[i].source:
                notebook.cells[i] = existing.cells[i]
    nbf.write(notebook, path)
    return path


if __name__ == '__main__':
    print(build())
