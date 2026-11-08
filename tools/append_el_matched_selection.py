"""Append paired CV/penalty upper-bound results without altering earlier studies."""
import json
from pathlib import Path

import nbformat as nbf


TAG = 'el-matched-cv-penalty-quarter-locations-v1'


def append():
    path = Path('E_L_upper_bound.ipynb')
    before = path.read_bytes()
    notebook = json.loads(before)
    original = [c for c in notebook['cells']
                if TAG not in c.get('metadata', {}).get('tags', [])]
    previous = [c for c in notebook['cells']
                if TAG in c.get('metadata', {}).get('tags', [])]
    backup = Path('results/E_L_matched_CV_penalty/notebook_before_matched_selection.ipynb')
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(before)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r'''## Matched upper-bound sensitivity: WERM+CV and WERM+P

Both procedures now use **the same 100 datasets** at $n=1000$, $p=100$, with
three true change points at **$(0.25,0.50,0.75)n$** ($L^\star=4$). Each dataset
is reused for every upper bound $L=1,\ldots,6$. The locations are specified in
advance; no layout search or calibration is performed for this comparison.

Retain the existing upper-bound experiment's generator and base seed 20261120:
$X_{ij}\sim N(0,1/n)$, noise standard deviation $0.5$,
$\beta^{(1)}\sim N(0,(n/p)I_p)$, and independent successive coefficient jumps
distributed as $N(0,1.5^2(n/p)I_p)$. Both methods use half-squared loss and
unpenalized coefficient fits.

- **WERM+CV:** for each candidate $K\le L$, use its fixed-count conditional
  uniform-prior weights, perform exhaustive interlaced 10-fold CV, refit each
  estimated training segment by ordinary squared-loss regression, and score
  held-out observations by absolute error. Select the smallest score and rerun
  fixed-count WERM on all observations. Candidate scores are computed once and
  their prefixes give the selections for each $L$.
- **WERM+P:** for each $L$, fit its $L$ weighted regressions using a uniform prior
  on $K\in\{1,\ldots,L\}$ and a conditional uniform location prior. Search using
  the first $K$ of these fitted columns, with the frozen penalty
  $P(K)=6\log\binom{n-1}{K-1}$. There is no candidate-wise coefficient refit or
  tuning of $\gamma=6$.

The differing prior weights are part of the definitions of these two procedures.
Both use the same native 50-point reference grid with gap 5. Marginal curves are
frozen independently of the observed responses using seed 0, as in the preceding
penalty study, and reused across trials and repeated candidates. Full-data weights
are reused from that study; CV training weights use the corresponding 900-row
grid. The earlier CV run regenerated marginal curves using trial-dependent
optimizer seeds; this matched study freezes that numerical approximation.

Both searches preserve the implemented internal change-point separation:
$\Delta_n=100$ on all observations and $\Delta_m=90$ in training folds; endpoint
segments need only be nonempty, following the native implementation.

The empirical comparison is followed by the existing penalty state-evolution
construction evaluated on this same setting. Theory uses 20 independent draws
per true signal matrix and applies its own penalized search; empirical estimates
and selected counts never enter it. These are predictions for **WERM+P** only.
No CV theory overlay is asserted here, since the original CV procedure permits
segment refits with sampling ratio at or below one.'''),
    code('''# Self-contained: run directly after restarting the kernel.
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from tools import el_matched_selection as matched

MATCH_ROOT,MATCH_SPEC,MATCH_WEIGHTS=matched.prepare(trials=100,theory_draws=20)
MATCH_FILE=MATCH_ROOT/'study.pkl'
MATCH_STUDY=(pickle.loads(MATCH_FILE.read_bytes()) if MATCH_FILE.exists()
             else matched.run(MATCH_ROOT,MATCH_SPEC,MATCH_WEIGHTS,
                 jobs=int(os.environ.get('EL_MATCHED_JOBS','12'))))
MATCH_SUMMARY=matched.summary(MATCH_STUDY)
print('Matched study:',MATCH_ROOT)
print('True locations:',MATCH_SPEC['true_fractions'],'; gamma:',MATCH_SPEC['gamma'])
display(MATCH_SUMMARY.round(5))'''),
    md(r'''### Joint comparison with penalty Theory

Black: **WERM+CV**; purple: **WERM+P**. Both curves summarize the same 100 trials.
Blue diamonds overlay the **WERM+P Theory** predictions on the joint mean panels;
they have no connecting lines or error bars. The boxplots and histograms show
the two empirical methods.
The horizontal gray count reference is the true value, three changes. Mean panels
retain the canonical M1/M2/M3 canvas and mean-to-quartile bars; these bars are not
confidence intervals. Boxplots show 1.5-IQR whiskers and faint trial points.

Hausdorff distance is normalized by $n$ and conditional on a nonempty estimate.
The summary table retains empty-estimate counts; no trials are removed from count
summaries. In particular $L=1$ remains a tick but has no Hausdorff observation.
The compact location histograms show only $L=3$ and $L=6$, use the canonical
12 bins and five-fold bandwidth-selected KDE, and mark the true fractional
locations. The shared legend appears only on the count panels. The
density display limit is 18; any clipped peaks are reported by the plotting helper.
PDF and PNG copies are exported to the printed result directory.'''),
    code(r'''# Independently load the completed matched study for plotting.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import tools.consistent_plotting as match_style

# Plot the completed, frozen study without preparing or running new experiments.
MATCH_ROOT=Path('results/E_L_matched_CV_penalty/3af038cb5e34')
match_style.OUTPUT_DIR=MATCH_ROOT/'figures'
match_style.SOURCE_HASHES.clear();match_style.FIGURE_AUDIT.clear()
MATCH_SAVED=match_style.load_saved(MATCH_ROOT/'study.pkl')
MATCH_SPEC=MATCH_SAVED['specification']
MATCH_RECORDS=MATCH_SAVED['records']
MATCH_BOUNDS=np.arange(1,7)
MATCH_METHODS=('cv','penalty')
MATCH_LABELS=(r'\textbf{WERM+CV}',r'\textbf{WERM+P}')
MATCH_COLORS=('black','#AA3377')

def matched_groups(method,metric):
    return [np.array([fit[metric] for r in MATCH_RECORDS for fit in
        (r['fits'][L]['theory'] if method=='theory' else [r['fits'][L][method]])])
        for L in MATCH_BOUNDS]

for metric,ylabel in [('count','No. change points'),('haus',r'Hausdorff dist. / $n$')]:
    for kind in ('line','box'):
        with plt.rc_context(match_style.CANONICAL_STYLE if kind=='line' else match_style.PAPER_STYLE):
            if kind=='line':
                fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            else:
                fig,ax=match_style.paper_panel();rng=np.random.default_rng(42)
            for j,method in enumerate(MATCH_METHODS):
                groups=matched_groups(method,metric)
                if kind=='line':
                    match_style.error_series(ax,MATCH_BOUNDS,groups,MATCH_LABELS[j],
                        MATCH_COLORS[j],('D','o')[j],('-','--')[j],scale=1)
                else:
                    for i,values in enumerate(groups):
                        match_style.box_group(ax,values,3*i+j,MATCH_COLORS[j],rng)
            ax.set(xlabel=r'$L$',ylabel=ylabel)
            if kind=='line':
                theory=matched_groups('theory',metric)
                means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in theory]
                theory_label=r'\textbf{Theory}'
                ax.scatter(MATCH_BOUNDS,means,label=theory_label,marker='d',
                    color='#4477AA',s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
                ax.set_xticks(MATCH_BOUNDS)
                handles,labels=ax.get_legend_handles_labels()
                ordered=(*MATCH_LABELS,theory_label)
                if metric=='count':
                    ax.legend([handles[labels.index(label)] for label in ordered],ordered,
                        loc='best',prop={'size':14},framealpha=.4)
            else:
                ax.set_xticks(np.arange(len(MATCH_BOUNDS))*3+.5,
                              [str(v) for v in MATCH_BOUNDS])
                if metric=='count':
                    ax.legend(handles=[Patch(facecolor=c,alpha=.28,edgecolor='black',label=l)
                        for c,l in zip(MATCH_COLORS,MATCH_LABELS)],loc='best',fontsize=7)
            if metric=='count':
                ax.axhline(3,color='gray',lw=2.5 if kind=='line' else 1.4,alpha=.8,zorder=0)
                ax.set_yticks(range(6))
            if kind=='line':fig.tight_layout()
            match_style.export_plot(fig,f'matched_upper_L_{metric}_{kind}')

MATCH_HIST_BOUNDS=(3,6)
MATCH_POOLS=[[np.concatenate([np.asarray(r['fits'][L][method]['cp'])/MATCH_SPEC['n']
                              for r in MATCH_RECORDS]) for method in MATCH_METHODS]
             for L in MATCH_HIST_BOUNDS]
match_style.histogram_grid(MATCH_POOLS,MATCH_LABELS,MATCH_COLORS,
    MATCH_SPEC['true_fractions'],'matched_upper_L_histograms_L3_L6',
    row_labels=[rf'$L={L}$' for L in MATCH_HIST_BOUNDS],
    density_label_x=.005,figsize=(5,4))
match_style.verify_sources()'''),
    md(r'''### Penalty theory on the matched setting

These two panels compare **WERM+P** with **Theory**, using the squared-loss SE
equations and proximal fitted-value construction documented above. The theory
shares the same frozen upper-bound weights, true signal matrix, spacing, and
$\gamma=6$ penalty. Its empirical design/noise and theoretical Gaussian draws are
independent conditional on the signal matrix. All 2,000 realizations enter each
theory mean. Blue diamonds have no connecting lines or error bars, following the
SE estimation notebook.'''),
    code(r'''for metric,ylabel in [('count','No. change points'),('haus',r'Hausdorff dist. / $n$')]:
    with plt.rc_context(match_style.CANONICAL_STYLE):
        fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
        match_style.error_series(ax,MATCH_BOUNDS,matched_groups('penalty',metric),
            MATCH_LABELS[1],MATCH_COLORS[1],'o','--',scale=1)
        theory=matched_groups('theory',metric)
        means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in theory]
        ax.scatter(MATCH_BOUNDS,means,label=r'\textbf{Theory}',marker='d',
            color='#4477AA',s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
        ax.set(xlabel=r'Upper bound $L$',ylabel=ylabel,xticks=MATCH_BOUNDS)
        if metric=='count':
            ax.axhline(3,color='gray',lw=2.5,alpha=.8,zorder=0)
            ax.set_yticks(range(6))
        handles,labels=ax.get_legend_handles_labels()
        order=[labels.index(label) for label in (MATCH_LABELS[1],r'\textbf{Theory}')]
        ax.legend([handles[i] for i in order],(MATCH_LABELS[1],r'\textbf{Theory}'),
                  loc='best',prop={'size':14},framealpha=.4)
        fig.tight_layout();match_style.export_plot(fig,f'matched_upper_L_penalty_theory_{metric}')
match_style.verify_sources()'''),
    md('''### Completeness and provenance

Each trial stores one shared data hash, both empirical procedures, every candidate
CV score, all six upper-bound selections, and the independent penalty-theory
outcomes. The checks retain every trial and verify the fixed settings and SE
residuals. Raw source data are not altered by plotting. Earlier notebook sections
and artifacts remain unchanged.'''),
    code('''assert MATCH_SPEC['gamma']==6.
assert MATCH_SPEC['n']==1000 and MATCH_SPEC['p']==100
np.testing.assert_array_equal(MATCH_SPEC['true_fractions'],[.25,.5,.75])
assert len(MATCH_RECORDS)==100
assert {r['trial'] for r in MATCH_RECORDS}==set(range(100))
assert len({r['data_hash'] for r in MATCH_RECORDS})==100
for record in MATCH_RECORDS:
    np.testing.assert_array_equal(record['truth'],[250,500,750])
    assert set(record['fits'])==set(MATCH_BOUNDS)
    scores=np.asarray(record['cv']['cv_scores'])
    assert scores.shape==(6,) and np.isfinite(scores).all()
    for L in MATCH_BOUNDS:
        outcomes=record['fits'][L]
        assert outcomes['cv']['segments']==int(np.argmin(scores[:L]))+1
        assert len(outcomes['theory'])==20
        for fit in [outcomes['cv'],outcomes['penalty'],*outcomes['theory']]:
            assert fit['count']==len(fit['cp'])==fit['segments']-1
            assert 0<=fit['count']<L
            assert bool(fit['empty'])==(fit['count']==0)
            assert (not np.isfinite(fit['haus'])) if fit['empty'] else np.isfinite(fit['haus'])
max_residual=max(r['se_residual'] for r in MATCH_RECORDS)
assert max_residual<1e-9
match_style.verify_sources()
print('Validated 100 paired datasets, six bounds, gamma=6, and all theory realizations.')
print('Maximum SE residual:',max_residual)
print('Implementation: tools/el_matched_selection.py')
print('Summary:',MATCH_ROOT/'summary.csv')
print('Figures:',MATCH_ROOT/'figures')''')]
    for i, cell in enumerate(cells):
        cell['metadata']['tags'] = [TAG]
        if i < len(previous) and cell.source == ''.join(previous[i]['source']):
            cells[i] = nbf.from_dict(previous[i])
    appended = json.loads(nbf.writes(nbf.v4.new_notebook(cells=cells)))['cells']
    notebook['cells'] = original + appended
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    assert json.loads(path.read_text())['cells'][:len(original)] == original
    return path, len(original)


if __name__ == '__main__':
    print(append())
