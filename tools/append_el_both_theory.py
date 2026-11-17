"""Append the approved n=1500 comparison with Theory for both methods."""
import json
from pathlib import Path

import nbformat as nbf


TAG = 'el-matched-both-theory-n1500-v1'


def append():
    path = Path('E_L_upper_bound.ipynb')
    before = path.read_bytes()
    notebook = json.loads(before)
    original = [c for c in notebook['cells']
                if TAG not in c.get('metadata', {}).get('tags', [])]
    previous = [c for c in notebook['cells']
                if TAG in c.get('metadata', {}).get('tags', [])]
    backup = Path('results/E_L_matched_CV_penalty_both_theory/notebook_before_both_theory.ipynb')
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(before)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r'''## Joint comparison with Theory for WERM+CV and WERM+P

This approved companion experiment uses **$n=1500$, $p=100$, 100 paired trials**,
with true changes at **$(0.25,0.50,0.75)n$**, upper bounds $L=1,\ldots,6$, and
the frozen penalty coefficient **$\gamma=6$**. Both empirical methods and both
theory predictions use this new setting. The preceding $n=1000$ results remain
a separate comparison.

The larger sample size supports the uniform-refit CV theory: the earlier
$n=1000$ setting produced 90-row refits for some five- and six-segment
candidates, below $p=100$. Here each training fold has 1350 observations.
The native internal-gap constraints are 150 rows in full-data searches and
135 rows in training searches; endpoint segments remain merely nonempty.
Every empirical and theoretical segment, including endpoints and unselected
candidates, is audited below for $m_s/p>1$. No trial or candidate is removed
to satisfy that audit.

Retain the Gaussian coefficient random walk, design normalization
$X_{ij}\sim N(0,1/n)$, noise standard deviation $0.5$, base seed 20261120,
half-squared fitting loss, and unpenalized coefficients. The initial coefficient
variance is $n/p$ and each independent jump has variance $1.5^2n/p$.
WERM+CV uses exhaustive interlaced 10-fold CV, squared-loss segment refits,
absolute held-out error, and conditional fixed-count prior weights.
WERM+P uses upper-bound mixture weights and minimizes the unnormalized
half-squared partition loss plus $6\log\binom{n-1}{K-1}$, with $1\le K\le L$.
Native marginal curves use the same 50-point grid, gap 5, and frozen seed 0,
recomputed at the new full and training sample sizes.'''),
    md(r'''### CV Theory construction

Use the same first-order construction as `E-WERM-CV-theory.ipynb`.
For a training fold of size $m$, solve the weighted squared-loss equations with
$\delta_{\rm train}=m/p$ and $\Gamma=B^\top B/n$. The latter is unchanged by
the equivalent fold normalization $\widetilde X=X_{\rm train}/\sqrt{m/n}$,
$\widetilde B=\sqrt{m/n}B$.

For each candidate count, search the deterministic expected SE residual-loss
profile over the same admissible training partitions. In each resulting segment
$s$, let $q_s$ be its true-regime proportions, $\delta_s=m_s/p$, and
$D_s=q_s^\top\operatorname{diag}(\Gamma)$, $Q_s=q_s^\top\Gamma q_s$.
Uniform-weight squared-loss refit theory gives
$$\lambda_s=q_s,\qquad
\kappa_s=\frac{\sigma^2+D_s-Q_s}{\delta_s-1},\qquad \delta_s>1.$$
For a held-out observation in true regime $r$, the predicted absolute loss is
$$R_{sr}=\sqrt{\frac{2}{\pi}}
\sqrt{\sigma^2+(q_s-e_r)^\top\Gamma(q_s-e_r)+\kappa_s}.$$
Average over the actual held-out segment/regime overlaps and folds, then take
the theory's own CV argmin among $1,\ldots,L$. At that selected count, solve SE
again using the full-data weights and ratio, and apply the fixed-count partition
search to 20 independent proximal fitted-value realizations. WERM+P Theory
uses the separate upper-bound weights and its own penalized search, as above.

Neither prediction receives empirical fitted coefficients, estimated partitions,
validation scores, or selected counts. The oracle true signal matrix is used
only to evaluate theory predictions. CV Theory has one selected count per true
signal matrix; its 20 final draws describe localization, not 20 separate CV
selections. They are not treated as independent datasets.

This composition is a **first-order extension to CV**, requiring stability of
the limiting training partitions and their data-selected refits, and separated
limiting score minima. It is not a proved finite-sample joint law for overlapping
folds or selection frequencies. Finite-union concentration does not require
independent folds, but does not by itself supply those stability assumptions.
As in the earlier CV theory notebook, native weights can have zero-support
regions; extending the paper's strict-positivity regularity assumption remains
a separate theoretical requirement.'''),
    code('''# Self-contained companion run; matching completed checkpoints are reused.
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from tools import el_matched_cv_theory as both

BOTH_ROOT,BOTH_SPEC,BOTH_WEIGHTS=both.prepare(trials=100,theory_draws=20)
BOTH_FILE=BOTH_ROOT/'study.pkl'
BOTH_STUDY=(pickle.loads(BOTH_FILE.read_bytes()) if BOTH_FILE.exists()
            else both.run(BOTH_ROOT,BOTH_SPEC,BOTH_WEIGHTS,
                jobs=int(os.environ.get('EL_BOTH_THEORY_JOBS','12'))))
BOTH_SUMMARY=both.summary(BOTH_STUDY)
print('Both-Theory companion:',BOTH_ROOT)
print('n =',BOTH_SPEC['n'],'; p =',BOTH_SPEC['p'],'; gamma =',BOTH_SPEC['gamma'])
display(BOTH_SUMMARY.round(5))'''),
    md(r'''### Joint count and localization plots

Black: **WERM+CV**; purple: **WERM+P**. Both empirical curves retain the
canonical mean-to-quartile bars. Hollow green diamonds are **Theory (CV)** and
filled blue diamonds are **Theory (P)**, so both remain visible when predictions
overlap. Both are scatter markers without connecting lines
or error bars. The count reference line is three true changes.

Hausdorff distance is normalized by $n$ and conditional on a nonempty estimate;
empty estimates remain in count summaries. Thus $L=1$ has no Hausdorff value.
The plots use the same 100 datasets across bounds and methods, with all 20 final
theory realizations retained for localization.'''),
    code(r'''# Independently load the completed companion results for these plots.
import numpy as np
import matplotlib.pyplot as plt
from tools import el_matched_cv_theory as both
import tools.consistent_plotting as both_style

BOTH_ROOT,BOTH_SPEC,_=both.prepare(trials=100,theory_draws=20)
both_style.OUTPUT_DIR=BOTH_ROOT/'figures'
both_style.SOURCE_HASHES.clear();both_style.FIGURE_AUDIT.clear()
BOTH_SAVED=both_style.load_saved(BOTH_ROOT/'study.pkl')
BOTH_RECORDS=BOTH_SAVED['records']
BOTH_BOUNDS=np.arange(1,7)
BOTH_LABELS=(r'\textbf{WERM+CV}',r'\textbf{WERM+P}',
             r'\textbf{Theory (CV)}',r'\textbf{Theory (P)}')
BOTH_COLORS=('black','#AA3377','#228833','#4477AA')

def both_groups(method,metric):
    if method=='cv_theory' and metric=='count':
        return [np.array([r['cv_theory']['selected_by_bound'][L]['selected_segments']-1
                          for r in BOTH_RECORDS],float) for L in BOTH_BOUNDS]
    return [np.array([fit[metric] for r in BOTH_RECORDS for fit in
        (r['fits'][L][method] if method.endswith('_theory') else [r['fits'][L][method]])])
        for L in BOTH_BOUNDS]

for metric,ylabel in [('count','No. change points'),('haus',r'Hausdorff dist. / $n$')]:
    with plt.rc_context(both_style.CANONICAL_STYLE):
        fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
        for j,method in enumerate(('cv','penalty')):
            both_style.error_series(ax,BOTH_BOUNDS,both_groups(method,metric),
                BOTH_LABELS[j],BOTH_COLORS[j],('D','o')[j],('-','--')[j],scale=1)
        for j,method in enumerate(('cv_theory','penalty_theory'),start=2):
            values=both_groups(method,metric)
            means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in values]
            if method=='cv_theory':
                ax.scatter(BOTH_BOUNDS,means,label=BOTH_LABELS[j],marker='D',
                    facecolor='none',edgecolor=BOTH_COLORS[j],s=120,linewidth=2,alpha=.99,zorder=9)
            else:
                ax.scatter(BOTH_BOUNDS,means,label=BOTH_LABELS[j],marker='d',
                    color=BOTH_COLORS[j],s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=8)
        ax.set(xlabel=r'Upper bound $L$',ylabel=ylabel,xticks=BOTH_BOUNDS)
        if metric=='count':
            ax.axhline(3,color='gray',lw=2.5,alpha=.8,zorder=0)
            ax.set_yticks(range(6))
        handles,labels=ax.get_legend_handles_labels()
        ax.legend([handles[labels.index(label)] for label in BOTH_LABELS],BOTH_LABELS,
                  loc='best',prop={'size':14},framealpha=.4)
        fig.tight_layout();both_style.export_plot(fig,f'matched_both_theory_{metric}')
both_style.verify_sources()'''),
    md(r'''### Candidate-score agreement and numerical validity

The CV score panel displays all six candidates before upper-bound selection.
Scores are normalized absolute held-out errors, averaged across the 100
datasets. Theory uses its own limiting intervals and uniform-refit risks.
This check helps distinguish agreement of mean count curves from agreement of
the underlying risks. The table reports the minimum empirical and theoretical
segment ratios across all candidates/folds and the fixed-point residuals.
Every dataset and every candidate is retained.'''),
    code(r'''empirical_scores=np.array([r['cv']['normalized_cv_scores'] for r in BOTH_RECORDS])
theory_scores=np.array([r['cv_theory']['normalized_cv_scores'] for r in BOTH_RECORDS])
with plt.rc_context(both_style.CANONICAL_STYLE):
    fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
    both_style.error_series(ax,BOTH_BOUNDS,empirical_scores.T,
        BOTH_LABELS[0],BOTH_COLORS[0],'D','-',scale=1)
    ax.scatter(BOTH_BOUNDS,theory_scores.mean(axis=0),label=BOTH_LABELS[2],marker='D',
        facecolor='none',edgecolor=BOTH_COLORS[2],s=120,linewidth=2,alpha=.99,zorder=5)
    ax.set(xlabel=r'Candidate segments $K$',ylabel='Absolute prediction error',xticks=BOTH_BOUNDS)
    handles,labels=ax.get_legend_handles_labels()
    ordered=(BOTH_LABELS[0],BOTH_LABELS[2])
    ax.legend([handles[labels.index(label)] for label in ordered],ordered,
              loc='best',prop={'size':14},framealpha=.4)
    fig.tight_layout();both_style.export_plot(fig,'matched_both_theory_cv_scores')

import pandas as pd
from IPython.display import display
BOTH_DIAGNOSTICS=pd.DataFrame([dict(trial=r['trial'],**r['diagnostics']) for r in BOTH_RECORDS])
BOTH_DIAGNOSTICS.to_csv(BOTH_ROOT/'notebook_diagnostics.csv',index=False)
display(BOTH_DIAGNOSTICS.describe().round(6))
assert BOTH_SPEC['n']==1500 and BOTH_SPEC['p']==100 and BOTH_SPEC['gamma']==6.
np.testing.assert_array_equal(BOTH_SPEC['true_fractions'],[.25,.5,.75])
assert len(BOTH_RECORDS)==100
assert {r['trial'] for r in BOTH_RECORDS}==set(range(100))
assert len({r['data_hash'] for r in BOTH_RECORDS})==100
assert np.isfinite(empirical_scores).all() and np.isfinite(theory_scores).all()
for record in BOTH_RECORDS:
    np.testing.assert_array_equal(record['truth'],[375,750,1125])
    assert len(record['cv']['fold_results'])==len(record['cv_theory']['fold_results'])==60
    assert all(np.min(f['segment_sample_ratios'])>1 for f in record['cv']['fold_results'].values())
    assert all(np.min(f['segment_ratios'])>1 for f in record['cv_theory']['fold_results'].values())
    for L in BOTH_BOUNDS:
        fits=record['fits'][L]
        empirical_k=int(np.argmin(record['cv']['cv_scores'][:L]))+1
        theory_k=int(np.argmin(record['cv_theory']['cv_scores'][:L]))+1
        assert fits['cv']['segments']==empirical_k
        assert record['cv_theory']['selected_by_bound'][L]['selected_segments']==theory_k
        assert len(fits['cv_theory'])==len(fits['penalty_theory'])==20
        assert all(fit['segments']==theory_k for fit in fits['cv_theory'])
        for fit in [fits['cv'],fits['penalty'],*fits['cv_theory'],*fits['penalty_theory']]:
            assert fit['count']==len(fit['cp'])==fit['segments']-1
            assert 0<=fit['count']<L
            assert bool(fit['empty'])==(fit['count']==0)
assert BOTH_DIAGNOSTICS.min_empirical_segment_ratio.min()>1
assert BOTH_DIAGNOSTICS.min_theory_segment_ratio.min()>1
assert BOTH_DIAGNOSTICS.max_se_residual.max()<1e-9
both_style.verify_sources()
print('Validated all 100 datasets and 6000 empirical / 6000 theoretical CV folds.')
print('Minimum empirical segment ratio:',BOTH_DIAGNOSTICS.min_empirical_segment_ratio.min())
print('Minimum theoretical segment ratio:',BOTH_DIAGNOSTICS.min_theory_segment_ratio.min())
print('Maximum SE residual:',BOTH_DIAGNOSTICS.max_se_residual.max())
print('Figures:',BOTH_ROOT/'figures')''')]
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
