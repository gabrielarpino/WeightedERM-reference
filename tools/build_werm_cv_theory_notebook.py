"""Build the isolated WERM-CV/SE notebook; existing experiment notebooks are untouched."""
from pathlib import Path
import nbformat as nbf


def build():
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md(r'''# WERM-CV: empirical performance and a state-evolution prediction

Only **unpenalized WERM with exhaustive interlaced 10-fold CV** is studied.
There is no adjusted estimator and no search penalty. Training, segment refits,
and final partition searches use squared loss; **only the held-out score uses
absolute error**. This reproduces the loss convention of the canonical M2 and
E_L experiments.

The user-approved theory-focused designs retain the original data-generating
mechanisms and locations, use $p=200$ and $n<3000$, and impose a minimum fitted
segment size so that every segment-refit sampling ratio is greater than one:

| Experiment | Dimension | Sample sizes | True locations | Candidate segments |
|---|---:|---|---|---|
| M2 | $p=200$ | $n\in\{1800,2200,2600\}$, $n/p\in\{9,11,13\}$ | $(0.20,0.50)n$ | $k=1,\ldots,4$ |
| E_L upper bound | $p=200$ | $n=2400$, $n/p=12$ | $(0.20,0.50,0.80)n$ | $k=1,\ldots,L$, $L=1,\ldots,6$ |

M2 uses **300 datasets per sampling ratio**; E_L uses **100 datasets**. Both use
**20 independent SE realizations** of the final full-data partition per dataset.
This notebook displays one M2 setting throughout, selected after comparing the
saved runs for its closest Hausdorff overlay and comparable mean-count agreement.
It is not best on every diagnostic. The other attempts remain archived under
`results/M2_CV_theory_convergence/`; no individual trials or ratios are omitted
from the selected run.

The empirical and theoretical predictions
share the true signal matrix and frozen sample weights; the theory never receives
empirical fitted coefficients, estimated partitions, validation scores, or selected
counts. The oracle signal-strength matrix is used only for assessing the theory.

**Interpretation.** This is a first-order SE *extension to the CV pipeline*.
It composes the paper's weighted-fit characterization with a population partition
search, a uniform-weight segment-refit risk, and a CV argmin. It is not presented
as an already-proved finite-sample theorem for CV count-selection frequencies.
Data-selected segment refits require stability, and selection requires separated
limiting risk minimizers. All empirical outcomes and numerical discrepancies
within the displayed setting are retained.'''),
    md(r'''## What the empirical CV procedure does

For each candidate segment count $k$ and held-out interlaced fold $v$:

1. Fit the $k$ weighted regressions on the ordered training observations, using
   weights conditional on exactly $k$ segments, and search for $k-1$ boundaries.
2. Map the training boundaries to the original observation indices.
3. Refit an ordinary squared-loss regression in each estimated segment using
   training observations only, with uniform within-segment sample weights.
4. Sum absolute prediction errors on the held-out observations. After all ten
   folds, divide by $n$ to report the normalized CV score.
5. Choose the candidate with the smallest score (smaller $k$ breaks exact ties),
   then rerun fixed-count WERM on all observations at that selected $k$.

Candidate scores are computed once per dataset. E_L selection at upper bound
$L$ uses the prefix $1,\ldots,L$ of that same score vector. The sampling scheme,
index mapping, and refit/scoring losses match `weighted_erm_joint_cv`. The new
minimum-size constraint is applied consistently in the empirical and theoretical
training searches and in both full-data searches. The optimized additive-loss
dynamic program is independently checked against exhaustive constrained search.

M2 retains `SparseDiffSignal(8, sqrt(400*delta), 0.3, 4)` and Gaussian noise with
variance $0.1$; the prior rescales changed coefficient entries as in the original
notebook. E_L retains its four-state Gaussian coefficient random walk and noise
standard deviation $0.5$. Both use $X_{ij}\sim N(0,1/n)$.
For a dataset with $n$ observations, set the minimum segment fraction to
$240/(0.9n)$. Every fitted segment, **including the first and last**, must contain
at least the ceiling of this fraction times the fitting sample size. Thus a
training segment has at least 240 observations ($m_s/p\ge1.2$), and a full-data
segment has at least 267 (integer ceilings are applied directly). The marginal weights are exact uniform-prior
probabilities over integer compositions satisfying these same minimum sizes.
They are frozen and shared by empirical and theoretical fits.

These constraints and weights deliberately differ from the canonical notebooks'
internal-boundary-only spacing and interpolated marginals. This is a documented
companion design for the CV theory experiment, not a rerun of the original curves.'''),
    md(r'''## SE construction and paper cross-reference

The source is `../WERM-paper/JRSSB-revision-rebuttal/JRSSB_revision.tex`: the
squared-loss equations `eq:ls-b-linear-model`–`eq:ls-kappa-linear-model`, the
fit representation `eq:ls-Theta-tilde`, the expected residual profile `eq:ls-v`,
and the population partition objective `eq:ls-F`. The CV convention is described
at `par:CV` and in `sec:sensitivity-number-of-change-points`.

Let $m$ be the training-fold size and $\rho=m/n$. To use the paper's design
normalization, set $\widetilde X=X_{\rm train}/\sqrt\rho$ and
$\widetilde B=\sqrt\rho B$. Then the predictors are unchanged,
$\delta_{\rm train}=m/p$, and
$$\Gamma_{\rm train}=\widetilde B^\top\widetilde B/m=B^\top B/n=\Gamma.$$
There is no spurious factor $n/m$ in the signal-strength matrix.

For each candidate $k$, solve the squared-loss SE using its exact training-fold
weights and the true regime labels in training order. With half-squared loss,
$$\frac1m\sum_i\frac1{1+b_\ell\pi_{i\ell}}=1-\delta_{\rm train}^{-1},
\qquad \omega_{i\ell}=\frac{b_\ell\pi_{i\ell}}{1+b_\ell\pi_{i\ell}}.$$
The paper's unhalved squared loss merely rescales $b$; it leaves the fits,
$\omega$, $\Lambda$, $K$, and all predictions below unchanged.
The true dimension is kept separate from the fitted dimension: $\Gamma$ is
$R\times R$, $\Lambda$ is $R\times k$, and $K$ is $k\times k$, with $R=3$ in M2
and $R=4$ in E_L.

The theoretical training partition minimizes the sum of expected residual profiles
over the same admissible fixed-count partitions:
$$v_{i\ell}=(1-\omega_{i\ell})^2\left[
(\lambda_\ell-e_{r_i})^\top\Gamma(\lambda_\ell-e_{r_i})
+K_{\ell\ell}+\sigma^2\right].$$
This uses the SE mean objective, rather than independently sampling noisy CV
scores and taking their argmin. The latter would silently invent a dependence
structure between candidate scores.'''),
    md(r'''### Uniform-weight segment refits and absolute held-out risk

For each theoretically predicted segment $s$, let $m_s$ be its training size,
$\delta_s=m_s/p$, and $q_s$ its vector of true-regime proportions among training
rows. Uniform-weight squared-loss SE has the closed form
$$\lambda_s=q_s,\qquad
\kappa_s=\frac{\sigma^2+\sum_r q_{sr}\Gamma_{rr}-q_s^\top\Gamma q_s}
{\delta_s-1},\qquad \delta_s>1.$$
The numerator includes the effect of mixing true regimes inside a fitted segment.
For a held-out observation from true regime $r$ assigned to segment $s$,
$$R_{sr}^{\rm abs}=\sqrt{\frac2\pi}\,
\sqrt{\sigma^2+(q_s-e_r)^\top\Gamma(q_s-e_r)+\kappa_s}.$$
Average these risks over the actual held-out regime/segment overlaps and then
over folds. The resulting theoretical score vector selects its own count.
The absolute loss is evaluated as an observable of the squared-loss refit;
there is no absolute-loss regression or absolute-loss fixed-point solve.

At the theory-selected count, recompute SE at the **full-data** sampling ratio
and weights. Draw independent $Z_i\sim N(0,\Gamma)$, $G_i\sim N(0,K)$ and noise,
construct the proximal fitted values from `eq:ls-Theta-tilde`, and apply the same
fixed-count partition search. These 20 draws describe the final localization
prediction conditional on the first-order theory-selected count.

For fixed candidate and fold counts, a finite union bound combines concentration
of the fold scores without requiring fold independence. A separated limiting
minimum then transfers it to count selection. This construction still assumes
stability of the training partition and its data-selected segment refits; those
steps are not supplied automatically by a theorem for fixed weights. The
minimum-size prior also creates zero-weight regions, whereas the paper's stated
regularity assumption requires weights positive almost everywhere. A rigorous
extension must relax that condition using sufficient positive support and
curvature; numerical root/support diagnostics do not replace this justification.
Near-tied
risks and unsupported $\delta_s\le1$ must be reported, not hidden by dropping a
candidate or substituting a different refit model.'''),
    code('''# Run from WeightedERM-reference. Load the selected completed runs; no simulation is rerun.
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import tools.consistent_plotting as style
ROOT=Path('results/E-WERM-CV-theory/presentation_m2_p200_trials300')
ROOT.mkdir(parents=True,exist_ok=True)
INPUTS={
    'M2':Path('results/M2_CV_theory_convergence/p200_6961bbb9765c/results.pkl'),
    'EL':Path('results/E-WERM-CV-theory/ed5fc8b279bb/results.pkl'),
}
EXPECTED_HASHES={
    'M2':'36cf40a584f19fa5939bc1e92ffb6de24070e9c17c42f688d300c3655cc8fffd',
    'EL':'8ac99031d1af563a2565307af0c00529b5f4607f08e284703bb708c62cc1c2c7',
}
style.SOURCE_HASHES.clear();style.FIGURE_AUDIT.clear()
SOURCES={name:style.load_saved(path) for name,path in INPUTS.items()}
assert all(style.SOURCE_HASHES[str(INPUTS[name])]==digest
           for name,digest in EXPECTED_HASHES.items())
SPEC=dict(settings=SOURCES['M2']['specification']['settings']+
    [s for s in SOURCES['EL']['specification']['settings'] if s['experiment']=='EL'],
    trials_by_experiment={'M2':300,'EL':100},theory_draws=20)
RECORDS=[r for name,payload in SOURCES.items() for r in payload['records']
         if r['experiment']==name]
(ROOT/'presentation_manifest.json').write_text(json.dumps(dict(
    specification=SPEC,input_hashes=style.SOURCE_HASHES,
    selection='M2 selected after comparison of saved runs; all trials and ratios retained'),indent=2))
print('Results:',ROOT)
print('Trials per setting:',SPEC['trials_by_experiment'])
print('Theory draws per final fit:',SPEC['theory_draws'])
DESIGN=pd.DataFrame(SPEC['settings'])
DESIGN['trials']=DESIGN['experiment'].map(SPEC['trials_by_experiment'])
display(DESIGN[['experiment','n','p','delta','trials','max_segments','minimum_segment_fraction']])'''),
    md('''## Tables and plotting conventions

WERM-CV is black. SE predictions are blue diamond scatter markers (`d`) with
white edges, without connecting lines or error bars, following the SE estimation
plotting notebook. Empirical bars show absolute distances from the mean to the
first and third quartiles; they are not confidence intervals. The canvas, fonts,
axis labels, and marker sizes follow the canonical M1/M2/M3 style.

Hausdorff distance is divided by $n$ and is reported conditional on a nonempty
estimate. Count summaries retain empty estimates. Theory count predictions have
one selected count per signal matrix, not 20 independent CV selections. The 20
final SE draws contribute to localization only.'''),
    code('''import tools.consistent_plotting as style
import matplotlib.pyplot as plt
style.OUTPUT_DIR=ROOT/'figures'
style.FIGURE_AUDIT.clear()
LABELS=(r'\\textbf{WERM-CV}',r'\\textbf{Theory}')
COLORS=('black','#4477AA')

def selection(record,source,bound):
    return record[source]['selected_by_bound'][bound]

def outcomes(record,source,bound):
    selected=selection(record,source,bound)
    if source=='empirical':
        return [selected]
    return selected.get('final_fits',[])

def groups(records,source,bound,metric):
    if metric=='count':
        return np.array([selection(r,source,bound)['selected_segments']-1 for r in records],float)
    return np.array([f[metric] for r in records for f in outcomes(r,source,bound)],float)

def settings(name):
    subset=[r for r in RECORDS if r['experiment']==name]
    if name=='M2':
        return [(float(d),[r for r in subset if r['delta']==d],4)
                for d in sorted(set(r['delta'] for r in subset))]
    return [(L,subset,L) for L in range(1,7)]

summary_rows=[]
for name in ('M2','EL'):
    for x,records,bound in settings(name):
        true_count=len(records[0]['truth'])
        for source in ('empirical','theory'):
            counts=groups(records,source,bound,'count')
            haus=groups(records,source,bound,'haus');finite=haus[np.isfinite(haus)]
            summary_rows.append(dict(experiment=name,x=x,source=source,trials=len(records),
                mean_count=counts.mean(),correct_count=np.mean(counts==true_count),
                empty=int(np.sum(counts==0)),mean_haus=finite.mean() if len(finite) else np.nan,
                localization_draws=len(haus),finite_localization_draws=len(finite)))
SUMMARY=pd.DataFrame(summary_rows)
SUMMARY.to_csv(ROOT/'notebook_summary.csv',index=False)
display(SUMMARY.round(5))

def show_match(name):
    panels=settings(name);x=[s[0] for s in panels]
    for metric,ylabel in [('count','No. change points'),('haus',r'Hausdorff dist. / $n$')]:
        with plt.rc_context(style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            empirical=[groups(records,'empirical',bound,metric) for _,records,bound in panels]
            theory=[groups(records,'theory',bound,metric) for _,records,bound in panels]
            style.error_series(ax,x,empirical,LABELS[0],COLORS[0],'D','-',scale=1)
            means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in theory]
            ax.scatter(x,means,label=LABELS[1],marker='d',color=COLORS[1],s=120,
                       edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
            ax.set(xlabel=r'$\\delta=n/p$' if name=='M2' else r'Upper bound $L$',
                   ylabel=ylabel,xticks=x)
            if metric=='count':
                ax.axhline(len(panels[0][1][0]['truth']),color='gray',alpha=.8,lw=2.5,zorder=0)
                ax.set_yticks(range(4 if name=='M2' else 6))
            handles,labels=ax.get_legend_handles_labels()
            ax.legend([handles[labels.index(l)] for l in LABELS],LABELS,
                      loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();style.export_plot(fig,name+'_'+metric+'_match')

show_match('M2')
show_match('EL')'''),
    md('''## Candidate-score agreement and count-selection stability

Agreement of selected counts can hide differences in validation risks. These
panels compare the normalized absolute-error CV score for every candidate count.
The theory constructs each score independently of the observed CV scores.
For E_L the same six scores determine all six upper-bound selections.

The diagnostic table reports agreement of the empirical and theory-selected
counts, the mean absolute candidate-score discrepancy, and the gaps between
the smallest and second-smallest scores. Small gaps indicate sensitivity to
finite-sample fluctuations; matching first-order risk curves does not by itself
prove a match of finite-sample selection probabilities.'''),
    code('''score_rows=[]
for name in ('M2','EL'):
    slices=settings(name) if name=='M2' else [settings(name)[-1]]
    for x,records,bound in slices:
        empirical=np.array([r['empirical']['normalized_cv_scores'] for r in records])
        theory=np.array([r['theory']['normalized_cv_scores'] for r in records])
        counts=np.arange(1,empirical.shape[1]+1)
        emp_gap=np.diff(np.sort(empirical,axis=1)[:,:2],axis=1).ravel()
        se_gap=np.diff(np.sort(theory,axis=1)[:,:2],axis=1).ravel()
        conditional=np.array([r['conditional_validation_risk'] for r in records])
        refit_se=np.array([r['refit_se_at_empirical_partitions'] for r in records])
        score_rows.append(dict(experiment=name,x=x,
            count_agreement=np.mean(np.argmin(empirical,axis=1)==np.argmin(theory,axis=1)),
            mean_absolute_score_discrepancy=np.mean(np.abs(empirical-theory)),
            observed_vs_conditional_mae=np.mean(np.abs(empirical-conditional)),
            conditional_vs_refit_se_mae=np.mean(np.abs(conditional-refit_se)),
            empirical_vs_theory_partition_risk_mae=np.mean(np.abs(refit_se-theory)),
            median_empirical_gap=np.median(emp_gap),minimum_empirical_gap=emp_gap.min(),
            median_theory_gap=np.median(se_gap),minimum_theory_gap=se_gap.min()))
        with plt.rc_context(style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            style.error_series(ax,counts,empirical.T,LABELS[0],COLORS[0],'D','-',scale=1)
            ax.scatter(counts,theory.mean(axis=0),label=LABELS[1],marker='d',color=COLORS[1],
                       s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
            ax.set(xlabel=r'Candidate segments $k$',ylabel='Absolute prediction error',xticks=counts)
            if name=='M2':ax.set_title(rf'$\\delta={x:g}$',fontsize=14)
            handles,labels=ax.get_legend_handles_labels()
            ax.legend([handles[labels.index(l)] for l in LABELS],LABELS,
                      loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();style.export_plot(fig,f'{name}_scores_{x:g}')
SCORE_DIAGNOSTICS=pd.DataFrame(score_rows)
SCORE_DIAGNOSTICS.to_csv(ROOT/'notebook_score_diagnostics.csv',index=False)
display(SCORE_DIAGNOSTICS.round(6))'''),
    md(r'''The extra risk diagnostics separate three sources of discrepancy:

- `observed_vs_conditional_mae`: observed validation error versus its exact
  conditional Gaussian expectation using the actual refitted coefficients.
- `conditional_vs_refit_se_mae`: that conditional risk versus uniform-refit SE
  evaluated at the **empirical** intervals.
- `empirical_vs_theory_partition_risk_mae`: refit SE at empirical intervals versus
  the full prediction using the SE-derived intervals.

These are diagnostic mean absolute discrepancies, not an additive decomposition
of error. The empirical intervals and coefficients enter **only these diagnostics**;
they never enter the main theory score, its count selection, or its final partition.
This distinguishes validation noise, refit-risk error, and partition-related error
when the overall match is imperfect.'''),
    md('''## Numerical validity and provenance

The local-ratio audit includes the endpoint segments and all CV candidates,
not only the ultimately selected candidate. Uniform-refit SE requires $m_s/p>1$.
The theory's interval-selection and refit-stability assumptions remain separate
from these numerical checks. Source hashes and frozen marginal weights are stored
with the checkpoints; plotting verifies that its saved input file is unchanged.'''),
    code('''diagnostic_rows=[dict(experiment=r['experiment'],delta=r['delta'],trial=r['trial'],
                      **r.get('diagnostics',{})) for r in RECORDS]
DIAGNOSTICS=pd.DataFrame(diagnostic_rows)
DIAGNOSTICS.to_csv(ROOT/'notebook_trial_diagnostics.csv',index=False)
display(DIAGNOSTICS.groupby(['experiment','delta']).agg(['min','max']))
assert len(RECORDS)==sum(SPEC['trials_by_experiment'][s['experiment']] for s in SPEC['settings'])
for i,s in enumerate(SPEC['settings']):
    selected=[r for r in RECORDS if r['setting']==i]
    assert {r['trial'] for r in selected}==set(range(SPEC['trials_by_experiment'][s['experiment']]))
    assert all(r['experiment']==s['experiment'] and r['p']==s['p'] and r['n']==s['n'] for r in selected)
assert len({(r['setting'],r['trial']) for r in RECORDS})==len(RECORDS)
assert all(r['p']>=200 and r['n']<3000 for r in RECORDS)
assert DIAGNOSTICS.min_empirical_segment_ratio.min()>1
assert DIAGNOSTICS.min_theory_segment_ratio.min()>1
assert DIAGNOSTICS.max_se_residual.max()<1e-9
assert all(np.isfinite(r[s]['normalized_cv_scores']).all() for r in RECORDS for s in ('empirical','theory'))
style.verify_sources()
print('Implementation: tools/werm_cv_theory_experiment.py and tools/werm_cv_empirical.py')
print('Validated 900 M2 and 100 E_L datasets. Source experiment artifacts are unchanged.')''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'werm_cv_presentation':{'m2':'p200_trials300','archived_attempts':True},
        'kernelspec':{'name':'python3','display_name':'Python (weighted-erm)','language':'python'}})
    path=Path('E-WERM-CV-theory.ipynb')
    if path.exists():
        old=nbf.read(path,as_version=4)
        for i,cell in enumerate(notebook.cells):
            if i<len(old.cells) and cell.source==old.cells[i].source:
                notebook.cells[i]=old.cells[i]
    nbf.write(notebook,path)
    return path


if __name__=='__main__':
    print(build())
