"""Append the two SE studies, preserving every pre-existing notebook cell."""
import json
from pathlib import Path
import nbformat as nbf

TAG = 'el-se-penalty-fixed-v1'


def append():
    path = Path('E_L_upper_bound.ipynb')
    before = path.read_bytes()
    notebook = json.loads(before)
    original = [c for c in notebook['cells'] if TAG not in c.get('metadata',{}).get('tags',[])]
    backup = Path('results/E_L_SE_penalty_fixed/notebook_before_se.ipynb')
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():
        backup.write_bytes(before)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r'''## Search-penalty count selection: empirical WERM and state evolution

This additional study retains the existing **$n=1000$, $p=100$, 100 paired trials**,
Gaussian data generator, base seed 20261120, noise standard deviation 0.5,
and three true changes at $(0.2,0.5,0.8)n$ ($L^\star=4$).
The CV results above remain a separate experiment.

For each upper bound $L\in\{1,\ldots,6\}$, fit the $L$ unpenalized weighted
least-squares regressions **once**, with upper-bound occupancy weights from a
uniform prior on segment counts $1,\ldots,L$ and a conditional uniform location prior.
Then minimize
$$\sum_{i=1}^n \frac12(y_i-\hat\theta_{i,\psi_i})^2
+6\log\binom{n-1}{K-1},\qquad 1\le K\le L.$$
This is the exact search penalty from `E_search_penalty.ipynb`, with **$\gamma=6$**
as requested, matching that notebook's frozen M3 multiplier. It is used here
without calibration against this study's CV or true counts. Coefficients are
not refitted at each candidate $K$; the search uses the first $K$ of the $L$ fits.
The internal-boundary separation is $n/10$; as in the native DP, end segments need
only be nonempty. There is **no CV and no adjusted-WERM search** in these additions.

Native marginal interpolation uses the same 50-point reference grid (gap 5),
with seed 0 and frozen weights shared by empirical WERM and theory.
The empirical data exactly reproduce the existing study's data generator.
All original cells and outputs above are preserved.'''),
    md(r'''### SE equations and the distinction between true and fitted dimensions

Use the squared-loss specialization in `JRSSB_revision.tex`,
`eq:ls-b-linear-model`–`eq:ls-kappa-linear-model`, equivalently the general
fixed-point equations `eq:b-fp-implicit`–`eq:cross-cov-fp-implicit`.
For **half-squared** loss, solve
$$\frac1n\sum_i(1+b_\ell\pi_i^{(\ell)})^{-1}=1-\delta^{-1},
\qquad \omega_{i\ell}=\frac{b_\ell\pi_i^{(\ell)}}{1+b_\ell\pi_i^{(\ell)}}.$$
The factor 2 in the paper's unhalved squared loss is absorbed into $b$; the resulting
$\omega$, bias matrix and covariance are unchanged by this loss rescaling.
Keep all four true regimes, even when $L<4$: $\Gamma=B^\top B/n$ is $4\times4$,
$\Lambda$ is $4\times L$, and $K$ is $L\times L$. With $e_r\in\mathbb R^4$,
$$\lambda_\ell=\frac\delta n\sum_i\omega_{i\ell}e_{\Psi_i},\qquad
s_{\ell m}=\frac\delta n\sum_i\omega_{i\ell}\omega_{im},$$
$$K_{\ell m}=\frac{\frac\delta n\sum_i\omega_{i\ell}\omega_{im}
(\lambda_\ell-e_{\Psi_i})^\top\Gamma(\lambda_m-e_{\Psi_i})+s_{\ell m}\sigma^2}
{1-s_{\ell m}}.$$
These finite-sum equations compute the SE parameters directly after scalar root solves;
they do not use a low-dimensional population regression in place of SE.
The rectangular form retains the original four-state response law rather than
discarding true regimes when fitting fewer columns. This is a numerical study of
that extension, not a new proof of a misspecification theorem.

Independently of the empirical design and noise, draw $Z_i\sim N(0,\Gamma)$,
$G_i\sim N(0,K)$, and $\varepsilon_i^{\rm SE}\sim N(0,0.5^2)$; set
$$y_i^{\rm SE}=Z_{i,\Psi_i}+\varepsilon_i^{\rm SE},\qquad
\theta_{i\ell}^{\rm SE}=(1-\omega_{i\ell})(Z_i^\top\lambda_\ell+G_{i\ell})
+\omega_{i\ell}y_i^{\rm SE}.$$
Apply **the identical penalized or fixed-count partition search** to these proximal
fits and synthetic responses. This is the fitted-$\Theta$ representation in the
paper, not its adjusted-$\Theta$ representation.

Theory uses the oracle signal-strength matrix only to evaluate predictions;
the empirical estimator never receives it, the true count, or the true locations.
There are 20 independent SE draws for each of the 100 signal matrices, giving
2,000 theory realizations per setting. No empirical estimate or selected count
is supplied to the theory search. These are finite-size SE predictions; agreement
is assessed from the output, not enforced.'''),
    code('''# Self-contained setup: run here without executing any original CV cells.
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from tools.el_state_evolution import prepare, run, summary, MODES
SE_ROOT, SE_SPEC, SE_WEIGHTS = prepare(trials=100,theory_draws=20,penalty_weight=6.)
print('SE experiment directory:',SE_ROOT)
# Shared trial job computes both sections; validated checkpoints are reused.
SE_STUDY = run(SE_ROOT,SE_SPEC,SE_WEIGHTS,jobs=int(os.environ.get('EL_SE_JOBS','12')))
SE_SUMMARY = summary(SE_STUDY)
display(SE_SUMMARY[SE_SUMMARY['mode']=='penalty'].round(4))'''),
    md('''### Plots: search-penalty upper-bound study

Black curves: empirical WERM; blue diamond scatter markers: SE mean predictions,
following `SE match estimation - plotting.ipynb` (marker `d`, white edges).
SE markers have no connecting lines or error bars. Mean panels use the canonical
final-M1 5×4-inch canvas, 18 pt labels and 14 pt ticks. Empirical bars show means
with absolute distances to the first/third quartiles across 100 trials, not confidence
intervals. SE means use all 2,000 realizations.
Boxplots use 1.5-IQR whiskers and deterministic faint jitter (SE jitter
is thinned to 100 points for legibility; its box statistics use all draws).

Hausdorff distance is normalized by $n$ and conditional on a nonempty estimate;
the summary retains empty rates. In particular $L=1$ has no Hausdorff value.
Histograms use the first independent SE realization from each of the 100 signal
matrices, matching the empirical number of realizations. This choice is fixed by
draw index, not outcomes; all 2,000 draws still enter line and boxplot statistics.
Histograms use canonical 12-bin densities, five-fold bandwidth-selected Gaussian
KDE, truth ticks and the fixed 0–18 density display range. Clipped peaks are reported.
The x-axis below is the **upper bound on segments**, not the selected count.'''),
    code('''import tools.consistent_plotting as se_style
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
se_style.OUTPUT_DIR=SE_ROOT/'figures'
se_style.SOURCE_HASHES.clear();se_style.FIGURE_AUDIT.clear()
SE_LABELS=(r'\\textbf{WERM}',r'\\textbf{Theory}')
SE_COLORS=('black','#4477AA')

def se_groups(payload,mode,source,metric):
    output=[]
    for count in MODES[mode]:
        settings=[r['fits'][mode,count] for r in payload['records']]
        fits=([s['empirical'] for s in settings] if source=='empirical'
              else [fit for s in settings for fit in s['theory']])
        output.append(np.array([f[metric] for f in fits]))
    return output

def plot_el_se(mode):
    saved=se_style.load_saved(SE_ROOT/'study.pkl')
    counts=np.array(MODES[mode]);x=counts if mode=='penalty' else counts-1
    xlabel=r'Upper bound $L$' if mode=='penalty' else 'Assumed number of change points'
    for metric,ylabel in [('haus',r'Hausdorff dist. / $n$'),('count','No. change points')]:
        for kind in ('line','box'):
            with plt.rc_context(se_style.CANONICAL_STYLE if kind=='line' else se_style.PAPER_STYLE):
                if kind=='line':
                    fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
                else:
                    fig,ax=se_style.paper_panel();rng=np.random.default_rng(42)
                for j,source in enumerate(('empirical','theory')):
                    groups=se_groups(saved,mode,source,metric)
                    if kind=='line':
                        if source=='theory':
                            means=np.array([np.mean(v[np.isfinite(v)])
                                if np.isfinite(v).any() else np.nan for v in groups])
                            ax.scatter(x,means,label=SE_LABELS[j],marker='d',
                                color=SE_COLORS[j],s=120,edgecolor='white',
                                linewidth=2.5,alpha=.99,zorder=5)
                        else:
                            se_style.error_series(ax,x,groups,SE_LABELS[j],SE_COLORS[j],
                                'D','-',scale=1)
                    else:
                        for i,values in enumerate(groups):
                            previous=len(ax.collections)
                            se_style.box_group(ax,values,3*i+j,SE_COLORS[j],rng)
                            # Box statistics retain all SE draws; show a comparable
                            # number of raw points for each method in the scatter.
                            if source=='theory':
                                for points in ax.collections[previous:]:
                                    offsets=points.get_offsets()
                                    if len(offsets)>100:
                                        points.set_offsets(offsets[np.linspace(0,len(offsets)-1,100,dtype=int)])
                if kind=='line':
                    ax.set_xticks(x)
                else:
                    ax.set_xticks(np.arange(len(x))*3+.5,[str(v) for v in x])
                ax.set(xlabel=xlabel,ylabel=ylabel)
                if metric=='count':
                    ax.axhline(3,color='gray',lw=2.5 if kind=='line' else 1.4,alpha=.8)
                    ax.set_yticks(range(6))
                if kind=='line':
                    ax.legend(loc='best',prop={'size':14},framealpha=.4)
                    fig.tight_layout()
                else:
                    ax.legend(handles=[Patch(facecolor=c,alpha=.28,label=l) for c,l in zip(SE_COLORS,SE_LABELS)],
                              loc='best',fontsize=7)
                se_style.export_plot(fig,f'{mode}_{metric}_{kind}')
    pools=[]
    for count in counts:
        settings=[r['fits'][mode,int(count)] for r in saved['records']]
        empirical=np.concatenate([s['empirical']['cp']/SE_SPEC['n'] for s in settings])
        theory=np.concatenate([s['theory'][0]['cp']/SE_SPEC['n'] for s in settings])
        pools.append([empirical,theory])
    row_labels=([rf'$L={c}$' for c in counts] if mode=='penalty'
                else [rf'$K-1={c-1}$' for c in counts])
    se_style.histogram_grid(pools,SE_LABELS,SE_COLORS,[.2,.5,.8],f'{mode}_histograms',
        row_labels=row_labels,density_label_x=.005)
    se_style.verify_sources()

plot_el_se('penalty')'''),
    md(r'''## Fixed assumed count: under- and over-specification with SE

Keep $L^\star=4$ (three true changes) and the same 100 datasets, but now assume
**exactly 1, 2, 3, 4 or 5 change points**: fitted segment counts $K=2,\ldots,6$.
Each setting has its own fixed-count prior weights and coefficient fits. The search
is constrained to exactly $K$ segments, with **no search penalty and no CV**.
The SE calculation uses those same weights and applies that same exact-count search.

The x-axis is now the **assumed number of change points**; it is not an upper bound.
Hausdorff error and location distributions provide the nontrivial theory comparison.
The count plot is deterministic by construction and is included only as a protocol check.
All four true regimes are retained for the underspecified settings.'''),
    code("display(SE_SUMMARY[SE_SUMMARY['mode']=='fixed'].round(4))\nplot_el_se('fixed')"),
    md('''### Agreement and solver diagnostics

The table reports SE minus empirical differences in mean count and conditional
mean normalized Hausdorff error. Empty estimates are kept in count summaries and
omitted only from conditional localization and location densities. The two sets
of draws are independent conditional on each trial's true signal matrix; 2,000
SE realizations are not 2,000 independent draws of that matrix.

Residuals check the scalar curvature equations, Gaussian score orthogonality and
the full covariance fixed-point equation. Tests also independently evaluate the
general moment equations by Monte Carlo, verify classical homogeneous least
squares, and compare the search against the native exact DP.'''),
    code('''agreement=[]
for mode,counts in MODES.items():
    for count in counts:
        frame=SE_SUMMARY[(SE_SUMMARY['mode']==mode)&(SE_SUMMARY.fitted_segments==count)].set_index('source')
        agreement.append(dict(mode=mode,x=count if mode=='penalty' else count-1,
            empirical_haus=frame.loc['empirical','mean_haus'],
            theory_haus=frame.loc['theory','mean_haus'],
            haus_difference=frame.loc['theory','mean_haus']-frame.loc['empirical','mean_haus'],
            empirical_count=frame.loc['empirical','mean_count'],
            theory_count=frame.loc['theory','mean_count'],
            count_difference=frame.loc['theory','mean_count']-frame.loc['empirical','mean_count']))
agreement=pd.DataFrame(agreement)
agreement.to_csv(SE_ROOT/'agreement.csv',index=False)
display(agreement.round(4))
from IPython.display import Markdown
penalty_rows=SE_SUMMARY[SE_SUMMARY['mode']=='penalty']
if np.allclose(penalty_rows['mean_count'],penalty_rows['fitted_segments']-1):
    display(Markdown(f"**Observed count-selection limitation:** at gamma={SE_SPEC['penalty_weight']:g}, "
        'every empirical and SE search selected the maximum allowed count. '
        'The count overlay is therefore exact but supplies only a limited count-selection check. '
        'Gamma was not retuned after seeing these results.'))
display(Markdown('Maximum absolute empirical–SE difference in mean normalized Hausdorff error: '
                 f"**{agreement.haus_difference.abs().max():.4f}**. These are finite-size Monte Carlo results."))
states=[fit['se'] for row in SE_STUDY['records'] for fit in row['fits'].values()]
diagnostics={key:max(s[key] for s in states)
             for key in ('b_residual','bias_residual','covariance_residual')}
diagnostics['minimum_covariance_eigenvalue']=min(s['min_eigenvalue'] for s in states)
display(pd.Series(diagnostics,name='SE diagnostics'))
assert len(SE_STUDY['records'])==100
assert all(len(f['theory'])==20 for r in SE_STUDY['records'] for f in r['fits'].values())
assert max(diagnostics[k] for k in ('b_residual','bias_residual','covariance_residual'))<1e-9
se_style.verify_sources()
print('Completed 100 empirical trials and 2,000 SE realizations at each setting.')'''),
    md('''Implementation: `tools/el_state_evolution.py`; validation:
`tests/test_el_state_evolution.py`. Source hashes, frozen weights, data hashes,
per-trial SE matrices/residuals, empirical estimates, independent theoretical
estimates and all candidate penalized search scores are saved under the printed
result directory. Re-running these new sections reuses matching checkpoints.
Original CV cells and results are unchanged.''')]
    for cell in cells:
        cell['metadata']['tags']=[TAG]
    notebook['cells']=original+cells
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    written=json.loads(path.read_text())
    assert written['cells'][:len(original)]==original
    return path,len(original)


if __name__=='__main__':
    print(append())
