"""Visible-code intercept sweep with automatic overall and matched top tens."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - finite intercept best ones.ipynb')


def build():
    intro=r'''# Finite-intercept experiment and best ones

This new study enables an **unpenalized intercept in both stages**. The preferred
class uses logistic loss, the identical optimizer and iteration budget, and the
same fitted coefficient vectors for auxiliary and posterior-input construction.
The permitted alternative uses exact weighted least squares **with an intercept**
for the auxiliary fit, followed by finite-iteration logistic with an intercept.
No L1 or L2 fitting penalties are used anywhere in this sweep.

Optimizers: monotone FISTA, deterministic full-batch gradient descent, and L-BFGS-B.
Budgets: 1, 3, 10, 30, 100 updates. Each fit starts with zero slopes and the weighted
intercept-only logistic MLE. The constant column is $1/\sqrt n$, so its norm matches
the whitened feature columns; its fitted coefficient is converted back to an
ordinary intercept. This preconditioning is declared because early stopping depends
on parameterization. FISTA/GD use the reciprocal exact global logistic curvature
bound; L-BFGS uses memory 10, at most 20 line-search steps, and zero convergence
tolerances. Actual iteration counts and intercept score residuals are recorded.

Minimum segment sizes are $\Delta=1,2,5$. Rho updates are 1, 3, 10, 20, 30, with
start $I_2$, damping 0.25 and eigenvalue clipping to $[10^{-4},100]$. LS also includes
the direct approximate moment fit. Additional starts $0.1I_2$ and $10I_2$ are tested
at Delta=1, fitting budgets 10/30, all three optimizers, and 3/10/30 rho updates.
The design contains **531 primary posterior comparisons** when all fits/states
are numerically evaluable. Failed settings are recorded, not silently discarded.

Data/preprocessing remain 1,692 age-sorted observations, 110 mean-imputed centered
predictors, covariance whitening with $10^{-6}I$ and division by $\sqrt n$.
Prior odds for zero/one change remain one-half each; conditional location is uniform
over admissible distinct-age splits. The pilot segmentation is estimated from the
auxiliary loss and held fixed during rho updates. Posterior locations are not
restricted to the pilot. The logistic posterior uses the existing intercept-aware,
global-prevalence-calibrated observation model. This is an **empirical-Bayes heuristic
extension**, not the original zero-offset theorem. The limiting state equations
are not exact finite-iteration theory; adjusted-score means, intercept scores, and
moment residuals are therefore included as diagnostics.

Primary states: 256 Sobol draws/seed 42, at most 60 evaluations, 160-node quadrature.
The top ten shapes in each auxiliary class, plus nearby shape/probability trade-offs,
receive checks with 512 draws/seed 42 and 1,024 draws/seed 123, at most 100 evaluations.
Rank by mean TV to the full archived age distribution; no P(1)>0.5 eligibility
threshold. Outcome-guided ranking is exploratory and the numerical ranges are not
confidence intervals. Existing no-intercept results remain unchanged.
'''
    source=Path('tools/myocardial_finite_intercept_sweep.py').read_text().split("if __name__=='__main__':")[0]
    execute='''root=run_sweep()
summary=pd.read_csv(root/'summary.csv')
ranked=pd.read_csv(root/'ranked.csv')
failures=json.loads((root/'errors.json').read_text())
print('Primary comparisons:',len(summary),'recorded failures:',len(failures))
if failures:display(pd.DataFrame(failures))
columns=['group','method','fit_budget','Delta','rho_updates','initial_scale','P1_min','P1_max',
         'mean_TV','numerical_curve_TV','mass_above55_min','peak_min','peak_max',
         'relative_moment_residual','posterior_adjusted_mean_norm','intercept_score_max']
print('Top ten overall');display(ranked.head(10)[columns])
print('Top ten matched logistic/logistic');display(ranked[ranked.group=='matched_logistic'].head(10)[columns])
'''
    plots=r'''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'axes.labelsize':12,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
reference=reference_distribution();a=np.array(sorted(reference));p=np.array([reference[x] for x in a])
def draw(ax,row):
    ax.plot(a,p,color='0.6',lw=1.4,label='Archived conditional posterior')
    for tag,color,label in [('q9_seed42','black','512 draws'),('q10_seed123','#4178a8','1,024 draws')]:
        rec=pickle.load((root/(row.case+'__'+row.posterior_fit+'__'+tag+'.pkl')).open('rb'))
        mapping=dict(zip(rec['ages'],rec['conditional']))
        ax.plot(a,[mapping.get(x,0.) for x in a],color=color,lw=1.4,label=label)
    ax.axvline(55,color='0.6',ls=':',lw=.8);ax.set_xlabel('Age');ax.set_ylabel('Probability')
    ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.yaxis.set_major_locator(MaxNLocator(3))
for label,table in [('Overall',ranked.head(10)),('Matched logistic/logistic',ranked[ranked.group=='matched_logistic'].head(10))]:
    print(label)
    fig,axes=plt.subplots(5,2,figsize=(10,12.5),sharex=True)
    for i,(ax,row) in enumerate(zip(axes.flat,table.itertuples()),1):
        draw(ax,row);ax.set_title(f'#{i}: TV {row.mean_TV:.3f}; P(1) {row.P1_min:.3f}–{row.P1_max:.3f}',fontsize=10)
    for ax in list(axes.flat)[len(table):]:ax.set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.97));plt.show()
    for i,row in enumerate(table.itertuples(),1):
        print(f'{label} #{i}: {row.group}, {row.method}, {row.fit_budget} fit updates; '
              f'Delta={row.Delta}, rho updates={row.rho_updates}, initial scale={row.initial_scale}; '
              f'pilot age={row.pilot_age:g}; P(1)={row.P1_min:.4g}–{row.P1_max:.4g}; '
              f'mean TV={row.mean_TV:.4g}; relative moment residual={row.relative_moment_residual:.4g}.')
        fig,ax=plt.subplots(figsize=(6,2.8));draw(ax,row);ax.legend(frameon=False,fontsize=8)
        fig.tight_layout();plt.show()
'''
    overview=r'''print('All primary results: shape/probability trade-off, not a selection threshold')
fig,axes=plt.subplots(1,2,figsize=(10,3.2),sharex=True,sharey=True)
for ax,group in zip(axes,('matched_logistic','LS_auxiliary')):
    data=summary[summary.group==group]
    for method,color in [('fista','black'),('lbfgs','#c69c22'),('gd','#4178a8')]:
        sub=data[data.method==method];ax.scatter(sub.TV_to_reference,sub.P_1,s=12,alpha=.5,color=color,label=method)
    ax.axhline(.5,color='0.6',ls=':',lw=.8);ax.set_xlabel('TV from archived curve')
    ax.set_title(group.replace('_',' '),fontsize=11);ax.yaxis.set_major_locator(MaxNLocator(3))
axes[0].set_ylabel('P(1 chgpt)');axes[0].legend(frameon=False,fontsize=9)
fig.tight_layout();plt.show()
print('All fits have matched intercept=True; no explicit fitting penalties.')
print('Numerical checks assess integration sensitivity, not independent-data validation.')
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(plots),nbf.v4.new_code_cell(overview)])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'notebook_cell','exec')
    nbf.write(nb,PATH);print(PATH)


if __name__=='__main__':build()
