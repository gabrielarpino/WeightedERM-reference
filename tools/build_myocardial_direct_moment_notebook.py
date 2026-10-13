"""Build the full experiment and automatic top-20 report in one notebook."""
from pathlib import Path
import nbformat as nb

PATH=Path('Myocardial Infarction Experiment - direct constrained moments top 20.ipynb')
source=Path('tools/myocardial_direct_moment_experiment.py').read_text().split("if __name__=='__main__':")[0]
cells=[nb.v4.new_markdown_cell(r'''# Direct constrained moment fitting — previous top 20 and matched logistic fits

**Goal:** investigate whether fitting a valid covariance matrix directly can
improve moment agreement compared with clipped updates. No posterior-probability
or correlation target is imposed on covariance estimation.

Minimize the squared Frobenius moment mismatch
$$\|\Lambda(\rho)^T\rho\Lambda(\rho)+K(\rho)-S\|_F^2,$$
with $S=V_{aux}^TV_{aux}/n$, using
$$\rho=\begin{pmatrix}e^{u_1}&c e^{(u_1+u_2)/2}\\c e^{(u_1+u_2)/2}&e^{u_2}\end{pmatrix}.$$
Variances are bounded in [1e-10,100], correlation in [-.9999,.9999]. These are
declared numerical bounds, **not eigenvalue clipping**. Residuals use the three
symmetric entries with sqrt(2) weighting on the off-diagonal, normalized by the
observed moment's Frobenius norm. Every proposed rho recomputes its state.

Three starts: variances .003/.003 with correlation0, .03/.03 with correlation+.5,
and .3/.3 with correlation-.5. Bounded nonlinear least squares, max60 outer
function evaluations, finite-difference parameter step .001, ftol/xtol1e-7,
gtol1e-6. Choose the smallest moment residual, **not** the most attractive posterior.
Report termination, stationarity, boundary flags, all starts, and residuals.
Multiple starts are not a global-optimality certificate.

**54 rows:** the exact auxiliary/posterior fits, intercept choice, Delta and prior
of the preceding overall top20; plus matched logistic FISTA/L-BFGS/GD at
20,30,40,50 iterations and matched full-solve L2 constants
1e-8,1e-6,1e-5,4.21696503429e-5,1e-4. New variants use Delta2, both intercept choices.
Matched means identical coefficient fits in both stages. L2 acts on slopes only.
Repeated old rho-update counts no longer define different direct estimators;
all old ranks remain in before/after tables, but duplicate direct configurations
are removed before ranking. Existing fits are reused without changing them.

Same n=1692, p=110, centered/imputed/whitened design; count prior .5/.5 and
uniform conditional admissible age grid. Pilot split uses the auxiliary loss.
With intercepts, retain the existing empirical-prevalence observation-offset
model; without intercepts, retain the existing zero-offset model. Neither the
observation model nor the approximation behind the state equations is repaired
merely by changing the rho optimizer. Penalized/early-stopped states remain heuristic.

Logistic rho-search state calculations use **512 Sobol draws**, seed123, max100 inner
evaluations. Selected rho estimates are rechecked at **1,024 draws**, and posterior
evaluation also uses **1,024 draws**. Both search and checked moment residuals are
reported without reoptimizing to the finer check. Posterior likelihood quadrature160. LS auxiliary states use the
existing analytic moment expression with Gaussian quadrature. At most36 parallel
jobs, one BLAS thread each; every fit/start/state/result has its own checkpoint.

**Reporting:** all old-top20 before/after results, then the 20 unique primary
shape leaders with independent 1,024-draw/seed42 posterior checks at fixed fitted
rho. Rank those by mean conditional TV to the original curve. There is no P(1)
cutoff. Numerical differences are not confidence intervals or calibration claims.
If a fit fails, the error is retained and reported, not silently replaced.
'''),nb.v4.new_code_cell(source),
nb.v4.new_code_cell('''from IPython.display import display
display(pd.DataFrame([{k:v for k,v in s.items() if k!='old'} for s in specifications()]))
root=run()
print('Results:',root)
'''),nb.v4.new_code_cell('''from IPython.display import Markdown
summary=pd.read_csv(root/'summary.csv')
best=pd.read_csv(root/'top_twenty.csv')
starts=pd.read_csv(root/'multistart_diagnostics.csv')
errors=json.loads((root/'errors.json').read_text())
print('Recorded failures:',len(errors));display(errors if errors else 'None')
print('All original top20: old rho procedure versus direct constrained fitting')
comparison=summary[summary.origin=='previous_top20'].sort_values('old_rank')
display(comparison[['old_rank','auxiliary','posterior','intercept','Delta','old_P1','P_1',
    'old_TV','TV','rho_11','rho_12','rho_22','correlation','relative_residual','boundary','optimizer_success']])
print('Old TV is the earlier two-check mean; new TV in this table uses seed123.')
print('All multistart diagnostics (selected by moment residual only):');display(starts)
print('Top20 settings:')
display(best[['rank','id','auxiliary','posterior','intercept','Delta']].style.hide(axis='index'))
print('Top20 results:')
display(best[['rank','P1_min','P1_max','mean_TV','numerical_TV','rho_11','rho_12','rho_22',
    'correlation','moment_residual','relative_residual','relative_residual_1024','boundary','optimizer_success',
    'optimality','inner_residual','max_state_residual']].style.hide(axis='index').format(precision=6))
'''),nb.v4.new_code_cell('''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
reference=reference_distribution();a=np.array(sorted(reference));ref=np.array([reference[x] for x in a])
def align(result):
    mapping=dict(zip(result['ages'],result['conditional']))
    return np.array([mapping.get(x,0.) for x in a])
print('Previous top20 before/after: identical coefficient fits; only rho fitting changes')
fig,axes=plt.subplots(10,2,figsize=(11,24),sharex=True)
old_table=pd.read_csv(TOP)
for ax,row in zip(axes.flat,comparison.itertuples()):
    old_row=old_table[old_table['rank']==row.old_rank].iloc[0]
    old=pickle.load(open(old_row.check_checkpoint,'rb'))
    # Older checkpoints omit ages: use the actual original fit setting/grid.
    new=pickle.load(open(row.checkpoint,'rb'))
    if 'ages' not in old:old=dict(old,ages=new['ages'])
    ax.plot(a,ref,color='.7',label='Original reference')
    ax.plot(a,align(old),color='#0072B2',ls='--',label='Previous rho')
    ax.plot(a,align(new),color='black',label='Direct rho')
    ax.set_title(f'Old rank {int(row.old_rank)}: P(1) {row.old_P1:.3f} → {row.P_1:.3f}',fontsize=10)
    ax.set(xlabel='Age',ylabel='Conditional probability')
axes.flat[0].legend(fontsize=8,frameon=False);fig.tight_layout();plt.show()
'''),nb.v4.new_code_cell('''for row in best.itertuples():
    display(Markdown(f"### #{row.rank}: {row.id}\\n\\nAuxiliary: **{row.auxiliary}**; posterior: **{row.posterior}**. "
        f"Intercept: {row.intercept}; Delta={row.Delta}. Relative moment residual={row.relative_residual:.4g}; "
        f"correlation={row.correlation:.4g}; boundary={row.boundary}; optimizer success={row.optimizer_success}."))
    first=pickle.load(open(row.checkpoint,'rb'));second=pickle.load(open(row.check_checkpoint,'rb'))
    fig,(ax,cdf)=plt.subplots(1,2,figsize=(11,3.5),gridspec_kw={'width_ratios':[1.8,1]})
    for values,color,ls,label in [(ref,'.6','-','Original reference'),
        (align(first),'black','-',f"1024 draws, seed123; P(1)={first['P_1']:.4f}"),
        (align(second),'#0072B2','--',f"1024 draws, seed42; P(1)={second['P_1']:.4f}")]:
        ax.plot(a,values,color=color,ls=ls,lw=1.5,label=label)
        cdf.plot(a,np.cumsum(values),color=color,ls=ls,lw=1.5)
    ax.set(xlabel='Age',ylabel='Conditional probability');cdf.set(xlabel='Age',ylabel='Cumulative probability',ylim=(0,1.02))
    ax.legend(frameon=False,fontsize=8)
    fig.suptitle(f'#{row.rank}: shape TV={row.mean_TV:.3f}; numerical TV={row.numerical_TV:.3f}',fontsize=12)
    fig.tight_layout();plt.show()
fig,ax=plt.subplots(figsize=(7,4))
for origin,g in summary.groupby('origin'):
    ax.scatter(g.relative_residual,g.TV,label=origin,alpha=.7,s=25)
ax.set(xlabel='Relative moment residual',ylabel='Conditional TV to original')
ax.legend(frameon=False);fig.tight_layout();plt.show()
''')]
notebook=nb.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec']=dict(name='python3',language='python',display_name='Python 3')
for c in cells:
    if c.cell_type=='code':compile(c.source,str(PATH),'exec')
nb.validate(notebook);nb.write(notebook,PATH);print(PATH)
