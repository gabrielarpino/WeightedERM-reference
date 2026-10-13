"""Embed the complete reproducible exploration and its plots in a new notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'Myocardial Infarction Experiment - heuristic exploration.ipynb'
source=(ROOT/'tools/myocardial_exploration.py').read_text().split("if __name__=='__main__':")[0]
cells=[nbf.v4.new_markdown_cell(r'''# Myocardial infarction: exploratory approximate signal-strength analysis

This notebook explores **heuristic sensitivity**, not confirmatory selection of a prior or a proof of convergence.
**Updated target:** $P(K=1)>1/2$ and $P(\tau>55\mid K=1)>1/2$; exact agreement with the old curve is not required.
The stated targets—resembling the archived conditional location posterior and obtaining $P(K=1)>1/2$—
are reported for **every setting**, not imposed as numerical acceptance criteria. Selecting a setting after
looking at these outcomes is exploratory and cannot establish evidence for a change on independent data.

The raw UCI myocardial data, target ZSN, 1,692 observations and 110 features are preprocessed exactly as in
the fresh audit: age sorting, mean imputation, centering, covariance whitening with ridge $10^{-6}$,
and division by $\sqrt n$. There is no intercept. The observation model and posterior remain logistic.
For each minimum segment length $\Delta\in\{1,85,169\}$, retain distinct-age splits leaving at least
$\Delta$ observations on **both** sides. Give the single no-change state probability $1/2$ and divide
the other $1/2$ equally over admissible splits. Compute the fitting weights by marginalizing this same prior.
Thus changing $\Delta$ changes the prior support and weights, never the total prior odds on the count.

Auxiliary fits for $\hat\rho$: exact weighted least squares, or weighted logistic loss with
L2 strengths $10^{-8},10^{-6}$ or L1 strengths $10^{-6},10^{-4}$. The convention is
$\sum_iw_i\ell_i/\sum_iw_i+\lambda\|\beta\|_q^q$, with no $1/2$ in L2.
All four penalized logistic choices are also compared independently for the subsequent posterior fit.
They regularize the separated logistic fit; their relative practical strength depends on the whitened design.
The adjustment and state equations are the **unpenalized** ones, used as a small-penalty approximation;
no exact penalized-theory claim is made.

Initialize $\rho=I_2$, alternate state evaluation and covariance moment matching for
1, 3, 10, or 30 updates, use damping 0.25, and project eigenvalues to $[10^{-4},100]$.
The covariance bounds and damping are explicit numerical regularizers, not estimated parameters.
This is a reproducible original-*style* finite alternating scheme, **not a bitwise replay** of the legacy
three-update routine with frozen diagonals and changing Monte Carlo seeds. Also include the independent
constrained least-squares moment approximation (iteration label “joint”). Keep actual moment residuals.
Logistic state equations use common scrambled Sobol Gaussian draws, analytic Bernoulli averaging,
and Stein's identity. The initial screen uses 128 draws and at most 40 nonlinear-solver evaluations
per candidate; auxiliary iterations use the separately recorded settings visible below.

The primary likelihood is the audited **joint** density $p(V,Y\mid\eta)$, including $p(V\mid\eta)$
and the correct cross covariance $\rho\Lambda$. Unlike the legacy evaluation, it does not silently cancel
candidate-dependent terms. Consequently reproducing the archived *shape* is a diagnostic, not a reason
to discard these terms. See `docs/myocardial_signal_strength_audit.md` for the derivation and explanation
of why the older finite-iteration calculation returned values without solving the moment equations.

**Archived baseline context (not a fresh estimate):** under the archived likelihood and fitted scores,
the previous two-boundary mixture null gives $P(K=1)=0.334942$ after imposing half–half count odds;
using its all-signal-0 boundary alone gives $0.267894$. The earlier high probability near 0.94 used
equal mass per configuration, hence much larger total prior mass on one change. The original
conditional location curve and the count posterior are distinct objects.

All code below is visible and executable. Matching checkpoints are reused; deleting the uniquely named
checkpoint directory recomputes the grid. Original notebooks and archived results are not overwritten.
'''),nbf.v4.new_code_cell(source),nbf.v4.new_code_cell('output = run()')]
cells.append(nbf.v4.new_code_cell(r'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display
fit_checks=[]
for d in DELTAS:
    setting=pickle.load((output/f'fits_delta{d}.pkl').open('rb'))
    for name,fit in setting['fits'].items():
        fit_checks.append(dict(Delta=d,fit=name,max_KKT=max(c['kkt'] for c in fit['checks']),
            all_fit_stops_successful=all(c['success'] for c in fit['checks']),
            adjusted_second_moment_norm=np.linalg.norm(fit['V'].T@fit['V']/len(fit['V'])),
            empirical_b=list(fit['b'])))
print('Fitting diagnostics (separate from state and covariance moment residuals):')
display(pd.DataFrame(fit_checks))
import jax
jax.config.update('jax_enable_x64',True)
reference=pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl','rb'))
reference_ages=np.asarray(reference['unique_ages']).ravel()
reference_weights=np.asarray(reference['posterior']).ravel()
valid=np.isfinite(reference_ages)
# Gold plot includes empty boundaries. Compare conditional proper age-split masses only.
X,y,ages,locations=load_data()
valid &= np.isin(reference_ages,ages[locations])
reference_ages=reference_ages[valid];reference_weights=reference_weights[valid]
reference_weights=reference_weights/reference_weights.sum()
reference_map=dict(zip(reference_ages,reference_weights))
records=[pickle.load(p.open('rb')) for p in sorted(output.glob('d*_q7_b40.pkl'))]
rows=[]
for rec in records:
    support=ages[rec['eta'][1:]]
    ref=np.array([reference_map.get(a,0.) for a in support]);retained=ref.sum();ref/=retained
    for fit,result in rec['results'].items():
        tv_on_support=.5*np.abs(result['conditional']-ref).sum()
        tv=.5*(np.abs(result['conditional']-ref*retained).sum()+1-retained)
        result['tv_to_reference_on_support']=float(tv)
        rows.append(dict(setting=rec['label'],Delta=rec['delta_min'],auxiliary=rec['scenario']['auxiliary'],
            iterations=rec['scenario']['iteration'],posterior_fit=fit,P_1=result['presence'],
            mass_above55=float(result['conditional'][support>55].sum()),
            peak_age=result['peak_age'],TV_to_reference=tv,TV_on_common_support=tv_on_support,reference_mass_retained=retained,
            log_Bayes_factor=logsumexp(result['log_likelihood'][1:])-np.log(len(support))-result['log_likelihood'][0],
            moment_residual=rec['scenario']['moment_residual'],FP_residual_max=max(rec['residuals'])))
summary=pd.DataFrame(rows)
summary.to_csv(output/'summary_with_reference.csv',index=False)
print('All comparisons; lower total variation (TV) means a closer conditional curve.')
with pd.option_context('display.max_rows',300,'display.max_columns',20): display(summary)
print('P(1)>0.5:',int((summary.P_1>.5).sum()),'of',len(summary))
summary['meets_updated_target']=(summary.P_1>.5)&(summary.mass_above55>.5)
print('Meets both updated targets:',int(summary.meets_updated_target.sum()),'of',len(summary))
display(summary[summary.meets_updated_target].sort_values('moment_residual').head(20))
print('Closest conditional curves (explicit outcome-based exploratory ranking):')
display(summary.sort_values('TV_to_reference').head(12))
print('Closest conditional curves among P(1)>0.5, if any:')
display(summary[summary.P_1>.5].sort_values('TV_to_reference').head(12))
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':14,
                     'axes.labelsize':16,'xtick.labelsize':12,'ytick.labelsize':12,'figure.dpi':150})
'''))
cells.append(nbf.v4.new_markdown_cell('## Complete conditional-posterior atlas\nDashed grey: archived reference renormalized over the same admissible support. Colored curves: every finite update budget; black: joint least-squares moment approximation.'))
cells.append(nbf.v4.new_code_cell(r'''
print('Closest curves: exploratory selection, not an independent validation.')
ranked=summary.sort_values('TV_to_reference').head(3)
qualifying=summary[summary.P_1>.5].sort_values('TV_to_reference').head(3)
for row in pd.concat([ranked,qualifying]).drop_duplicates(['setting','posterior_fit']).itertuples():
    rec=next(r for r in records if r['label']==row.setting)
    stats=rec['results'][row.posterior_fit];support=ages[rec['eta'][1:]]
    ref=np.array([reference_map.get(a,0.) for a in support]);ref/=ref.sum()
    print(row.setting,row.posterior_fit,'P(1)=',row.P_1,'TV=',row.TV_to_reference)
    with plt.rc_context({'font.size':22,'axes.labelsize':22,'xtick.labelsize':22,'ytick.labelsize':22}):
        fig,ax=plt.subplots(figsize=(12,3))
        ax.plot(support,ref,color='.6',ls='--',lw=1.5,label='Archived conditional')
        marker,stems,baseline=ax.stem(support,stats['conditional'],linefmt='k-',markerfmt='k.',basefmt=' ')
        plt.setp(stems,linewidth=1.2);plt.setp(marker,markersize=3)
        ax.set_xlabel('Age');ax.set_ylabel('Probability')
        ax.set_xticks([26,40,50,60,75,81,92]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_code_cell(r'''
colors={1:'#d95f02',3:'#7570b3',10:'#1b9e77',30:'#e7298a',-1:'black'}
for d in DELTAS:
    for fit in (f'{k}_{v:g}' for k,v in FIT_SETTINGS):
        print('Minimum segment length:',d,'; posterior fit:',fit)
        fig,axes=plt.subplots(2,3,figsize=(15,6),sharex=True,sharey=True)
        for ax,aux in zip(axes.flat,['least_squares']+[f'{k}_{v:g}' for k,v in FIT_SETTINGS]):
            subset=[r for r in records if r['delta_min']==d and r['scenario']['auxiliary']==aux]
            if not subset:continue
            support=ages[subset[0]['eta'][1:]]
            ref=np.array([reference_map.get(a,0.) for a in support]);ref/=ref.sum()
            ax.plot(support,ref,color='.55',ls='--',lw=1.5,label='Archived')
            for rec in sorted(subset,key=lambda r:r['scenario']['iteration']):
                it=rec['scenario']['iteration'];stats=rec['results'][fit]
                ax.plot(support,stats['conditional'],color=colors[it],lw=1.3,
                    label=('joint' if it==-1 else str(it))+f"; P(1)={stats['presence']:.3g}")
            ax.set_title(aux);ax.set_xlabel('Age');ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.legend(fontsize=8,frameon=False)
        axes[0,0].set_ylabel('Conditional probability');axes[1,0].set_ylabel('Conditional probability')
        axes.flat[-1].axis('off');fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_markdown_cell('## Count probability and residual sensitivity\nNo setting is hidden because it has a nonzero equation residual or an undesired count posterior. A small state-equation residual alone does not establish moment fit or model adequacy.'))
cells.append(nbf.v4.new_code_cell(r'''
for d in DELTAS:
    print('Minimum segment length:',d)
    fig,axes=plt.subplots(1,3,figsize=(15,3.5))
    subset=summary[(summary.Delta==d)&(summary.iterations>0)]
    for (aux,fit),part in subset.groupby(['auxiliary','posterior_fit']):
        part=part.sort_values('iterations');label=aux+' / '+fit
        axes[0].plot(part.iterations,part.P_1,'.-',label=label)
        axes[1].plot(part.iterations,part.TV_to_reference,'.-')
    for aux,part in subset.drop_duplicates(['auxiliary','iterations']).groupby('auxiliary'):
        part=part.sort_values('iterations');axes[2].plot(part.iterations,part.moment_residual,'.-',label=aux)
    axes[0].axhline(.5,color='.6',ls='--');axes[0].set_ylabel('P(1 chgpt)');axes[0].set_ylim(-.03,1.03)
    axes[1].set_ylabel('TV to archived conditional curve');axes[2].set_ylabel('Moment residual')
    axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:ax.set_xlabel('Rho updates');ax.set_xticks(ITERATIONS);ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_markdown_cell(r'''## Numerical-budget sensitivity and likelihood decomposition

The cases below were specified without consulting posterior outcomes: all five auxiliary fits at
3 and 30 rho updates for $\Delta=1$, every LS update budget there, plus the joint LS approximation at each $\Delta$.
Compare the 128-draw/40-evaluation screen with 128/10 and 512/100. The conditional-only column
deliberately omits $p(V\mid\eta)$ to show its impact; it is a **diagnostic pseudo-posterior**, not
the primary joint posterior or a justification for selecting the desired count probability.
'''))
refinement=(ROOT/'tools/myocardial_exploration_refinement.py').read_text().split("if __name__=='__main__':")[0]
refinement=refinement.replace('from tools.myocardial_exploration import *','')
cells.append(nbf.v4.new_code_cell(refinement+'\nrefine(output)'))
cells.append(nbf.v4.new_code_cell(r'''
numerical=pd.read_csv(output/'numerical_refinement.csv')
numerical['meets_updated_target']=(numerical.P_1>.5)&(numerical.mass_above55>.5)
decomposition=pd.read_csv(output/'conditional_only_diagnostic.csv')
quadrature_check=pd.read_csv(output/'quadrature_check.csv')
print('Numerical budget comparison:')
with pd.option_context('display.max_rows',150):display(numerical)
print('Joint posterior versus conditional-only scoring (not interchangeable):')
with pd.option_context('display.max_rows',300):display(decomposition)
print('Largest absolute count probability change:',numerical.delta_P_1.abs().max())
print('Largest conditional-curve TV change:',numerical.TV_from_coarse.max())
print('Refined cases meeting both updated targets:')
display(numerical[numerical.meets_updated_target])
print('Independent 80-versus-160-node likelihood quadrature check:')
display(quadrature_check)
fig,axes=plt.subplots(1,2,figsize=(10,3.3))
axes[0].scatter(numerical.coarse_P_1,numerical.P_1,s=14,alpha=.5)
axes[0].plot([0,1],[0,1],color='black',lw=1)
axes[0].set_xlabel('Coarse P(1 chgpt)');axes[0].set_ylabel('Other budget P(1 chgpt)')
axes[1].scatter(decomposition.joint_P_1,decomposition.conditional_only_P_1,s=14,alpha=.5)
axes[1].plot([0,1],[0,1],color='black',lw=1)
axes[1].set_xlabel('Joint posterior P(1 chgpt)');axes[1].set_ylabel('Conditional-only score')
for ax in axes:ax.xaxis.set_major_locator(ticker.MaxNLocator(3));ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_markdown_cell(r'''## Denser penalty exploration

To resolve sensitivity between the initial penalty choices, use 13 logarithmically spaced L2
strengths from $10^{-9}$ to $10^{-3}$ and nine L1 strengths from $10^{-7}$ to $10^{-3}$.
Keep $\Delta=1$ and compare all five LS covariance estimates and the moderate-ridge auxiliary
estimate at 3 and 30 updates. Reuse the **512-draw, 100-evaluation** refined states and use 160-node
likelihood quadrature. This is 154 additional comparisons; none changes the prior odds.
'''))
sweep=(ROOT/'tools/myocardial_penalty_sweep.py').read_text().split("if __name__=='__main__':")[0]
sweep=sweep.replace('from tools.myocardial_exploration import *','')
cells.append(nbf.v4.new_code_cell(sweep+'\nsweep_results=penalty_sweep(output)'))
cells.append(nbf.v4.new_code_cell(r'''
sweep_rows=[]
for r in sweep_results:
    row={k:v for k,v in r.items() if k!='conditional'}
    if 'error' not in r:
        row['TV_to_reference']=.5*np.abs(r['conditional']-np.array([reference_map[a] for a in ages[locations]])).sum()
    sweep_rows.append(row)
sweep_summary=pd.DataFrame(sweep_rows)
sweep_summary['meets_updated_target']=(sweep_summary.P_1>.5)&(sweep_summary.mass_above55>.5)
sweep_summary.to_csv(output/'denser_penalty_grid'/'summary_with_reference.csv',index=False)
with pd.option_context('display.max_rows',200):display(sweep_summary)
print('Closest refined curves among P(1)>0.5; exploratory ranking:')
display(sweep_summary[sweep_summary.P_1>.5].sort_values('TV_to_reference').head(15))
print('Refined penalty settings meeting both updated targets:')
display(sweep_summary[sweep_summary.meets_updated_target].sort_values('TV_to_reference'))
for kind in ('l2','l1'):
    print('Posterior penalty:',kind)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for name,part in sweep_summary[sweep_summary.penalty==kind].groupby('base_setting'):
        part=part.sort_values('strength')
        axes[0].semilogx(part.strength,part.P_1,'.-',label=name.replace('d1_','').replace('_q9_b100',''))
        axes[1].semilogx(part.strength,part.TV_to_reference,'.-')
    axes[0].axhline(.5,color='.6',ls='--');axes[0].set_ylabel('P(1 chgpt)')
    axes[1].set_ylabel('TV to archived conditional curve')
    axes[0].legend(fontsize=7,frameon=False)
    for ax in axes:ax.set_xlabel('Penalty strength');ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    fig.tight_layout();plt.show()
for row in sweep_summary.sort_values('TV_to_reference').head(3).itertuples():
    r=next(r for r in sweep_results if r['label']==row.label)
    print(row.label,'P(1)=',row.P_1,'TV=',row.TV_to_reference)
    fig,ax=plt.subplots(figsize=(12,3))
    ax.plot(reference_ages,reference_weights,color='.6',ls='--',label='Archived conditional')
    ax.plot(ages[locations],r['conditional'],color='black',lw=1.5,label='Exploratory fit')
    ax.set_xlabel('Age');ax.set_ylabel('Probability');ax.set_xticks([26,40,50,60,75,81,92])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.legend(fontsize=10,frameon=False)
    fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_markdown_cell(r'''## Independent-draw follow-up

After inspecting the grid, follow up the LS 3-update (sharper) and 30-update (diffuse) cases
with a fixed independent seed 123 and 1,024 draws. This is an explicitly outcome-informed
follow-up, not another seed search. All 22 posterior penalties are reevaluated at both settings.
The uniform conditional age-grid prior already puts more than half its mass above 55;
meeting that location criterion alone need not imply concentration beyond the prior.
'''))
validation=(ROOT/'tools/myocardial_independent_draw_check.py').read_text().split("if __name__=='__main__':")[0]
validation=validation.replace('from tools.myocardial_exploration import *','')
cells.append(nbf.v4.new_code_cell(validation+'\ncheck_baseline(output)'))
cells.append(nbf.v4.new_code_cell(r'''
validation=pd.read_csv(output/'independent_draw_check.csv')
print('Conditional prior mass above age 55:',np.mean(ages[locations]>55))
with pd.option_context('display.max_rows',60):display(validation)
for iteration,name in [(3,'l2_1e-06'),(30,'l1_0.00031622777')]:
    case=f'd1_least_squares_it{iteration}_q9_b100'
    primary=pickle.load((output/'denser_penalty_grid'/(case+'_'+name+'.pkl')).open('rb'))
    independent=pickle.load((output/'denser_penalty_grid'/(case+'_'+name+'_independent.pkl')).open('rb'))
    print(case,name,'P(1):',primary['P_1'],independent['P_1'])
    fig,ax=plt.subplots(figsize=(12,3))
    ax.plot(reference_ages,reference_weights,color='.6',ls='--',label='Archived conditional')
    ax.plot(ages[locations],primary['conditional'],color='black',lw=1.5,label='512 draws; seed 42')
    ax.plot(ages[locations],independent['conditional'],color='#377eb8',lw=1.2,label='1024 draws; seed 123')
    ax.axvline(55,color='.7',ls=':',lw=1);ax.set_xlabel('Age');ax.set_ylabel('Probability')
    ax.set_xticks([26,40,50,60,75,81,92]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.legend(frameon=False,fontsize=10)
    fig.tight_layout();plt.show()
'''))
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (phase1)','language':'python','name':'python3'}})
nbf.write(nb,path)
print(path)
