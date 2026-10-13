from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'Myocardial Infarction Experiment - intercept exploration.ipynb'
base=(ROOT/'tools/myocardial_exploration.py').read_text().split("if __name__=='__main__':")[0]
extension=(ROOT/'tools/myocardial_intercept_exploration.py').read_text().split("if __name__=='__main__':")[0]
extension=extension.replace('from tools.myocardial_exploration import *','')
cells=[nbf.v4.new_markdown_cell(r'''# Myocardial infarction: centering and intercept extension

**Targets:** $P(K=1)>1/2$ and most conditional change-location mass above age 55;
among those settings, a curve closer to the archived curve is preferable. This is exploratory
selection on the same data, not confirmatory evidence or proof of posterior calibration.

Centering the feature columns does not replace a logistic intercept: without an intercept,
the fitted probability at the mean feature vector is 1/2. The response rate is about 0.232.
With an unpenalized intercept and a fixed whitening/scaling transformation, centering is an exact
affine reparameterization; the notebook verifies this identity on the actual fitted predictions.
All data processing, prior count odds, age grid, and 110-feature design otherwise match the
preceding heuristic exploration. Here $\Delta=1$ throughout.

## Model-aware intercept branch

Fit unpenalized intercepts with weighted LS, or logistic L2 ($10^{-6}$) / L1 ($10^{-4}$)
for auxiliary signal-strength estimation. Keep logistic L2 ($10^{-8},10^{-6}$) and L1
($10^{-6},10^{-4}$) fits for posterior evaluation. Center adjusted scores by their fitted
total offsets. In the auxiliary squared-loss state equations, the fitted offset is the response
rate $r$, and the noise second-moment term becomes $r(1-r)$, not $1/2$.

Use the extended observation model $Y\mid Z_s\sim\mathrm{Bernoulli}(\sigma(\mu_s+Z_s))$.
At each proposed covariance, calibrate each $\mu_s$ to the **same global observed rate**
$E\sigma(\mu_s+N(0,\rho_{ss}))=r$. No pilot split or target change age is used to set that rate.
This is an explicit empirical-Bayes model extension; it is not silently passed off as the
original centered, zero-intercept observation model. The logistic state solve includes two
additional unknown fitted intercepts and their two mean-score equations. Evaluate the
correct joint density of the centered adjusted scores and the logistic observations, with
the response offsets included in the conditional observation probabilities.

Initialize $\rho=I_2$, retain 1/3/10/30 damped updates (weight 0.25), with covariance eigenvalues
clipped to $[10^{-4},100]$. Add a directly optimized LS moment fit, with log standard deviations
in $[-8,\log 10]$ and correlation in $[-0.9999,0.9999]$. Return nonzero residuals rather than
rejecting estimates. Posterior states use 512 common scrambled Sobol draws, at most 100 nonlinear
solver evaluations, and 160-node likelihood quadrature. There are 52 comparisons in this branch.
Small fitting penalties still use approximately unpenalized state equations; no exact penalized
asymptotic theorem is claimed.

## Fit-only uncentered diagnostic

As a separate diagnostic, do not subtract feature means before applying the **same** whitening
transform, omit the intercept, recompute LS auxiliary estimates and all four logistic posterior
fits, and retain the original zero-mean equations. This deliberately tests the requested fitting
knob but violates that model's centered-design assumption. These 20 comparisons are explicitly
labeled and are not interchangeable with the model-aware intercept branch.

The preceding notebook retains the complete centered/no-intercept grid, numerical refinements,
and denser penalty sweep. Original reference files are not changed. All experiment code is visible below.
'''),nbf.v4.new_code_cell(base),nbf.v4.new_code_cell(extension),nbf.v4.new_code_cell('output=run_intercept_exploration()')]
cells.append(nbf.v4.new_code_cell(r'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display
import jax
jax.config.update('jax_enable_x64',True)
reference=pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl','rb'))
X,y,ages,locations=load_data()
ref_ages=np.asarray(reference['unique_ages']).ravel();ref_mass=np.asarray(reference['posterior']).ravel()
keep=np.isfinite(ref_ages)&np.isin(ref_ages,ages[locations]);ref_ages=ref_ages[keep];ref_mass=ref_mass[keep];ref_mass/=ref_mass.sum()
ref_map=dict(zip(ref_ages,ref_mass));reference_vector=np.array([ref_map[a] for a in ages[locations]])
records=[pickle.load(p.open('rb')) for p in sorted(output.glob('offset_*_q9_b100.pkl'))]
summary=pd.read_csv(output/'summary.csv')
summary['meets_targets']=(summary.P_1>.5)&(summary.mass_above55>.5)
summary['TV_to_reference']=np.nan
for rec in records:
    for name,result in rec['results'].items():
        index=(summary.setting==rec['label'])&(summary.posterior_fit==name)
        summary.loc[index,'TV_to_reference']=.5*np.abs(result['conditional']-reference_vector).sum()
summary.to_csv(output/'summary_with_reference.csv',index=False)
fits=pickle.load((output/'offset_fits.pkl').open('rb'))
print('Centering with an intercept: exact reparameterization check')
display(pd.DataFrame(fits['affine']))
print('All outcomes, including diagnostics that retain inappropriate old mean equations:')
with pd.option_context('display.max_rows',100,'display.max_columns',20):display(summary)
eligible=summary[(summary.model=='prevalence-calibrated intercept')&summary.meets_targets].sort_values('TV_to_reference')
print('Model-aware intercept settings meeting both targets, ranked by curve similarity (exploratory):')
display(eligible)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':14,
                     'axes.labelsize':16,'xtick.labelsize':12,'ytick.labelsize':12,'figure.dpi':150})
'''))
cells.append(nbf.v4.new_code_cell(r'''
colors={1:'#d95f02',3:'#7570b3',10:'#1b9e77',30:'#e7298a',-1:'black'}
for fit in (f'{k}_{v:g}' for k,v in FIT_SETTINGS):
    print('Logistic posterior fit:',fit)
    fig,axes=plt.subplots(1,3,figsize=(15,3.4),sharex=True,sharey=True)
    for ax,aux in zip(axes,['least_squares','l2_1e-06','l1_0.0001']):
        ax.plot(ref_ages,ref_mass,color='.6',ls='--',label='Archived conditional')
        for rec in records:
            if rec['scenario']['auxiliary']!=aux:continue
            it=rec['scenario']['iteration'];result=rec['results'][fit]
            ax.plot(ages[locations],result['conditional'],color=colors[it],lw=1.4,
                label=('joint' if it==-1 else str(it))+f"; P(1)={result['P_1']:.3g}")
        ax.axvline(55,color='.75',ls=':',lw=1);ax.set_xlabel('Age');ax.set_title(aux)
        ax.set_xticks([40,55,70,85]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.legend(fontsize=8,frameon=False)
    axes[0].set_ylabel('Conditional probability');fig.tight_layout();plt.show()
'''))
cells.append(nbf.v4.new_code_cell(r'''
print('Closest eligible intercept curves; these are outcome-selected examples.')
for row in eligible.head(5).itertuples():
    rec=next(r for r in records if r['label']==row.setting);stats=rec['results'][row.posterior_fit]
    print(row.setting,row.posterior_fit,'P(1)=',row.P_1,'P(age>55 | 1)=',row.mass_above55,'TV=',row.TV_to_reference)
    with plt.rc_context({'font.size':22,'axes.labelsize':22,'xtick.labelsize':22,'ytick.labelsize':22}):
        fig,ax=plt.subplots(figsize=(12,3))
        ax.plot(ref_ages,ref_mass,color='.6',ls='--',lw=1.5)
        marker,stems,baseline=ax.stem(ages[locations],stats['conditional'],linefmt='k-',markerfmt='k.',basefmt=' ')
        plt.setp(stems,linewidth=1.2);plt.setp(marker,markersize=3)
        ax.axvline(55,color='.7',ls=':',lw=1);ax.set_xlabel('Age');ax.set_ylabel('Probability')
        ax.set_xticks([26,40,50,60,75,81,92]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        fig.tight_layout();plt.show()
fig,axes=plt.subplots(1,2,figsize=(10,3.3))
for model,part in summary.groupby('model'):
    axes[0].scatter(part.P_1,part.mass_above55,s=20,alpha=.6,label=model)
    axes[1].scatter(part.moment_residual,part.P_1,s=20,alpha=.6,label=model)
axes[0].axvline(.5,color='.6',ls='--');axes[0].axhline(.5,color='.6',ls='--')
axes[0].set_xlabel('P(1 chgpt)');axes[0].set_ylabel('P(age > 55 | 1 chgpt)')
axes[1].set_xlabel('Moment residual');axes[1].set_ylabel('P(1 chgpt)')
axes[0].legend(fontsize=8,frameon=False)
for ax in axes:ax.xaxis.set_major_locator(ticker.MaxNLocator(3));ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
fig.tight_layout();plt.show()
'''))
from myocardial_handoff_cells import intercept_cells
cells.extend(intercept_cells(ROOT))
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (phase1)','language':'python','name':'python3'}})
nbf.write(nb,path);print(path)
