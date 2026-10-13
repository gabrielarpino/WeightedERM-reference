"""Notebook with visible experiment code and automatically rendered comparisons."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'Myocardial Infarction Experiment - finite iteration exploration.ipynb'


def build():
    intro=r'''# Myocardial infarction: finite-iteration logistic fits

**Preferred model:** no intercept, centered/whitened predictors, logistic posterior
likelihood, and no explicit fitting penalty. The auxiliary fit is either exact
weighted least squares or finite-iteration weighted logistic FISTA. Each logistic
fit starts at zero, uses weighted mean loss, a maximum step equal to the reciprocal
of the global weighted logistic curvature bound, and monotone FISTA/backtracking.
Budgets are **1, 3, 10, 30, 100, 300, 1,000** iterations. Actual iteration counts,
gradient residuals, and coefficient norms are recorded; failure to converge is
intentional in this early-stopping study, not silently treated as an exact optimum.

The 1,692 age-sorted observations have 110 mean-imputed, centered predictors,
whitened using covariance plus $10^{-6}I$ and divided by $\sqrt n$.
Minimum segment size is $\Delta\in\{1,85,169\}$. At each setting, the count prior
assigns probability $1/2$ to zero and one change; conditional location is uniform
over admissible distinct-age splits. There is one null configuration (all signal 0).
No fitted intercept, prevalence offset, or likelihood temperature is introduced.

Signal-strength estimation starts at $I_2$, uses damping 0.25 and eigenvalue bounds
$[10^{-4},100]$, and saves **1, 3, 10, 30** updates; the direct approximate LS moment
fit is also included. Logistic auxiliary and posterior budgets are fully crossed,
with matched-budget results identified separately. The baseline contains **693**
posterior comparisons (3 minimum segment sizes × 33 rho settings × 7 posterior fits).

Primary states use 256 Sobol draws (seed 42), at most 60 state-solver evaluations,
and 160-node Gaussian quadrature for reported likelihoods. In each auxiliary class,
the three closest reference curves and two highest-count-probability curves within
0.15 TV of that class's best are selected for checks using 512 draws/seed 42 and
1,024 draws/seed 123, both with at most 100 solver evaluations. **There is no
$P(1)>0.5$ eligibility threshold.** TV is computed on the full reference age grid,
including reference probability excluded by a larger $\Delta$.

This is a **post-hoc heuristic sensitivity exploration**. The existing asymptotic
state equations are used as an approximation for early-stopped fits; they do not
become exact finite-iteration theory. Independent numerical draws assess integration
sensitivity, not independent-data validity. Nonzero moment residuals are retained.
The zero-offset Gaussian logistic model implies marginal prevalence 1/2, versus
about 0.232 observed; centering does not fix that mismatch. Similarity to the old
curve is an explicit exploratory objective, not evidence of statistical correctness.
'''
    source=(ROOT/'tools/myocardial_finite_iteration_exploration.py').read_text().split("if __name__=='__main__':")[0]
    execution='''root=run()
summary=pd.read_csv(root/'summary.csv')
validation=pd.read_csv(root/'validation.csv')
ranked=pd.read_csv(root/'ranked.csv')
print('Numerically checked shortlist, sorted by similarity to the archived conditional curve')
display(ranked)
'''
    overview=r'''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
                     'axes.labelsize':12,'figure.dpi':150})
colors={'least_squares':'black','logistic_matched':'#c69c22','logistic_different_budgets':'#4178a8'}
print('All settings: shape versus count evidence; primary numerical accuracy only')
fig,axes=plt.subplots(1,3,figsize=(12,3.2),sharex=True,sharey=True)
for ax,(group,data) in zip(axes,summary.groupby('group')):
    ax.scatter(data.TV_to_reference,data.P_1,s=15,alpha=.55,color=colors[group])
    ax.set_title(group.replace('_',' '),fontsize=10)
    ax.set_xlabel('TV from archived curve');ax.axhline(.5,color='0.6',ls=':',lw=.8)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(3))
axes[0].set_ylabel('P(1 chgpt)');fig.tight_layout();plt.show()

print('Least-squares auxiliary: iteration-budget sensitivity')
for delta in DELTAS:
    data=summary[(summary.group=='least_squares')&(summary.Delta==delta)]
    print('Minimum segment size Delta =',delta)
    fig,axes=plt.subplots(1,3,figsize=(12,3.2))
    for updates,sub in data.groupby('rho_updates'):
        sub=sub.sort_values('posterior_budget');label='Direct LS rho' if updates<0 else f'{updates} rho updates'
        for ax,key in zip(axes,('P_1','TV_to_reference','mass_above55')):
            ax.semilogx(sub.posterior_budget,sub[key],'-D',ms=3,label=label)
    for ax,label in zip(axes,('P(1 chgpt)','TV from archived curve','P(age > 55 | 1 chgpt)')):
        ax.set_ylabel(label);ax.set_xlabel('Logistic fit iterations');ax.yaxis.set_major_locator(MaxNLocator(3))
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()
'''
    curves=r'''reference=reference_distribution();a=np.array(sorted(reference));p=np.array([reference[x] for x in a])
print('Refined comparisons: all shortlisted settings, including unfavorable numerical checks')
for row in ranked.itertuples():
    subset=validation[(validation.case==row.case)&(validation.posterior_fit==row.posterior_fit)]
    desc=subset.iloc[0]
    print(f'{row.group}: Delta={desc.Delta}; auxiliary budget={desc.auxiliary_budget}; '
          f'posterior budget={desc.posterior_budget}; rho updates={desc.rho_updates}')
    print(f'P(1) range {row.P1_min:.4g}–{row.P1_max:.4g}; mean TV={row.mean_TV:.4g}; '
          f'moment residual={row.moment_residual:.4g}')
    fig,ax=plt.subplots(figsize=(6,2.8))
    ax.plot(a,p,color='0.65',lw=1.3,label='Archived conditional posterior')
    for tag,color,label in [('q9_seed42','black','512 draws'),('q10_seed123','#4178a8','1,024 draws')]:
        saved=pickle.load((root/(row.case+'_'+row.posterior_fit+'_'+tag+'.pkl')).open('rb'))
        ax.plot(saved['ages'],saved['conditional'],color=color,lw=1.3,label=label)
    ax.axvline(55,color='0.6',ls=':',lw=.8);ax.set_xlabel('Age');ax.set_ylabel('Probability')
    ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()
display(validation[['group','Delta','auxiliary_budget','posterior_budget','rho_updates','tag',
                    'P_1','mass_above55','peak_age','interval90_lower','interval90_upper',
                    'TV_to_reference','moment_residual','FP_residual_max','fit_kkt_max','coefficient_norm_max']])
'''
    historical='''## Earlier candidates retained for comparison

These historical results are not overwritten by the early-stopping study:

| Auxiliary / posterior-input fit | Intercept | P(1) across two numerical checks | Conditional location |
|---|---|---|---|
| LS / logistic L1 1e-4, 30 rho updates | No | 0.402–0.453 | Mode 79; 79–82% mass above 55; closest earlier no-intercept curve |
| LS / logistic L2 1e-6, 3 rho updates | No | 0.99988–0.99998 | Mode 87; sharper/later than archived curve |
| LS / logistic L2 1e-5, 30 rho updates | No | 0.481–0.484 | Mode 75; diffuse |
| Matched logistic L2 1e-4, 30 rho updates | No | 0.515–0.531 | Mode 65–66; diffuse |
| Logistic auxiliary L1 1e-4 / posterior L2 1.5e-6, 30 updates | Yes | 0.718–0.747 | Mode 81; mean TV 0.291; mismatched families and strengths |
| Logistic auxiliary L1 1e-4 / posterior L1 1e-5, 30 updates | Yes | 0.654–0.680 | Mode 81; mean TV 0.367; mismatched strengths |

All are heuristic and were inspected post hoc. Probability ranges above are not
confidence intervals. Full historical plots and diagnostics remain in the
heuristic, intercept, and exact-matching notebooks.
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execution),nbf.v4.new_code_cell(overview),nbf.v4.new_code_cell(curves),
        nbf.v4.new_markdown_cell(historical)])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(nb,PATH);print(PATH)


if __name__=='__main__':build()
