"""Build the preferred, fully visible exact-matching experiment notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'Myocardial Infarction Experiment - matched penalties no intercept.ipynb'


def build():
    intro = r'''# Myocardial infarction: no intercept, exactly matched penalties

This is the preferred sensitivity comparison: centered and whitened predictors,
**no fitted intercept**, and the **same logistic coefficient fit** (same penalty
family, strength, weights and normalization) reused in auxiliary signal-strength
estimation and posterior-input construction. The outcome remains logistic.
L1 and L2 constants are $10^{-8},10^{-7},10^{-6},10^{-5},10^{-4}$, on a weighted
mean-loss scale; L2 means $\lambda\|\beta\|_2^2$, without a factor of one-half.
Small constants are preferred, but numerical diagnostics and all outcomes are shown.

There are 1,692 observations and 110 predictors. Missing predictor values are mean
imputed; centered covariance whitening uses a $10^{-6}I$ ridge, followed by division
by $\sqrt n$. The prior gives 0 and 1 change point probability $1/2$ each, with
uniform location conditional on one change over 61 proper distinct-age splits.
Minimum segment size is $\Delta=1$. The signal-strength heuristic starts at $I_2$,
uses damping 0.25, projects eigenvalues to $[10^{-4},100]$, and is saved after
1, 3, 10 and 30 updates. Nonzero equation residuals are retained and reported.

Primary posterior states use 512 Sobol draws (seed 42), at most 100 state-solver
evaluations, and 160-node Gaussian quadrature for reported probabilities. Independent
checks use 1,024 draws (seed 123). The smallest strength is checked at **every**
iteration count regardless of outcome; the closest eligible curve in each family
is also checked. This latter comparison is explicitly outcome-guided exploration,
not independent-data validation or calibrated selection.

The penalties stabilize coefficient fits; they are **not** added to the posterior
log likelihood. Approximately unpenalized state equations are still used, so these
finite-iteration penalized-fit calculations remain heuristic. In particular,
centering predictors does not replace a logistic intercept: the zero-offset
Gaussian model implies marginal response probability 1/2, whereas the observed rate
is about 0.232. No-intercept results are preferred by request, not because that
model mismatch has disappeared. Intercept-aware results remain a separate sensitivity.
The conditional prior itself places 35/61 (about 0.574) mass above age 55.
'''
    code = (ROOT/'tools/myocardial_exact_no_intercept.py').read_text().split("if __name__ == '__main__':")[0]
    run = '''import jax
jax.config.update('jax_enable_x64', True)
output = Path('results/myocardial_exploration_4026d0e775d2')
summary, independent = run(output)
root = output/'exact_matching_no_intercept'
display(summary)
display(independent)
'''
    plots = r'''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm',
                     'font.size':11, 'axes.labelsize':12, 'figure.dpi':150})
colors = {'l1':'black', 'l2':'#c69c22'}
print('All exact matches: count evidence and conditional mass above 55')
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey='row')
for j, it in enumerate((1,3,10,30)):
    for family in ('l1','l2'):
        data = summary[(summary.iteration==it)&(summary.penalty==family)]
        for ax, key in zip(axes[:,j], ('P_1','mass_above55')):
            ax.semilogx(data.strength, data[key], '-D', ms=3, color=colors[family], label=family.upper())
            ax.axhline(.5, color='grey', ls=':', lw=.8)
            ax.set_ylim(-.03,1.03); ax.yaxis.set_major_locator(MaxNLocator(3))
    axes[0,j].set_title(f'{it} updates'); axes[1,j].set_xlabel(r'Penalty $\lambda$')
axes[0,0].set_ylabel('P(1 chgpt)'); axes[1,0].set_ylabel('P(age > 55 | 1 chgpt)')
axes[0,0].legend(frameon=False); fig.tight_layout(); plt.show()

print('Conditional location posterior: every family, strength and iteration count')
X,y,ages,locations = load_data()
ref = pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl','rb'))
a=np.asarray(ref['unique_ages']).ravel(); p=np.asarray(ref['posterior']).ravel()
keep=np.isin(a,ages[locations]); a,p=a[keep],p[keep]; p=p/p.sum()
for family in ('l1','l2'):
    print(family.upper())
    fig, axes = plt.subplots(5,4,figsize=(12,10),sharex=True)
    for i,strength in enumerate((1e-8,1e-7,1e-6,1e-5,1e-4)):
        for j,it in enumerate((1,3,10,30)):
            row=summary[(summary.penalty==family)&(summary.strength==strength)&(summary.iteration==it)].iloc[0]
            rec=pickle.load((root/(row['case']+'_score.pkl')).open('rb'))
            ax=axes[i,j]; ax.plot(a,p,color='0.7',lw=1,label='Archived')
            ax.plot(ages[locations],rec['conditional'],color=colors[family],lw=1.3,label='Exact match')
            ax.axvline(55,color='0.6',ls=':',lw=.7)
            ax.set_title(fr'$\lambda={strength:g}$, {it} updates',fontsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
            if j==0:ax.set_ylabel('Probability')
            if i==4:ax.set_xlabel('Age')
    axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()

print('Independent numerical checks (not independent-data validation)')
for row in independent.itertuples():
    primary=pickle.load((root/(row.case+'_score.pkl')).open('rb'))
    check=pickle.load((root/(row.case+'_independent_score.pkl')).open('rb'))
    print(f'{row.penalty.upper()}, lambda={row.strength:g}, {row.iteration} updates: '
          f'P(1)={primary["P_1"]:.4g} / {check["P_1"]:.4g}; '
          f'moment residual={row.moment_residual:.4g}; max fit KKT={row.fit_kkt_max:.3g}')
    fig,ax=plt.subplots(figsize=(6,2.8))
    ax.plot(a,p,color='0.7',label='Archived')
    ax.plot(ages[locations],primary['conditional'],color='black',label='512 draws')
    ax.plot(ages[locations],check['conditional'],color='#4178a8',label='1,024 draws')
    ax.axvline(55,color='0.6',ls=':',lw=.7)
    ax.set_xlabel('Age');ax.set_ylabel('Probability');ax.legend(frameon=False,fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(3));fig.tight_layout();plt.show()
'''
    ls = '''print('Separate alternative: least-squares auxiliary fit, no intercept')
ls_table = pd.read_csv(output/'denser_penalty_grid'/'summary_with_reference.csv')
display(ls_table[ls_table['case'].str.contains('least_squares')]
        if 'case' in ls_table else ls_table)
'''
    # The existing LS comparison remains in the original heuristic notebook;
    # avoid assuming a schema for its independently generated summary tables.
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_code_cell(code), nbf.v4.new_code_cell(run), nbf.v4.new_code_cell(plots),
        nbf.v4.new_markdown_cell('''## Least-squares auxiliary alternative

The complete no-intercept least-squares auxiliary comparison, including small
logistic posterior penalties and independent numerical checks, is retained in
**Myocardial Infarction Experiment - heuristic exploration.ipynb**. It is separate
from the exact logistic/logistic matching requirement shown above.''')])
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(nb, PATH)
    print(PATH)


if __name__ == '__main__':build()
