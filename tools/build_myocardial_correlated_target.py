from pathlib import Path
import nbformat as nb

PATH=Path('Myocardial Infarction Experiment - correlated rank 3 candidates.ipynb')
source=Path('tools/myocardial_correlated_target.py').read_text().split("if __name__=='__main__':")[0]
cells=[nb.v4.new_markdown_cell(r'''# Nonzero-correlation candidates closest to the previous rank-3 experiment

**Exploratory sensitivity study—not a claim of improved covariance estimation.**
Fix the old rank-3 fits: LS auxiliary, 16-step unpenalized logistic FISTA posterior,
no intercept, Delta2, same data, weights and half/half count prior. Use 1024 draws,
seed123, and recheck shortlisted candidates with 1024 draws/seed42. Posterior
state max100 evaluations, likelihood quadrature160, up to36 parallel jobs.

Construct three covariance directions: the weighted LS coefficient Gram matrix;
the Gram matrix of LS coefficients fitted separately on either side of the
data-estimated pilot split; and the direct moment-fit matrix for old rank3.
All coefficient Grams are B-transpose-B/n, without estimation-noise correction.
They are data-informed but may be biased. No ground-truth split or signal is used.

For each direction, test convex mixtures with the old diagonal rho, either at its
raw scale or rescaled to the same trace as old rho (explicitly separates shape
from magnitude). Trace-matched weights: .01,.03,.06,.1,.2,.35,.5,.75,1.
Raw-scale weights: .001,.003,.01,.03,.06,.1,.2,.5,1. Include the exact old baseline:
**55 total candidates**. These paths inherit the old baseline's estimation
limitations; in particular trace matching is not independent signal-strength
estimation. All matrices and settings are saved.

Rank conditional shape similarity to **rank3**, not to the older original curve.
Require absolute correlation at least .05 for the shortlist, avoiding a vacuous
near-zero perturbation win. No P(1) cutoff. Recheck the top10 and at least two
per covariance family. Report correlation, rho entries, P(1), moment residuals,
and numerical sensitivity. Every candidate is retained, not only favorable ones.
Tuning covariance shrinkage to a desired posterior is outcome-guided exploration,
not calibration, independent validation, or evidence that the covariance is true.
'''),nb.v4.new_code_cell(source),nb.v4.new_code_cell('''from IPython.display import display,Markdown
root=run()
summary=pd.read_csv(root/'summary.csv');ranked=pd.read_csv(root/'ranked.csv')
print('All candidates:');display(summary)
print('Best candidates with absolute correlation >=0.05:')
display(ranked[['rank','family','trace_matched','weight','rho_11','rho_12','rho_22','correlation',
    'P1_min','P1_max','mean_TV_to_rank3','numerical_TV','relative_residual']].style.format(precision=6))
print('Errors:',json.loads((root/'errors.json').read_text()))
'''),nb.v4.new_code_cell('''import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
targets={seed:pickle.load((root/f'baseline_seed{seed}.pkl').open('rb')) for seed in (123,42)}
for row in ranked.head(10).itertuples():
    first=pickle.load(open(row.checkpoint,'rb'));second=pickle.load(open(row.check_checkpoint,'rb'))
    display(Markdown(f'### #{row.rank}: {row.family}; weight={row.weight:g}; trace matched={row.trace_matched}\\n\\n'
        f'Correlation={row.correlation:.4f}; relative moment residual={row.relative_residual:.3f}; '
        f'P(1)={row.P1_min:.4f}–{row.P1_max:.4f}.'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for ax,seed,r in zip(axes,(123,42),(first,second)):
        t=targets[seed]
        ax.plot(t['ages'],t['conditional'],color='.55',lw=1.8,label=f'Rank3 target: P(1)={t["P_1"]:.4f}')
        ax.plot(r['ages'],r['conditional'],color='black',lw=1.5,label=f'Correlated: P(1)={r["P_1"]:.4f}')
        ax.set(xlabel='Age',ylabel='Conditional probability',title=f'1024 draws, seed {seed}')
        ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();plt.show()
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for family,g in summary[summary.family!='baseline'].groupby('family'):
    axes[0].scatter(g.correlation,g.TV_to_rank3,s=20,label=family)
    axes[1].scatter(g.TV_to_rank3,g.P_1,s=20,label=family)
axes[0].set(xlabel='Correlation',ylabel='TV distance from rank3')
axes[1].set(xlabel='TV distance from rank3',ylabel='P(1 change point)')
axes[1].axhline(targets[123]['P_1'],color='.5',ls=':')
axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()
''')]
n=nb.v4.new_notebook(cells=cells);n.metadata['kernelspec']=dict(name='python3',language='python',display_name='Python 3')
for c in cells:
    if c.cell_type=='code':compile(c.source,str(PATH),'exec')
nb.validate(n);nb.write(n,PATH);print(PATH)
