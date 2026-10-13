"""Create a visible rank-3 reproduction using the original final plot style."""
from pathlib import Path
import nbformat as nb

PATH=Path('Myocardial Infarction Experiment - final redo.ipynb')
original=nb.read('Myocardial Infarction Experiment - rho sensitivity final.ipynb',4)
stem_style=original.cells[4].source.split('fig, ax = plot_posterior_compact(ages, saved_weights)')[0]
sensitivity_style=original.cells[12].source.split("plot_sensitivity('reference_weights'")[0]
sensitivity_style=sensitivity_style.replace('plt.subplots(3, 3, figsize=(13, 7)', 'plt.subplots(5, 5, figsize=(13*5/3, 7*5/3)').replace('if row == 2:', 'if row == 4:')
source=Path('tools/myocardial_final_redo.py').read_text()
cells=[nb.v4.new_markdown_cell(r'''# Myocardial infarction — final redo (updated rank 3)

This notebook **freshly recomputes the rank-3 experiment** from the raw data,
not from archived fitted coefficients. Uses LS auxiliary fits to estimate rho,
then unpenalized logistic FISTA stopped at **16 iterations** for the posterior;
**no intercept in either stage**. This is intentionally not matched logistic/logistic.

Data: UCI myocardial infarction, binary ZSN>0; remove eight missing-age rows,
sort by age; 110 non-age predictors, mean imputation, centering, covariance-ridge
whitening (1e-6), scale by 1/sqrt(n); n=1692. Delta=2 observations at either end.
The prior is **P(0 chgpts)=P(1 chgpt)=1/2** with uniform location conditional on
one change on the admissible distinct-age grid. The single null configuration
uses signal 0 throughout. Fitting weights are marginals of this same prior.

The LS pilot minimizes split squared error on this grid (including the null).
Starting rho=identity, take **20 damped moment updates**, damping .25,
pseudoinverse cutoff 1e-6, eigenvalue clipping [1e-4,100]. Fits start from zero,
using the package's global-curvature-bounded FISTA step. No explicit penalty.
The residual is reported, not forced to zero.

Posterior states: 512 Sobol draws/seed42 and 1,024/seed123, max100 state-solver
evaluations, order160 likelihood quadrature, 36 parallel jobs. Fresh rho, fitted
scores and baseline posterior must reproduce both saved rank-3 checks.
The main black-stem figure uses **512 draws/seed42**, as did the rank-3 primary
curve. Numerical ranges are not confidence intervals.

Sensitivity: independently scale the two ascending eigenvalues of rho by
.8,.9,1,1.1,1.2 (25 cases), holding eigenvectors, data, fits, prior and all numerical
settings fixed. Re-solve every candidate-specific state, using matched random
draws across perturbations. Repeat for both numerical checks. **Conditional
one-change curves sum to one; P(1) includes the null in normalization.**

This is a post-hoc selected heuristic, not a claim of calibrated Bayesian inference.
No attempt is made to force P(1)>1/2. Only plotting style—not the old inconsistent
normalization—is copied from the original final notebook. Checkpoints are unique
and resumable; the original notebooks/results are untouched.
'''),nb.v4.new_code_cell(source),
nb.v4.new_markdown_cell('## 1. Fresh auxiliary fits and signal-strength estimate'),
nb.v4.new_code_cell('''from IPython.display import display
data=reconstruct()
print('rho_hat:');display(pd.DataFrame(data['rho']))
print('Eigenvalues:',np.linalg.eigvalsh(data['rho']))
print('Pilot split observation index:',data['pilot'],'first age after split:',data['ages'][data['pilot']] if data['pilot']>=0 else '0 chgpts')
print('Moment-equation residual:',data['trajectory'][-1]['moment_residual'])
print('Posterior fitting diagnostics:');display(pd.DataFrame(data['fit']['checks']))
'''),nb.v4.new_markdown_cell('## 2. Probability of 0 versus 1 change point'),
nb.v4.new_code_cell('''records=compute_scenarios(data,[(1.,1.)])
baseline=records[((1.,1.),9,42)]
baseline_check=records[((1.,1.),10,123)]
probability_table=pd.DataFrame([{'Draws':2**power,'Seed':seed,'P(0 chgpts)':r['absence'],
    'P(1 chgpt)':r['presence'],'Max. state residual':r['residuals'].max()}
    for (_,power,seed),r in records.items()])
display(probability_table.style.format({'P(0 chgpts)':'{:.6f}','P(1 chgpt)':'{:.6f}','Max. state residual':'{:.3g}'}))
probability_table.to_csv(data['root']/'baseline_probabilities.csv',index=False)
print('Fresh baseline reproduces both saved numerical checks.')
'''),nb.v4.new_markdown_cell('## 3. Posterior location conditional on 1 change point\nOriginal final notebook stem-plot formatting, applied to normalized conditional probabilities.'),
nb.v4.new_code_cell(stem_style+"\nfig,ax=plot_posterior_compact(baseline['ages'],baseline['conditional'])\nplt.show()"),
nb.v4.new_markdown_cell('## 4. Paired ±10% and ±20% eigenvalue sensitivity\nThe baseline is unchanged. Only the eigenvalues are perturbed; all location-specific states are recomputed.'),
nb.v4.new_code_cell('''remaining=[levels for levels in product(LEVELS,repeat=2) if levels!=(1.,1.)]
records.update(compute_scenarios(data,remaining))
rows=[]
for (levels,power,seed),r in records.items():
    base=records[((1.,1.),power,seed)]
    rows.append({'lambda_1':f'{100*(levels[0]-1):+.0f}%'.replace('+0%','0%'),
        'lambda_2':f'{100*(levels[1]-1):+.0f}%'.replace('+0%','0%'),
        'Draws':2**power,'Seed':seed,'P(0 chgpts)':r['absence'],'P(1 chgpt)':r['presence'],
        'Conditional peak age':r['ages'][np.argmax(r['conditional'])],
        'Conditional mass above 55':r['conditional'][r['ages']>55].sum(),
        'Conditional TV vs baseline':.5*np.abs(r['conditional']-base['conditional']).sum(),
        'Max. state residual':r['residuals'].max()})
summary=pd.DataFrame(rows).sort_values(['Draws','lambda_1','lambda_2'])
np.testing.assert_allclose(summary['P(0 chgpts)']+summary['P(1 chgpt)'],1,atol=1e-12)
summary.to_csv(data['root']/'sensitivity_summary.csv',index=False)
level_order=['-20%','-10%','0%','+10%','+20%']
for power,seed in CHECKS:
    print(f'{2**power} draws, seed {seed}: P(1 chgpt)')
    table=summary[summary.Draws==2**power].pivot(index='lambda_1',columns='lambda_2',values='P(1 chgpt)').reindex(index=level_order,columns=level_order)
    display(table.style.format('{:.6f}'))
print('All sensitivity and numerical diagnostics:');display(summary)
'''),nb.v4.new_markdown_cell('## 5. Conditional posterior sensitivity — original panel format, expanded to 5×5'),
nb.v4.new_code_cell(sensitivity_style+'''
ages=baseline['ages']
for power,seed in CHECKS:
    print(f'Conditional on 1 change point: {2**power} draws, seed {seed}')
    scenarios={(level_order[i],level_order[j]):records[((u,v),power,seed)]
        for i,u in enumerate(LEVELS) for j,v in enumerate(LEVELS)}
    plot_sensitivity('conditional','Conditional posterior probability')
print('Completed fresh rank-3 reproduction and sensitivity analysis.')
''')]
from tools.myocardial_final_redo_diagnostics import diagnostic_cells
cells.extend(diagnostic_cells())
from tools.myocardial_final_redo_1024 import only_1024
cells=only_1024(cells)
from tools.myocardial_final_redo_plot_spacing import match_original_spacing
cells=match_original_spacing(cells)
notebook=nb.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec']=dict(name='python3',language='python',display_name='Python 3')
nb.validate(notebook)
for cell in cells:
    if cell.cell_type=='code':compile(cell.source,str(PATH),'exec')
nb.write(notebook,PATH)
print(PATH)
