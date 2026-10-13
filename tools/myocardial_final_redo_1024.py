"""Present only the 1024-draw analysis, retaining archived numerical checks."""
from pathlib import Path
import nbformat as nb


def only_1024(cells):
    cells[0].source=cells[0].source.replace(
        'Posterior states: 512 Sobol draws/seed42 and 1,024/seed123, max100 state-solver\nevaluations, order160 likelihood quadrature, 36 parallel jobs. Fresh rho, fitted\nscores and baseline posterior must reproduce both saved rank-3 checks.\nThe main black-stem figure uses **512 draws/seed42**, as did the rank-3 primary\ncurve. Numerical ranges are not confidence intervals.',
        'Posterior states: **1,024 Sobol draws, seed 123**, max100 state-solver\nevaluations, order160 likelihood quadrature, 36 parallel jobs. Fresh rho, fitted\nscores and baseline posterior must reproduce the saved rank-3 result for this\nnumerical setting. All reported probabilities and plots use this setting.')
    cells[0].source=cells[0].source.replace('draws across perturbations. Repeat for both numerical checks.',
        'draws across perturbations. Use the same 1,024-draw setting throughout.')
    # Retain the original computation fingerprint/checkpoint directory. Restrict
    # evaluation and presentation, without altering the archived two-check study.
    cells[1].source=cells[1].source.replace('for power,seed in CHECKS','for power,seed in REPORT_CHECKS')
    if 'REPORT_CHECKS=((10,123),)' not in cells[1].source:
        cells[1].source+='\nREPORT_CHECKS=((10,123),)\n'
    cells[5].source=cells[5].source.replace('baseline=records[((1.,1.),9,42)]','baseline=records[((1.,1.),10,123)]')
    cells[5].source=cells[5].source.replace('baseline_check=records[((1.,1.),10,123)]\n','')
    cells[5].source=cells[5].source.replace('both saved numerical checks','the saved 1,024-draw result')
    for i in (5,9,11):
        cells[i].source=cells[i].source.replace('for power,seed in CHECKS','for power,seed in REPORT_CHECKS')
    cells[5].source=cells[5].source.replace("'baseline_probabilities.csv'","'baseline_probabilities_1024.csv'")
    cells[9].source=cells[9].source.replace("'sensitivity_summary.csv'","'sensitivity_summary_1024.csv'")
    cells[14].source=r'''
checks=[pickle.load((diagnostic_root/'scenario_100_100_q10_seed123.pkl').open('rb'))]
reported=checks[0]
numerical_diagnostics=pd.DataFrame([{
    'QMC draws':1024,'Seed':123,
    'P(0 chgpts)':reported['absence'],'P(1 chgpt)':reported['presence'],
    'Max sampled-state equation residual':reported['residuals'].max(),
    'Posterior normalization discrepancy':abs(reported['probabilities'].sum()-1),
    'Conditional normalization discrepancy':abs(reported['conditional'].sum()-1),
    'Conditional posterior mode (age)':reported['ages'][np.argmax(reported['conditional'])]}])
display(numerical_diagnostics.T.rename(columns={0:'Value'}).style.format('{:.9g}'))
display(Markdown(
    'All reported results use **1,024 scrambled-Sobol draws, seed 123**. '
    'The approximately 1e-9 state residual measures agreement with the sampled equations, '
    '**not** integration error, exact population-equation error, or statistical calibration. '
    'A single reported QMC setting does not provide a Monte Carlo standard error or confidence interval.'))
numerical_diagnostics.to_csv(diagnostic_root/'posterior_numerical_diagnostics_1024.csv',index=False)
'''
    cells[15].source=cells[15].source.replace('zip(((9,42),(10,123)),checks)','zip(((10,123),),checks)')
    cells[15].source=cells[15].source.replace("'likelihood_quadrature_diagnostics.csv'","'likelihood_quadrature_diagnostics_1024.csv'")
    cells[15].source=cells[15].source.replace(
        'The conditional posterior has a reproducible broad shape across the two QMC checks, but numerical uncertainty in P(1) is about two percentage points in this comparison. ',
        'The quadrature comparison isolates likelihood-integration sensitivity with the 1,024-draw states held fixed; it is not an overall posterior error bound. ')
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'final redo','exec')
    return cells


if __name__=='__main__':
    path=Path('Myocardial Infarction Experiment - final redo.ipynb')
    notebook=nb.read(path,4)
    notebook.cells=only_1024(notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    nb.validate(notebook)
    temp=path.with_suffix('.1024.ipynb');nb.write(notebook,temp);temp.replace(path)
    from tools import execute_myocardial_rho_final as runner
    runner.PATH=path.resolve();runner.main()
