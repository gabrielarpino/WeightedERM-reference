"""Add and execute transparent numerical diagnostics without rerunning experiments."""
from pathlib import Path
import nbformat as nb

NOTEBOOK=Path('Myocardial Infarction Experiment - final redo.ipynb')
ROOT=Path('results/myocardial_final_redo_e3d965cc4b9b')
MARKER='final-redo-accuracy-diagnostics'


def diagnostic_cells():
    cells=[nb.v4.new_markdown_cell(r'''## Explicit accuracy diagnostics: rho estimation versus posterior numerics

Three distinct quantities must not be conflated: **(i)** the signal-strength
moment-equation mismatch; **(ii)** numerical accuracy of posterior evaluation at
the chosen rho; **(iii)** statistical accuracy/calibration, which is not measured
by either diagnostic. The calculations below use saved fits/states; only the
one-dimensional likelihood quadrature is reevaluated at a finer order.
'''),nb.v4.new_code_cell(r'''
import numpy as np
import pandas as pd
import pickle
from IPython.display import display, Markdown
from tools.myocardial_final_redo import (empirical_adjustment, pilot,
    squared_logistic_state, gaussian_logistic_loglikelihood, normalize_posterior)
diagnostic_root=data['root']
X,y,phi=data['X'],data['y'],data['phi']
n,p=X.shape
B_aux=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
theta_aux=X@B_aux
V_aux,_=empirical_adjustment(theta_aux,y,phi,n/p,loss='squared')
_,psi_aux=pilot(theta_aux,y,data['eta'],'squared')
observed_moment=V_aux.T@V_aux/n
predicted_moment=squared_logistic_state(data['rho'],phi,psi_aux,n/p)['moment']
rho_residual=np.linalg.norm(predicted_moment-observed_moment,'fro')
relative_rho_residual=rho_residual/np.linalg.norm(observed_moment,'fro')
np.testing.assert_allclose(rho_residual,data['trajectory'][-1]['moment_residual'],rtol=1e-10)
print('Inferred rho_hat (20 updates):')
display(pd.DataFrame(data['rho'],index=['Signal 1','Signal 2'],columns=['Signal 1','Signal 2']).style.format('{:.12f}'))
rho_diagnostics=pd.DataFrame([{
    'Absolute moment residual (Frobenius norm)':rho_residual,
    'Observed moment Frobenius norm':np.linalg.norm(observed_moment,'fro'),
    'Relative moment residual':relative_rho_residual,
    'Last rho update Frobenius norm':np.linalg.norm(data['trajectory'][-1]['rho']-data['trajectory'][-2]['rho']),
    'Smallest rho eigenvalue':np.linalg.eigvalsh(data['rho'])[0]}])
display(rho_diagnostics.T.rename(columns={0:'Value'}).style.format('{:.9g}'))
print('Observed and model-predicted second moments:')
display(pd.DataFrame(observed_moment));display(pd.DataFrame(predicted_moment))
display(Markdown(
    r'The diagnostic is $|\Lambda^T\hat\rho\Lambda+K-S\|_F$, with $S=V_{aux}^TV_{aux}/n$. '
    f'Its value is **{rho_residual:.6f}**, or **{100*relative_rho_residual:.2f}%** of '+
    r'$\|S\|_F$. This is **not** $\|\hat\rho-\rho_{true}\|$, a confidence interval, or a percentage accuracy of rho. '+
    'The true signal strength is unknown. This is a deliberately finite-iteration, constrained heuristic—not a certified moment-equation solution.'))
print('Last five rho updates (moment residual):')
display(pd.DataFrame([{'Update':r['iteration'],'Moment residual':r['moment_residual']} for r in data['trajectory'][-5:]]))
display(Markdown('Both estimated eigenvalues are equal. The ±10%/±20% analysis holds the coordinate eigenbasis fixed; separate eigenvalue perturbations are basis-dependent at this repeated eigenvalue.'))
rho_diagnostics.to_csv(diagnostic_root/'rho_accuracy_diagnostics.csv',index=False)
'''),nb.v4.new_code_cell(r'''
checks=[pickle.load((diagnostic_root/f'scenario_100_100_q{power}_seed{seed}.pkl').open('rb'))
        for power,seed in ((9,42),(10,123))]
a,b=checks
numerical_diagnostics=pd.DataFrame([{
    'P(1), 512 draws / seed 42':a['presence'],
    'P(1), 1024 draws / seed 123':b['presence'],
    'Absolute difference in P(1)':abs(a['presence']-b['presence']),
    'Conditional posterior TV difference':.5*np.abs(a['conditional']-b['conditional']).sum(),
    'Largest age-specific probability difference':np.max(np.abs(a['conditional']-b['conditional'])),
    'Full posterior TV difference (including null)':.5*np.abs(a['probabilities']-b['probabilities']).sum(),
    'Largest conditional CDF difference':np.max(np.abs(np.cumsum(a['conditional'])-np.cumsum(b['conditional']))),
    'Max sampled-state equation residual, 512 draws':a['residuals'].max(),
    'Max sampled-state equation residual, 1024 draws':b['residuals'].max()}])
display(numerical_diagnostics.T.rename(columns={0:'Value'}).style.format('{:.9g}'))
print('Baseline maximum-a-posteriori ages:',*[r['ages'][np.argmax(r['conditional'])] for r in checks])
display(Markdown(
    f'**Empirical numerical sensitivity:** P(1) changes by **{100*abs(a["presence"]-b["presence"]):.2f} percentage points**; '
    f'the conditional curves differ by **{.5*np.abs(a["conditional"]-b["conditional"]).sum():.4f} TV**. '
    'These compare two scrambled-QMC resolutions/seeds; they are **not rigorous error bounds, standard errors, or confidence intervals**. '
    'The approximately 1e-9 state residuals measure agreement with the sampled equations, not the exact population equations, posterior calibration, or error from the estimated rho.'))
numerical_diagnostics.to_csv(diagnostic_root/'posterior_numerical_diagnostics.csv',index=False)
'''),nb.v4.new_code_cell(r'''
# Quadrature-only check: freeze each saved state and double the Gaussian
# quadrature order from 160 to 320. This does NOT refit or re-estimate rho.
quadrature_rows=[]
for (power,seed),baseline_result in zip(((9,42),(10,123)),checks):
    likelihoods=[]
    for index,e in enumerate(data['eta']):
        state=pickle.load((diagnostic_root/f'eig100_100_q{power}_seed{seed}_location{index}.pkl').open('rb'))['state']
        psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
        likelihoods.append(gaussian_logistic_loglikelihood(data['fit']['V'],y,data['rho'],
            state['lambda'],state['kappa'],psi,quadrature_order=320))
    refined=normalize_posterior(likelihoods,data['prior'])
    quadrature_rows.append({'QMC draws':2**power,'Seed':seed,
        'P(1), quadrature 160':baseline_result['presence'],
        'P(1), quadrature 320':refined['presence'],
        'Absolute P(1) difference':abs(refined['presence']-baseline_result['presence']),
        'Conditional TV difference':.5*np.abs(refined['conditional']-baseline_result['conditional']).sum(),
        'Max absolute log-likelihood difference':np.max(np.abs(np.asarray(likelihoods)-baseline_result['log_likelihood']))})
quadrature_diagnostics=pd.DataFrame(quadrature_rows)
print('Independent error-source check: likelihood quadrature order 160 versus 320')
display(quadrature_diagnostics.style.format(precision=10))
quadrature_diagnostics.to_csv(diagnostic_root/'likelihood_quadrature_diagnostics.csv',index=False)
display(Markdown(
    '**Bottom line:** report the moment residual and numerical differences separately. '+
    'The conditional posterior has a reproducible broad shape across the two QMC checks, but numerical uncertainty in P(1) is about two percentage points in this comparison. '+
    'The much larger changes under rho perturbation in the sensitivity section are a separate limitation. '+
    'These diagnostics support transparency and numerical reproducibility, **not a claim of accurate signal-strength estimation or calibrated change-point probabilities**.'))
''')]
    for i,cell in enumerate(cells):cell['id']=f'{MARKER}-{i}'
    return cells


def enrich():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    cells=diagnostic_cells()
    setup=nb.v4.new_code_cell(f'import pickle\nfrom pathlib import Path\ndata=pickle.load(Path({str(ROOT/"fresh_fit_and_rho.pkl")!r}).open("rb"))')
    run=nb.v4.new_notebook(cells=[setup]+cells)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[str(Path.cwd()/'.venv-phase1/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(run,km=km,timeout=600,resources={'metadata':{'path':str(Path.cwd())}}).execute()
    original=nb.read(NOTEBOOK,4)
    original.cells=[c for c in original.cells if not c.id.startswith(MARKER)]
    count=max(c.execution_count or 0 for c in original.cells if c.cell_type=='code')
    for cell in run.cells[1:]:
        if cell.cell_type=='code':
            count+=1;cell.execution_count=count
            for output in cell.outputs:
                if 'execution_count' in output:output.execution_count=count
        original.cells.append(cell)
    nb.validate(original)
    temp=NOTEBOOK.with_suffix('.diagnostics.ipynb');nb.write(original,temp);temp.replace(NOTEBOOK)
    print('Accuracy diagnostics appended and executed; existing figures preserved.',flush=True)


if __name__=='__main__':enrich()
