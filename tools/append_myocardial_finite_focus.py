"""Append the declared focused experiment and plots, without removing outputs."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - finite iteration exploration.ipynb')
TAG='## Focused follow-up: minimum segment size and optimizer'


def append():
    nb=nbf.read(PATH,4)
    if any(c.cell_type=='markdown' and c.source.startswith(TAG) for c in nb.cells):
        print('Focused cells already present.');return
    intro=TAG+r'''

The broad grid did not clearly improve on the earlier LS/L1 candidate, so this
**post-hoc** follow-up tests $\Delta=1,2,5,10$ and LS rho updates
4, 6, 8, 12, 16, 20, 24, 30, 40, 60. FISTA fitting budgets are
2, 4, 6, 8, 12, 16, 20, 25, 40, 60, 100, 300. An alternative finite-iteration
L-BFGS-B optimizer uses budgets 1, 2, 3, 5, 10, 20, 40, 80; it starts at zero,
has no penalty or intercept, and uses memory 10, at most 20 line-search steps,
and zero gradient/function tolerances (actual termination diagnostics are retained).
Iteration numbers **are not equivalent across optimizers**.

In addition, logistic L-BFGS auxiliary fits with budgets 5, 10, 20 are evaluated
at Delta=1 and 3, 10, 30 rho updates, against every posterior-input fit. Matching
means both optimizer and budget, not just equal iteration numbers. This adds
**980 primary comparisons**, retaining all outcomes. The two closest curves and
the highest-P(1) curve within 0.10 TV of the best in each auxiliary/optimizer class
are rechecked at the same two finer numerical settings as above, with no P(1) cutoff.

Larger Delta can exclude the original curve's upper-age tail: the retained reference
mass is about 84% for Delta=5, 78% for Delta=10, 26% for Delta=85, and 4% for Delta=169.
This lost mass counts against reference similarity; it is not renormalized away.
'''
    source=Path('tools/myocardial_finite_focus.py').read_text().split("if __name__=='__main__':")[0]
    execute='''initial_root=root
focused_root=run_focus(initial_root)
root=focused_root
summary=pd.read_csv(root/'summary.csv')
validation=pd.read_csv(root/'validation.csv')
ranked=pd.read_csv(root/'ranked.csv')
display(ranked)
'''
    curves=nb.cells[4].source
    combined='''print('Combined finite-iteration shortlist: both sweeps, with numerical sensitivity')
combined=[]
for stage,folder in [('broad',initial_root),('focused',focused_root)]:
    table=pd.read_csv(folder/'ranked.csv')
    for row in table.to_dict('records'):
        recs=[pickle.load((folder/(row['case']+'_'+row['posterior_fit']+'_'+tag+'.pkl')).open('rb'))
              for tag in ('q9_seed42','q10_seed123')]
        row.update(stage=stage,Delta=recs[0]['Delta'],auxiliary=recs[0]['auxiliary'],
            auxiliary_budget=recs[0]['auxiliary_budget'],posterior_budget=recs[0]['posterior_budget'],
            rho_updates=recs[0]['rho_updates'],
            numerical_curve_TV=.5*np.abs(recs[0]['conditional']-recs[1]['conditional']).sum(),
            min_interval90_width=min(r['interval90_upper']-r['interval90_lower'] for r in recs),
            max_interval90_width=max(r['interval90_upper']-r['interval90_lower'] for r in recs))
        combined.append(row)
combined=pd.DataFrame(combined).sort_values('mean_TV')
combined.to_csv(initial_root/'combined_checked_candidates.csv',index=False)
display(combined)
print('These are exploratory comparisons, not independently validated model choices.')
'''
    nb.cells.extend([nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(curves),nbf.v4.new_code_cell(combined)])
    nbf.write(nb,PATH);print('Appended focused experiment and inline plots.')


if __name__=='__main__':append()
