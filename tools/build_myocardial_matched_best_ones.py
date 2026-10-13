"""Filter completed paired checks to exactly matched no-L1 logistic fits."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - matched logistic best ones.ipynb')


def build():
    old=nbf.read('Myocardial Infarction Experiment - best ones.ipynb',4)
    intro=r'''# Best matched logistic/logistic fits — existing results

This notebook **loads completed experiments only**. It performs no fitting or
new experiment and does not include the new intercept-enabled sweep while it runs.

Eligibility is deliberately strict: logistic loss in both stages, and either
**the identical L2 strength** in both convergence-seeking fits, or **the identical
optimizer and iteration budget** in both finite-iteration fits. No LS auxiliary,
no L1 penalty, and no mismatched fitting procedures are admitted. Intercept choice
must match too. The completed candidates entering this top ten all have **no
intercept**, Delta=1, and equal prior probability for zero/one change.

Rank by mean total-variation distance to the original conditional one-change curve
across two numerical checks. No P(1)>0.5 cutoff is imposed. This is a relative
shape ranking, **not a claim that these ten are good approximations**. Smaller TV
is closer. Numerical ranges are not confidence intervals.

Rows 1–6 are fully solved matched-L2 fits with 30 rho updates and identity rho
initialization. Rows 7–10 are unpenalized, 30-step FISTA in both stages, three rho
updates, and the initialization shown in the table. FISTA has zero coefficient
start and a step ceiling equal to the reciprocal exact global weighted logistic
curvature bound. Finite fitting iterations and rho updates are different quantities.
All likelihood/count-prior conventions remain those in the saved experiments.

Important limitations: the L2 curves remain diffuse, whereas the finite-FISTA
curves are excessively concentrated and some vary substantially with numerical
integration. Moment-equation residuals are shown rather than hidden. The existing
state equations remain heuristic for penalized or early-stopped coefficient fits.
'''
    source=Path('tools/myocardial_best_ones.py').read_text()
    execute='''all_ranked,curves,reference=load_candidates()
eligible=(all_ranked.auxiliary.str.startswith('Logistic ')
          & (all_ranked.auxiliary==all_ranked.posterior)
          & ~all_ranked.auxiliary.str.contains('L1',regex=False)
          & all_ranked.auxiliary.str.contains('L2|FISTA|L-BFGS|GD',regex=True))
ranked=all_ranked[eligible].copy().rename(columns={'rank':'overall_rank'}).reset_index(drop=True)
ranked.insert(0,'rank',np.arange(1,len(ranked)+1))
best=ranked.head(10).copy()
assert len(best)==10
assert (best.auxiliary==best.posterior).all()
assert not best.auxiliary.str.contains('L1|^LS$',regex=True).any()
assert not best.intercept.any(), 'Update explanatory text if new intercept results are added.'
print(f'{len(ranked)} distinct matched no-L1 logistic configurations have two completed numerical checks.')
table=summary_table(best)
table['Rho initialization']=best.initialization
display(table)
report_digest=hashlib.sha256(Path('tools/build_myocardial_matched_best_ones.py').read_bytes()).hexdigest()[:12]
report_root=Path('results')/('myocardial_matched_logistic_best_ones_'+report_digest)
report_root.mkdir(exist_ok=True)
ranked.to_csv(report_root/'all_matched_candidates.csv',index=False)
best.to_csv(report_root/'top_ten.csv',index=False)
'''
    overview=old.cells[3].source.replace('sharex=True,sharey=True','sharex=True,sharey=False')
    overview=overview.replace("print('Top ten conditional posteriors: common axes; zero mass shown outside each candidate support')",
        "print('Top ten matched fits: common age axes, separate y-axis scales; compare each curve with its gray reference')")
    overview=overview.replace('    ax.set_ylim(0,upper)','')
    tradeoff=old.cells[4].source.replace("color='#c69c22' if row.intercept else 'black'",
        "color='#c69c22' if 'L2' in row.posterior else 'black'")
    tradeoff=tradeoff.replace("label='No intercept'","label='Finite-iteration logistic'").replace("label='Uses intercept'","label='Full matched L2'")
    details=old.cells[5].source
    conclusions='''print('Interpretation of the matched-only top ten')
print('The best checked shape is matched L2 approximately 4.22e-5, 30 rho updates, identity start: '
      'P(1) approximately 0.511–0.521, but mean TV approximately 0.649 and a diffuse age-66–68 mode.')
print('Matched FISTA can have an age-81 mode and P(1) near one, but its shape is much too concentrated; '
      'a matching peak alone does not reproduce the original distribution.')
print('Numerical TV below quantifies disagreement between the two integration checks, not sampling uncertainty.')
display(best[['rank','initialization','moment_residual','numerical_curve_TV','primary_checkpoint','check_checkpoint']])
print('Initialization decoding: identity = I; scale10_corr0 = 10 I; scale0.1_corr0 = 0.1 I; '
      'scale1_corr-0.5 = [[1,-0.5],[-0.5,1]].')
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(overview),nbf.v4.new_code_cell(tradeoff),
        nbf.v4.new_code_cell(details),nbf.v4.new_code_cell(conclusions)])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'notebook_cell','exec')
    nbf.write(nb,PATH);print(PATH)


if __name__=='__main__':build()
