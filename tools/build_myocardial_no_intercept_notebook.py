"""Build a visible, executable experiment and automatic top-ten report."""
from pathlib import Path
import nbformat as nb

PATH=Path('Myocardial Infarction Experiment - no intercept solver best ones.ipynb')
source=Path('tools/myocardial_no_intercept_sweep.py').read_text().split("if __name__ == '__main__':")[0]
notebook=nb.v4.new_notebook(cells=[
nb.v4.new_markdown_cell(r'''# No-intercept matched logistic/logistic: solver sensitivity

Exploratory, outcome-guided sensitivity analysis, not independently validated
model selection. All fits use **logistic loss, zero intercept, zero explicit
penalty**, and zero coefficient starts. The same fit (solver, budget, step and
random seed) is used in the auxiliary and posterior stages. The count prior
remains 1/2 for each count; location given one change is uniform on admissible
distinct-age splits. Data preprocessing and the audited joint likelihood are
unchanged. No likelihood temperature or count-prior tuning is used.

45 fitting configurations: FISTA/GD budgets 1,3,8,16,30 at step multipliers .1,1;
L-BFGS budgets 1,2,3,5,8,12,20; SGD epochs 1,3,10 at step multipliers .1,1 and
seeds 42,123; SAGA epochs 1,3,10 and seeds 42,123. Solver iterations and stochastic
epochs are different computational units. FISTA/GD steps use global weighted
curvature; SGD uses a maximum weighted per-observation curvature bound.
Weights are normalized explicitly; no hidden penalty or intercept is fitted.
Liblinear is excluded: it does not support a genuinely unpenalized objective.

Delta=1,5,10; rho starts at identity, with damping .25 and eigenvalues clipped to
[1e-4,100]. Save rho updates 1,3,6,10,20,30: **810 matched comparisons**, run on
36 jobs. Primary state solves use 256 QMC points, seed42, budget60. Selected
shape leaders overall/by solver, and shape leaders with .5<=P(1)<.99, are
rechecked with 512/1024 QMC points and seeds42/123, budget100. Likelihood
quadrature order160. The explicit .99 cutoff is a reporting preference, not a
theoretical calibration guarantee. Nothing forces such a candidate to exist.

Report the ten closest conditional shapes and, separately, up to ten meeting
the nonextreme probability preference in **both** numerical checks. All primary
results, failures, residuals and checkpoints are retained. Finite-fit state
equations remain heuristic, numerical ranges are not confidence intervals,
and a probability rounding to 1 is not necessarily exact certainty: P(0) and
log-odds are saved too. Original reference curves are not edited.
'''),
nb.v4.new_code_cell(source),
nb.v4.new_code_cell('root = run()\nprint("Results:", root)'),
nb.v4.new_code_cell('''from IPython.display import display
ranked = pd.read_csv(root/'ranked.csv')
shape = pd.read_csv(root/'top_ten_shape.csv')
preferred = pd.read_csv(root/'top_ten_nonextreme.csv')
print('Ten closest shapes, regardless of probability:')
display(shape)
print('Closest shapes with 0.5 <= P(1) < 0.99 in both numerical checks:')
display(preferred)
print('Failures (retained):', len(json.loads((root/'errors.json').read_text())))
'''),
nb.v4.new_code_cell('''import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':10,
                     'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
reference = reference_distribution()
ref_ages=np.array(sorted(reference)); ref_prob=np.array([reference[a] for a in ref_ages])
validation=pd.read_csv(root/'validation.csv')
def plot_group(frame, heading):
    print(heading)
    if frame.empty:
        print('No candidate satisfies this reporting criterion; no favorable result is substituted.')
        return
    fig, axes=plt.subplots(int(np.ceil(len(frame)/2)),2,figsize=(9,2.3*int(np.ceil(len(frame)/2))),squeeze=False)
    for ax,row in zip(axes.flat,frame.itertuples()):
        ax.plot(ref_ages,ref_prob,color='.6',label='Original')
        records=validation[validation.case==row.case].sort_values('tag')
        for color,entry in zip(('black','tab:blue'),records.itertuples()):
            rec=pickle.load(open(entry.checkpoint,'rb'))
            ax.plot(rec['ages'],rec['conditional'],color=color,linewidth=1.2,
                    label=f"P(1)={rec['P_1']:.4g}")
        ax.set_title(f'{row.solver}, budget={row.fit_budget}, Δ={row.Delta}, rho={row.rho_updates}',fontsize=9)
        ax.set_xlabel('Age');ax.set_ylabel('Conditional probability');ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(frame):]:ax.set_visible(False)
    fig.tight_layout();plt.show()
plot_group(shape,'Top ten by conditional shape')
plot_group(preferred,'Top nonextreme-probability candidates')
primary=pd.read_csv(root/'summary.csv')
fig,ax=plt.subplots(figsize=(5,3))
for solver,g in primary.groupby('solver'):
    ax.scatter(g.TV_to_reference,g.P_1,s=12,alpha=.45,label=solver)
ax.axhline(.5,color='.6',linewidth=.8);ax.axhline(.99,color='.6',linewidth=.8,linestyle=':')
ax.set(xlabel='TV distance from original conditional posterior',ylabel='P(1 chgpt)')
ax.legend(fontsize=8);fig.tight_layout();plt.show()
'''),
nb.v4.new_markdown_cell('''## Interpretation
These rankings describe sensitivity to implicit regularization, not evidence for
choosing optimization settings because they produce a desired scientific answer.
Check moment/state residuals, seed sensitivity and neighboring settings before
using a curve. A sharp or diffuse posterior is reported even if undesirable.
The separate intercept-enabled study retains its own top-ten notebook.
''')])
notebook.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
nb.write(notebook,PATH)
print(PATH)
