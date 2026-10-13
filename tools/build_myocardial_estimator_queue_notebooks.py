"""Three visible sequential experiments with automatic top-20 handoff."""
from pathlib import Path
import nbformat as nb

NAMES=[
    'Myocardial Infarction Experiment - fixed point parameter search.ipynb',
    'Myocardial Infarction Experiment - whole coefficient Gram top 20.ipynb',
    'Myocardial Infarction Experiment - segmented coefficient Gram top 20.ipynb']

COMMON=r'''import matplotlib.pyplot as plt
from IPython.display import display,Markdown
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
summary=pd.read_csv(root/'summary.csv');ranked=pd.read_csv(root/'ranked.csv')
print('All tested outcomes:');display(summary)
print('Top20:');display(ranked.head(20).style.hide(axis='index').format(precision=5))
print('Errors/non-evaluable settings:');display(json.loads((root/'errors.json').read_text()))
target=pickle.load((BASE/'scenario_100_100_q10_seed123.pkl').open('rb'))
def load_result(path,ident):
    saved=pickle.load(open(path,'rb'))
    return next(r for r in saved['results'] if r['id']==ident) if 'results' in saved else saved
for row in ranked.head(20).itertuples():
    a=load_result(row.checkpoint,row.id);b=load_result(row.check_checkpoint,row.id)
    display(Markdown(f'### #{row.rank}: {row.id}'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for ax,r,label in zip(axes,(a,b),('Main calculation','Numerical recheck')):
        ax.plot(target['ages'],target['conditional'],color='.6',lw=1.8,label='Earlier rank3 target')
        ax.plot(r['ages'],r['conditional'],color='black',lw=1.5,label=f"P(1)={r['P_1']:.4f}")
        ax.set(xlabel='Age',ylabel='Conditional probability',title=label)
        ax.legend(frameon=False,fontsize=9)
    fig.tight_layout();plt.show()
'''


def write(name,description,source,execute,extra=''):
    cells=[nb.v4.new_markdown_cell(description),nb.v4.new_code_cell(source),
        nb.v4.new_code_cell(execute),nb.v4.new_code_cell(COMMON)]
    if extra:cells.append(nb.v4.new_code_cell(extra))
    out=nb.v4.new_notebook(cells=cells)
    out.metadata['kernelspec']=dict(name='python3',language='python',display_name='Python 3')
    for c in cells:
        if c.cell_type=='code':compile(c.source,name,'exec')
    nb.validate(out);nb.write(out,name);print(name)


def build():
    write(NAMES[0],r'''# Genuine fixed-point parameter search — no covariance mixtures

Use the original rho update, identity initialization, damping .25, covariance
eigenvalue bounds [1e-4,100] and pseudoinverse cutoff1e-6. **No Gram replacement,
manual correlation, covariance mixture, or direct optimization.**

Delta=1,2,5,10,20; with/without intercept. Auxiliary fits: exact LS, or unpenalized
logistic FISTA (8,12,16,20,30,50), L-BFGS (3,5,10,20), GD (8,16,30,50).
Save rho after 1,3,5,8,12,16,20,24,30,40 updates: 150 paths, 1500 snapshots.
Logistic auxiliary/posterior fits are either exactly matched or use FISTA16 as an
explicitly labeled alternative; LS auxiliary pairs with all logistic fit settings.
Pilot selection uses the auxiliary loss and includes the null. No truth is used.

Record every snapshot, raw-update eigenvalues, number of clipped eigenvalues,
and moment residual. Evaluate posterior curves for snapshots with absolute
correlation at least .001, plus the old rank3 control. This skips structurally
diagonal cases without pretending correlation validates an estimate. Screening
uses 512 integration draws. Final shortlisted posteriors use 1024 draws; a second
check reruns the **entire rho trajectory** with 1024 draws/seed42. The primary
trajectory used 512/seed123. Numerical sensitivity therefore includes estimation
and integration changes, not just likelihood evaluation at fixed rho.

Target: the earlier rank3 conditional curve. Shortlist the closest10 overall,
closest10 with absolute correlation>=.05, and leaders of the no-intercept and
matched-fit subsets. Display all shortlist results and top20 plots; mean TV
ranking has no P(1) bonus. Both main and rechecked correlations remain visible.
All settings/outcomes are retained. These are outcome-guided exploratory analyses,
not a proof of estimator accuracy or posterior calibration. 36 parallel jobs.
''',Path('tools/myocardial_fixedpoint_search.py').read_text().split("if __name__=='__main__':")[0],
        'root=run()\nprint(root)')
    source=Path('tools/myocardial_gram_comparison.py').read_text().split("if __name__=='__main__':")[0]
    description=r'''
Use the coefficient Gram matrix **directly**, divided by full-data n=1692 or
p=110. No mixtures, trace matching, eigenvalue clipping or artificial jitter.
Only slope coefficients enter B; fitted intercepts do not enter the Gram.
The distinction between /n and /p is explicit: they differ by n/p≈15.38.

Delta=1,2,5; both intercept choices; LS or logistic auxiliary loss. Unpenalized
posterior logistic fits: FISTA/L-BFGS/GD, budgets8,16,20,30,50,100. Unpenalized
LS auxiliary fits are exact (not artificially stopped). Penalized fits: matched
L1 or L2 family and strength1e-8,1e-6,1e-5,1e-4,1e-3. Squared loss is residual²,
logistic is log(1+exp(z))-yz; both are weighted means plus the declared penalty.
Intercepts are unpenalized. Logistic/logistic uses the identical fit in both
stages; LS/logistic matches penalty family and numerical strength, not loss.
Convergence-seeking penalized fits report KKT diagnostics; finite budgets apply
only to the unpenalized logistic cases. No raw coefficients are assumed noiseless.

336 model settings × two Gram divisors = **672 comparisons**. Screening uses
512 draws/seed123; finalists use 1024 draws with seeds123 and42. Select the best20
conditional shapes plus leaders within loss/penalty/divisor groups for checks;
rank checked results by mean TV to the earlier rank3 target, without a P(1) cutoff.
36 jobs. Every failure or undefined Gram is reported; no silent repair. State
equations remain heuristic for finite/penalized fits and plug-in Gram estimates.
Numerical checks are not confidence intervals or statistical calibration.
'''
    write(NAMES[1],'# Whole-data coefficient Gram study\n'+description+
        '\nB contains the two original sample-weighted auxiliary regression vectors.\n',source,
        "root=run('whole')\nprint(root)")
    extra=r'''paired_path=root/'paired_whole_vs_segmented.csv'
if paired_path.exists():
    paired=pd.read_csv(paired_path)
    paired['TV_improvement_from_segmenting']=paired.TV_to_rank3_whole-paired.TV_to_rank3_segmented
    print('Paired comparison, same model configuration and normalization (screening results):')
    display(paired[['id','denominator_whole','P_1_whole','P_1_segmented','correlation_whole',
        'correlation_segmented','TV_to_rank3_whole','TV_to_rank3_segmented','TV_improvement_from_segmenting']])
    fig,ax=plt.subplots(figsize=(5,4))
    for divisor,g in paired.groupby('denominator_whole'):
        ax.scatter(g.TV_to_rank3_whole,g.TV_to_rank3_segmented,s=12,alpha=.5,label='/'+divisor)
    ax.plot([0,1],[0,1],color='.6',ls=':')
    ax.set(xlabel='Whole-data Gram: TV to rank3',ylabel='Segment-refit Gram: TV to rank3')
    ax.legend(frameon=False);fig.tight_layout();plt.show()
'''
    write(NAMES[2],'# Estimated-segment coefficient Gram study\n'+description+r'''
First select a pilot split using the original auxiliary fits and auxiliary loss.
Then fit that same loss/penalty/solver separately on the two **estimated** segments
with unit within-segment weights. Stack those slope vectors into Bseg and divide
Bseg-transpose-Bseg by full-data n or p. Retain the global feature transformation;
do not rewhiten within segments. No true split is supplied. If the pilot selects
zero changes, the segment Gram is undefined and the case is reported as such.
Posterior regression fits and prior remain those of the paired whole-data study;
only the covariance estimator changes. A paired all-configuration comparison
with whole-data Gram follows the top20 figures.
''',source,"root=run('segmented')\nprint(root)",extra)
    out=nb.read(NAMES[2],as_version=4)
    out.cells.extend([
        nb.v4.new_markdown_cell('## Balanced-class segment refits\nOnly the segment fits use weights n_segment/(2 n_class), computed separately within each estimated segment. Pilot selection and posterior regression remain unchanged. A single-class segment is reported as non-evaluable. This is an exploratory estimator modification, not a calibration guarantee.'),
        nb.v4.new_code_cell("unbalanced_root=root\nroot=run('segmented_balanced')\nprint(root)"),
        nb.v4.new_code_cell(COMMON),nb.v4.new_code_cell(extra),
        nb.v4.new_code_cell("unbalanced=pd.read_csv(unbalanced_root/'summary.csv')\nbalanced=pd.read_csv(root/'summary.csv')\npaired=unbalanced.merge(balanced,on='id',suffixes=('_unbalanced','_balanced'))\npaired.to_csv(root/'paired_balanced_vs_unbalanced.csv',index=False)\ndisplay(paired[['id','P_1_unbalanced','P_1_balanced','TV_to_rank3_unbalanced','TV_to_rank3_balanced','correlation_unbalanced','correlation_balanced']])")])
    nb.validate(out);nb.write(out,NAMES[2])


if __name__=='__main__':build()
