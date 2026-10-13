"""Append and execute saved-result plots only; never rerun the experiment."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'Myocardial Infarction Experiment - whole coefficient Gram top 20.ipynb'
TAG = 'whole-gram-high-presence'
SOURCE = r'''
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root=Path('results/myocardial_gram_pair_1ce06b10b4f5/whole')
summary=pd.read_csv(root/'summary.csv')
selected=summary.loc[summary.P_1>=0.5].sort_values('TV_to_rank3').head(10).copy()
selected.insert(0,'subset_rank',range(1,len(selected)+1))
validation=pd.read_csv(root/'validation.csv')
checks=validation.groupby('id').agg(recheck_P1_min=('P_1','min'),recheck_P1_max=('P_1','max'),recheck_count=('P_1','size'))
selected=selected.merge(checks,left_on='id',right_index=True,how='left')
selected['recheck_count']=selected.recheck_count.fillna(0).astype(int)
cols=['subset_rank','id','P_1','TV_to_rank3','aux_loss','penalty','strength','method','budget','intercept','Delta','denominator','correlation','peak_age','mass_above55','recheck_count','recheck_P1_min','recheck_P1_max']
display(selected[cols].style.hide(axis='index').format(precision=5,na_rep='Not rechecked'))
target=pickle.load(open('results/myocardial_final_redo_e3d965cc4b9b/scenario_100_100_q10_seed123.pkl','rb'))
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
for row in selected.itertuples():
    result=pickle.load(open(row.checkpoint,'rb'))
    display(Markdown(f'### Higher-P(1) #{row.subset_rank}: {row.id}\n'
        f'Screening P(1)={row.P_1:.4f}; TV={row.TV_to_rank3:.4f}; correlation={row.correlation:.4f}.'))
    fig,ax=plt.subplots(figsize=(5.5,3.5))
    ax.plot(target['ages'],target['conditional'],color='.6',lw=1.8,label='Earlier rank3 target')
    ax.plot(result['ages'],result['conditional'],color='black',lw=1.5,label=f'Screening P(1)={row.P_1:.4f}')
    for check in validation.loc[validation.id==row.id].itertuples():
        checked=pickle.load(open(check.checkpoint,'rb'))
        ax.plot(checked['ages'],checked['conditional'],lw=1,ls='--',alpha=.7,label=f'Recheck seed {check.seed}: P(1)={check.P_1:.4f}')
    ax.set(xlabel='Age',ylabel='Conditional probability')
    ax.legend(frameon=False,fontsize=9);fig.tight_layout();plt.show()
'''


def main():
    notebook=nbformat.read(PATH,4)
    cells=[nbformat.v4.new_markdown_cell(
        '## Closest conditional curves among candidates with P(1) ≥ 0.5\n\n'
        'Top ten selected by smallest TV distance to the earlier rank-3 conditional curve, '
        'subject to screening P(1) ≥ 0.5. These are **512-draw screening results**, '
        'not a new numerically validated ranking. Existing 1024-draw rechecks are shown '
        'where available; missing rechecks are explicit. Selection uses the screening '
        'probability, so a recheck may cross the threshold. This outcome-guided '
        'comparison does not establish calibration or estimator accuracy.'),
        nbformat.v4.new_code_cell(SOURCE)]
    for cell in cells:cell.metadata['tags']=[TAG]
    temporary=nbformat.v4.new_notebook(cells=cells)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[str(ROOT/'.venv-phase1/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(temporary,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    images=sum('image/png' in o.get('data',{}) for c in temporary.cells for o in c.get('outputs',[]))
    assert images==10,images
    notebook.cells=[c for c in notebook.cells if TAG not in c.metadata.get('tags',[])]+temporary.cells
    nbformat.validate(notebook)
    staged=PATH.with_suffix('.high_presence.tmp.ipynb')
    nbformat.write(notebook,staged);staged.replace(PATH)
    print('Appended higher-P(1) table and',images,'rendered plots; existing outputs preserved.')


if __name__=='__main__':main()
