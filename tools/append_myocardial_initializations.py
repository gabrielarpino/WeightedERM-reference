"""Append the initialization study, with the full grid and sensitivity plots."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - finite iteration exploration.ipynb')
TAG='## Initialization sensitivity: finite logistic in both stages'


def append():
    nb=nbf.read(PATH,4)
    if any(c.cell_type=='markdown' and c.source.startswith(TAG) for c in nb.cells):return
    intro=TAG+r'''

To test whether a favorable result depends on the starting covariance, use the
explicit symmetric family
$$\rho^{(0)}=s\begin{pmatrix}1&r\\r&1\end{pmatrix},\qquad
s\in\{0.1,1,10\},\quad r\in\{-0.5,0,0.5\}.$$
All nine starts are positive definite and the identity is included. These are
generic scale/correlation perturbations, not matrices fitted to the archived curve.
The grid is disclosed in full rather than presenting a selected start as a default.

Both fitting stages are unpenalized, finite-iteration logistic FISTA with no
intercept. Auxiliary budgets are 1, 3, 10, 30, rho updates are 1, 3, 10, and
posterior-input budgets are the original seven values. Delta is fixed at 1 to retain
the complete archived age support. This gives **756 primary comparisons**. Identity
states are reused only after exact covariance and age-grid checks.

The two closest curves and the highest-P(1) curve within 0.10 TV of the best in each
matched/different-budget class are checked at finer numerical accuracy. The best
matched-budget candidate is additionally checked at neighboring starts and the
identity baseline. A favorable isolated start is not evidence of robustness.
P(1) near or above 0.5 and a reference-like shape are exploratory objectives, not
acceptance constraints imposed on the reported results. All prior/model/normalization
choices remain unchanged.
'''
    source=Path('tools/myocardial_initialization_sweep.py').read_text().split("if __name__=='__main__':")[0]
    execute='''initialization_root=run_initializations(initial_root)
root=initialization_root
summary=pd.read_csv(root/'summary.csv')
validation=pd.read_csv(root/'validation.csv')
ranked=pd.read_csv(root/'ranked.csv')
display(ranked)
'''
    heatmaps=r'''print('All matched-budget settings: initialization sensitivity (primary accuracy)')
matched=summary[summary.group=='logistic_matched']
for budget,data in matched.groupby('auxiliary_budget'):
    print('Both logistic fitting budgets =',budget)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,key,title in zip(axes,('P_1','TV_to_reference'),('P(1 chgpt)','TV from archived curve')):
        table=data.pivot(index='initialization',columns='rho_updates',values=key)
        im=ax.imshow(table.to_numpy(),vmin=0,vmax=1,aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(table.columns)),table.columns)
        ax.set_yticks(range(len(table.index)),table.index,fontsize=8)
        ax.set_xlabel('Rho updates');ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03,ticks=(0,.5,1))
    fig.tight_layout();plt.show()
'''
    curves=nb.cells[4].source.replace("print(f'{row.group}:","print(f'{row.initialization}; {row.group}:")
    combined=nb.cells[-1].source.replace("[('broad',initial_root),('focused',focused_root)]",
        "[('broad',initial_root),('focused',focused_root),('initializations',initialization_root)]")
    nb.cells.extend([nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(heatmaps),nbf.v4.new_code_cell(curves),
        nbf.v4.new_code_cell(combined)])
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'notebook_cell','exec')
    nbf.write(nb,PATH);print('Initialization cells appended and syntax-checked.')


if __name__=='__main__':append()
