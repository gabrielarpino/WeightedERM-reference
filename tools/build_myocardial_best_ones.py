"""Create a plot-only notebook from saved, numerically checked experiments."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - best ones.ipynb')


def build():
    intro=r'''# Myocardial infarction — best ones

This notebook **loads saved results only**: it does not refit models, estimate rho,
or rerun experiments. The shortlist covers the broad early-stopping sweep, focused
Delta/optimizer sweep, initialization sensitivity, fine matched-L2 full solves, and
earlier numerically checked penalized/intercept experiments.

**Ranking:** among distinct configurations with two numerical checks, sort by the
mean total-variation (TV) distance of the conditional one-change posterior from the
archived conditional curve. Lower is better. TV is recomputed from saved probability
vectors on the full reference age support; probability excluded by a larger Delta
is not renormalized away. Duplicate configurations are removed, preferring the newer
checked implementation. There is **no P(1)>0.5 cutoff** and no hidden probability bonus.
This is a shape ranking, not a universal ranking of inferential quality.

**Preferences:** rows using an intercept or different penalties are explicitly
flagged. They remain in the overall shape ranking rather than being presented as
satisfying the no-intercept/matched-fit preference. A separate preferred subset is
shown at the end. Rank 3 is the leading current no-intercept compromise between
original-like shape and P(1) close to one-half; rank 2 is slightly closer in shape.

All probability ranges compare 512 Sobol draws/seed 42 with 1,024 draws/seed 123;
they are **not confidence intervals**. These same-data, outcome-inspected heuristic
analyses do not establish calibration or validate a selected model. In particular,
rho estimates can have nonzero moment-equation residuals, and the existing state
equations are only approximations for early-stopped/penalized coefficient fits.
'''
    source=Path('tools/myocardial_best_ones.py').read_text()
    execute='''ranked,curves,reference=load_candidates()
best=ranked.head(10).copy()
assert len(best)==10
print(f'Ranked {len(ranked)} distinct configurations with two saved numerical checks.')
display(summary_table(best))
report_digest=hashlib.sha256(Path('tools/myocardial_best_ones.py').read_bytes()).hexdigest()[:12]
report_root=Path('results')/('myocardial_best_ones_'+report_digest)
report_root.mkdir(exist_ok=True)
ranked.to_csv(report_root/'all_checked_candidates.csv',index=False)
best.to_csv(report_root/'top_ten.csv',index=False)
print('Auditable ranking and exact checkpoint paths:',report_root)
'''
    overview=r'''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,
    'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
a=np.array(sorted(reference));p=np.array([reference[x] for x in a])
def aligned(record,key):
    mapping=dict(zip(record['ages'],record[key]))
    return np.array([mapping.get(x,0.) for x in a])
def draw_curves(ax,row,legend=False):
    saved=curves[row.id]
    ax.plot(a,p,color='0.60',lw=1.4,label='Archived conditional posterior')
    ax.plot(a,aligned(saved,'primary'),color='black',lw=1.5,label='512 draws (seed 42)')
    ax.plot(a,aligned(saved,'check'),color='#4178a8',lw=1.2,label='1,024 draws (seed 123)')
    ax.axvline(55,color='0.65',ls=':',lw=.8)
    ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel('Age');ax.set_ylabel('Probability')
    if legend:ax.legend(frameon=False,fontsize=8)
print('Top ten conditional posteriors: common axes; zero mass shown outside each candidate support')
fig,axes=plt.subplots(5,2,figsize=(10,12.5),sharex=True,sharey=True)
upper=1.08*max([p.max()]+[max(curves[row.id]['primary'].max(),curves[row.id]['check'].max()) for row in best.itertuples()])
for ax,row in zip(axes.flat,best.itertuples()):
    draw_curves(ax,row)
    flag='; intercept' if row.intercept else ''
    ax.set_title(f'#{row.rank}: TV {row.mean_TV:.3f}; P(1) {row.P1_min:.3f}–{row.P1_max:.3f}{flag}',fontsize=10)
    ax.set_ylim(0,upper)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
fig.tight_layout(rect=(0,0,1,.97));plt.show()
'''
    tradeoff=r'''print('Count probability across numerical checks (connecting lines are not confidence intervals)')
fig,ax=plt.subplots(figsize=(7,3))
for row in best.itertuples():
    ax.plot([row.rank,row.rank],[row.P1_min,row.P1_max],color='0.6',lw=1.2)
ax.plot(best['rank'],best.primary_P1,'D',color='black',ms=4,label='512 draws')
ax.plot(best['rank'],best.check_P1,'o',color='#4178a8',ms=3,label='1,024 draws')
ax.axhline(.5,color='0.55',ls=':',lw=1)
ax.set_xticks(best['rank']);ax.set_xlabel('Candidate rank');ax.set_ylabel('P(1 chgpt)')
ax.set_ylim(0,1);ax.yaxis.set_major_locator(MaxNLocator(3));ax.legend(frameon=False)
fig.tight_layout();plt.show()

print('Shape/probability trade-off: rank numbers identify the configurations above')
fig,ax=plt.subplots(figsize=(6,3))
for row in best.itertuples():
    color='#c69c22' if row.intercept else 'black'
    mean_p=(row.P1_min+row.P1_max)/2
    ax.scatter(row.mean_TV,mean_p,color=color,marker='D',s=22)
    ax.annotate(str(row.rank),(row.mean_TV,mean_p),xytext=(4,4),textcoords='offset points',fontsize=9)
ax.axhline(.5,color='0.55',ls=':',lw=1)
ax.set_xlabel('Mean TV from archived curve (lower is closer)');ax.set_ylabel('Mean P(1 chgpt)')
ax.scatter([],[],color='black',marker='D',s=22,label='No intercept')
ax.scatter([],[],color='#c69c22',marker='D',s=22,label='Uses intercept')
ax.legend(frameon=False);ax.yaxis.set_major_locator(MaxNLocator(3))
fig.tight_layout();plt.show()
'''
    details=r'''print('Individual plots and exact configurations')
for row in best.itertuples():
    print(f'#{row.rank}: auxiliary = {row.auxiliary}; posterior-input fit = {row.posterior}.')
    print(f'Intercept = {row.intercept}; Delta = {row.Delta}; rho updates = {row.rho_updates}; '
          f'rho initialization = {row.initialization}. {row.preference_note}.')
    print(f'P(1) = {row.P1_min:.6f}–{row.P1_max:.6f}; mean reference TV = {row.mean_TV:.6f}; '
          f'TV between numerical checks = {row.numerical_curve_TV:.6f}; '
          f'minimum mass above 55 = {row.mass_above55_min:.4f}; '
          f'moment residual = {row.moment_residual:.6g}.')
    retained=sum(reference.get(x,0.) for x in curves[row.id]['ages'])
    print(f'Reference mass retained by this admissible age grid: {retained:.4f}.')
    fig,ax=plt.subplots(figsize=(6,2.8));draw_curves(ax,row,legend=True)
    fig.tight_layout();plt.show()
'''
    preferred='''print('Top-ten candidates satisfying the no-intercept preference')
display(summary_table(best[~best.intercept]))
print('For a shape/probability compromise inspect #3 first, then #2.')
print('For shape alone inspect #1; its count probability is substantially below one-half.')
print('Candidates #7, #8, #10 use intercepts and mismatched penalties; they are historical alternatives, not preference-compliant recommendations.')
print('No matched logistic/logistic candidate ranks in this overall shape-based top ten.')
print('Exact input checkpoints and nonzero moment-equation residuals:')
display(best[['rank','moment_residual','primary_checkpoint','check_checkpoint']])
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(overview),nbf.v4.new_code_cell(tradeoff),
        nbf.v4.new_code_cell(details),nbf.v4.new_code_cell(preferred)])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook_cell','exec')
    nbf.write(nb,PATH);print(PATH)


if __name__=='__main__':build()
