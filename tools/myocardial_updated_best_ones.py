"""Updated top-20 report from existing numerical checks; never reruns a fit.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-phase1/bin/python tools/myocardial_updated_best_ones.py
"""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from tools.myocardial_best_ones import load_candidates, curve_metrics, rank_candidates

INTERCEPT=Path('results/myocardial_finite_intercept_e56e1d9d6bdb')
NO_INTERCEPT=Path('results/myocardial_no_intercept_solvers_57efc84323e8')
NOTEBOOK=Path('Myocardial Infarction Experiment - updated best ones top 20.ipynb')


def fitting_description(method,budget,step=1.,seed=42):
    names={'fista':'FISTA','lbfgs':'L-BFGS','gd':'GD','sgd':'SGD','saga':'SAGA'}
    unit='epochs' if method in ('sgd','saga') else 'iterations'
    extra=f'; step multiplier={step:g}' if method in ('fista','gd','sgd') else ''
    if method in ('sgd','saga'):extra+=f'; seed={seed}'
    return f'Logistic {names[method]} {budget} {unit}{extra} (no penalty)'


def paired_candidate(first,second,meta,first_path,second_path,reference):
    a=np.asarray(first['ages'],float); b=np.asarray(second['ages'],float)
    np.testing.assert_array_equal(a,b)
    p=np.asarray(first['conditional'],float);q=np.asarray(second['conditional'],float)
    m=curve_metrics(a,p,reference);n=curve_metrics(b,q,reference)
    probs=np.array([first['P_1'],second['P_1']],float)
    if not np.isfinite(probs).all() or min(probs)<-1e-8 or max(probs)>1+1e-8:
        raise ValueError('invalid count probability')
    probs=np.clip(probs,0,1)
    key=json.dumps({k:meta[k] for k in ('auxiliary','posterior','intercept','Delta','rho_updates','initialization')},sort_keys=True)
    ident=hashlib.sha256((str(first_path)+'|'+str(second_path)).encode()).hexdigest()[:12]
    row=dict(meta,id=ident,configuration=key,P1_min=float(min(probs)),P1_max=float(max(probs)),
        primary_P1=probs[0],check_P1=probs[1],mean_TV=(m['TV']+n['TV'])/2,
        max_TV=max(m['TV'],n['TV']),numerical_curve_TV=float(.5*np.abs(p-q).sum()),
        mass_above55_min=min(m['above55'],n['above55']),peak_min=min(m['peak'],n['peak']),
        peak_max=max(m['peak'],n['peak']),moment_residual=first.get('moment_residual',np.nan),
        FP_residual_max=max(first.get('FP_residual_max',np.nan),second.get('FP_residual_max',np.nan)),
        primary_checkpoint=str(first_path),check_checkpoint=str(second_path))
    return row,dict(ages=a,primary=p,check=q)


def load_updated():
    old,curves,reference=load_candidates()
    rows=old.drop(columns='rank').to_dict('records')
    validations=pd.read_csv(INTERCEPT/'validation.csv')
    for (case,fit),g in validations.groupby(['case','posterior_fit']):
        if set(g.tag)!= {'q9_seed42','q10_seed123'}:continue
        paths=[INTERCEPT/f'{case}__{fit}__{tag}.pkl' for tag in ('q9_seed42','q10_seed123')]
        first,second=[pickle.load(p.open('rb')) for p in paths]
        post=fitting_description(first['method'],first['fit_budget'])
        aux='LS' if first['group']=='LS_auxiliary' else post
        scale=first['initial_scale'];init='identity' if scale==1 else f'scale{scale:g}_corr0'
        row,curve=paired_candidate(first,second,dict(auxiliary=aux,posterior=post,
            intercept=True,Delta=first['Delta'],rho_updates=first['rho_updates'],initialization=init,
            stage='finite_intercept',priority=4,
            preference_note='Uses intercept; '+('LS auxiliary' if aux=='LS' else 'matched finite logistic fits')),*paths,reference)
        rows.append(row);curves[row['id']]=curve
    validations=pd.read_csv(NO_INTERCEPT/'validation.csv')
    for case,g in validations.groupby('case'):
        if set(g.tag)!= {'q9_seed42_b100','q10_seed123_b100'}:continue
        paths=[Path(g[g.tag==tag].iloc[0].checkpoint) for tag in ('q9_seed42_b100','q10_seed123_b100')]
        first,second=[pickle.load(p.open('rb')) for p in paths]
        name=fitting_description(first['solver'],first['posterior_budget'],first['step_scale'],first['fit_seed'])
        row,curve=paired_candidate(first,second,dict(auxiliary=name,posterior=name,intercept=False,
            Delta=first['Delta'],rho_updates=first['rho_updates'],initialization='identity',
            stage='no_intercept_solver_sweep',priority=4,preference_note='No intercept; matched finite logistic fits'),*paths,reference)
        rows.append(row);curves[row['id']]=curve
    ranked=rank_candidates(rows)
    ranked['matched_logistic']=(ranked.auxiliary==ranked.posterior)&ranked.auxiliary.str.startswith('Logistic')
    ranked['contains_L1']=ranked.auxiliary.str.contains('L1')|ranked.posterior.str.contains('L1')
    return ranked,curves,reference


def clear_tables(frame):
    settings=pd.DataFrame({'Rank':frame['rank'],'Auxiliary fit':frame.auxiliary,
        'Posterior fit':frame.posterior,'Intercept':np.where(frame.intercept,'Yes','No'),
        'Delta':frame.Delta,'Rho updates':frame.rho_updates,'Rho initialization':frame.initialization,
        'Matched logistic fits':np.where(frame.matched_logistic,'Yes','No')})
    metrics=pd.DataFrame({'Rank':frame['rank'],
        'P(1), numerical range':[f'{a:.4f} – {b:.4f}' for a,b in zip(frame.P1_min,frame.P1_max)],
        'Shape TV ↓':frame.mean_TV.round(4),'Check-to-check TV ↓':frame.numerical_curve_TV.round(4),
        'Conditional peak age':[f'{a:g}' if a==b else f'{a:g} – {b:g}' for a,b in zip(frame.peak_min,frame.peak_max)],
        'Min. mass above age 55':frame.mass_above55_min.round(4),
        'Rho moment residual':frame.moment_residual.map(lambda x:f'{x:.3g}'),
        'Study':frame.stage})
    return settings,metrics


def show_table(table):
    from IPython.display import display
    display(table.style.hide(axis='index').set_properties(**{'text-align':'left','white-space':'normal',
        'padding':'8px','font-size':'12px','max-width':'310px'}).set_table_styles([
        {'selector':'th','props':[('text-align','left'),('padding','8px'),('background-color','#e8edf2')]},
        {'selector':'tr:nth-child(even)','props':[('background-color','#f5f7f9')]}]))


def plot_candidate(row,curves,reference):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    rec=curves[row.id];a=np.array(sorted(set(reference)|set(rec['ages'])))
    def align(values):
        m=dict(zip(rec['ages'],values));return np.array([m.get(x,0.) for x in a])
    ref=np.array([reference.get(x,0.) for x in a]);p=align(rec['primary']);q=align(rec['check'])
    fig,(ax,cdf)=plt.subplots(1,2,figsize=(11,3.6),gridspec_kw={'width_ratios':[1.8,1]})
    for axis,cumulative in ((ax,False),(cdf,True)):
        for vals,color,lw,style,label in ((ref,'.58',1.8,'-','Original'),
            (p,'black',1.7,'-',f'512 draws; P(1)={row.primary_P1:.4f}'),
            (q,'#2878b5',1.4,'--',f'1,024 draws; P(1)={row.check_P1:.4f}')):
            axis.plot(a,np.cumsum(vals) if cumulative else vals,color=color,lw=lw,ls=style,label=label)
        axis.set_xlabel('Age');axis.xaxis.set_major_locator(MaxNLocator(6,integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(4));axis.set_ylim(bottom=0)
    ax.set_ylabel('Conditional probability');cdf.set_ylabel('Cumulative probability');cdf.set_ylim(0,1.02)
    ax.legend(frameon=False,fontsize=9)
    fig.suptitle(f'#{row["rank"]}   Shape TV={row.mean_TV:.3f}   |   P(1)={row.P1_min:.3f}–{row.P1_max:.3f}   |   Numerical TV={row.numerical_curve_TV:.3f}',fontsize=12)
    fig.tight_layout();plt.show()


def build_notebook():
    import nbformat as nbf
    cells=[nbf.v4.new_markdown_cell(r'''# Updated best ones — top 20 across completed sweeps

**Saved results only: no experiments rerun.** Includes all nine families of
comparable, paired numerical checks: broad finite iterations; focused budgets/Delta;
rho initialization; fully solved matched L2; earlier LS-auxiliary penalized fits;
earlier intercept-aware penalized fits; earlier exact matched penalties; the new
finite-intercept sweep; and the new no-intercept five-solver sweep.

**“Best” means closest conditional posterior shape to the archived original**, ranked
by mean total-variation distance across two numerical checks (0=identical, 1=maximal
difference). No P(1) cutoff or probability bonus. Full age support is used, including
mass excluded by Delta. Only configurations with two saved numerical checks qualify:
this is not a claim that every primary-screening candidate has been revalidated.
Historical pre-audit runs with incompatible count priors/likelihood conventions
are not pooled. Original data and archived posterior are unchanged.

All LS/L1/L2/finite-fit and intercept choices are clearly identified. The overall
list does **not** silently impose the later matched-logistic/no-L1 preference;
a separate preferred-subset table follows. L2/L1 constants in fit labels are the
actual objective penalty coefficients, not sklearn's inverse-regularization C.
“Full solve” denotes convergence-seeking fitting, not a finite iteration budget.
For iterative fits, budgets count solver iterations (SGD/SAGA count epochs).
Rho updates are a separate outer procedure; -1 denotes a direct approximate estimate.

The 512-draw/seed42 and 1,024-draw/seed123 ranges are **numerical sensitivity, not
confidence intervals**. These are post-hoc, same-data selections, not calibrated
model comparisons or independent-data validation. Moment residuals are reported;
state equations remain heuristic for penalized and early-stopped fits. Shape
similarity to the original is a requested descriptive target, not proof of validity.
'''),nbf.v4.new_code_cell('''from tools.myocardial_updated_best_ones import *
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'figure.dpi':160,
    'axes.spines.top':False,'axes.spines.right':False})
ranked,curves,reference=load_updated()
best=ranked.head(20).copy()
assert len(best)==20
digest=hashlib.sha256(Path('tools/myocardial_updated_best_ones.py').read_bytes()).hexdigest()[:12]
report_root=Path('results')/('myocardial_updated_best_ones_'+digest)
report_root.mkdir(exist_ok=True)
ranked.to_csv(report_root/'all_checked_candidates.csv',index=False)
best.to_csv(report_root/'top_twenty.csv',index=False)
print(f'{len(ranked)} distinct paired-check configurations across {ranked.stage.nunique()} study families.')
print('Ranked inventory by study:');display(ranked.groupby('stage').size().rename('Candidates').to_frame())
settings,metrics=clear_tables(best)
settings.to_csv(report_root/'top_twenty_settings.csv',index=False)
metrics.to_csv(report_root/'top_twenty_metrics.csv',index=False)
print('Settings — ranks link directly to the metrics and individual figures below:')
show_table(settings)
print('Outcomes — smaller shape TV is better; numerical ranges are not confidence intervals:')
show_table(metrics)
print('Exact checkpoint paths and machine-readable tables:',report_root)
'''),nbf.v4.new_code_cell('''fig,(ax,bx)=plt.subplots(1,2,figsize=(11,6.5),sharey=True)
y=np.arange(1,21)
ax.hlines(y,best.P1_min,best.P1_max,color='black',lw=2)
ax.scatter(best.primary_P1,y,color='black',s=22,label='512 draws')
ax.scatter(best.check_P1,y,color='#2878b5',marker='x',s=27,label='1,024 draws')
ax.axvline(.5,color='.65',lw=1,ls=':');ax.set_xlim(0,1)
ax.set(xlabel='P(1 change point)',ylabel='Overall rank',yticks=y)
ax.invert_yaxis();ax.legend(frameon=False)
bx.scatter(best.mean_TV,y,color='black',s=22,label='Shape TV to original')
bx.scatter(best.numerical_curve_TV,y,color='#2878b5',marker='x',s=27,label='Numerical check-to-check TV')
bx.set_xlabel('TV distance (lower is better)');bx.legend(frameon=False)
fig.tight_layout();plt.show()
''')]
    for rank in range(1,21):
        cells.append(nbf.v4.new_code_cell(f'''row=best.iloc[{rank-1}]
display(Markdown(f"### #{{row['rank']}} — {{row.stage}}\\n\\n"
    f"**Auxiliary:** {{row.auxiliary}}  \\n**Posterior:** {{row.posterior}}  \\n"
    f"**Intercept:** {{'Yes' if row.intercept else 'No'}}; **Delta:** {{row.Delta}}; "
    f"**rho updates:** {{row.rho_updates}}; **rho initialization:** {{row.initialization}}.  \\n"
    f"{{row.preference_note}}. Moment-equation residual: {{row.moment_residual:.3g}}."))
plot_candidate(row,curves,reference)
'''))
    cells.extend([nbf.v4.new_markdown_cell('''## Preference-restricted comparison and audit trail
Below: matched logistic/logistic fits only, excluding L1. Overall rank is retained,
so a restricted-subset leader cannot be mistaken for an overall top-20 result.
Intercept choice remains visible; no preference filters change the main ranking.
'''),nbf.v4.new_code_cell('''preferred=ranked[ranked.matched_logistic & ~ranked.contains_L1].head(10)
show_table(clear_tables(preferred)[0]);show_table(clear_tables(preferred)[1])
print('Exact sources for the overall top 20:')
display(best[['rank','primary_checkpoint','check_checkpoint']])
''')])
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,str(NOTEBOOK),'exec')
    nbf.write(notebook,NOTEBOOK)
    return NOTEBOOK


if __name__=='__main__':
    import sys
    path=build_notebook()
    print('Created:',path,flush=True)
    if '--build-only' not in sys.argv:
        from tools import execute_myocardial_rho_final as runner
        runner.PATH=path.resolve();runner.main()
