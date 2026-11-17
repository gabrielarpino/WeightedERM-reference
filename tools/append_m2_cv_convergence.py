"""Append a transparent dimension/trial-count follow-up; preserve prior outputs."""
import json
from pathlib import Path
import nbformat as nbf

TAG='m2-cv-theory-convergence-v1'


def append(dimensions=(200,220,400),trials=300):
    path=Path('E-WERM-CV-theory.ipynb')
    notebook=json.loads(path.read_text())
    if notebook.get('metadata',{}).get('werm_cv_presentation',{}).get('archived_attempts'):
        raise ValueError('This notebook now displays the selected M2 run only. '
                         'Historical convergence attempts remain in results/M2_CV_theory_convergence/.')
    original=[c for c in notebook['cells'] if TAG not in c.get('metadata',{}).get('tags',[])]
    backup=Path('results/M2_CV_theory_convergence/notebook_before_convergence.ipynb')
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_bytes(path.read_bytes())
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    design_rows='\n'.join(f'| {p} | {9*p}, {11*p}, {13*p} | {trials} | {1.2*p:g} |'
                          for p in dimensions)
    intro=r'''## M2: dimension and trial-count convergence

This follow-up investigates whether the original M2 discrepancy is finite-sample
error or Monte Carlo variation. The original 100-trial results and the E_L study
above are preserved. No model, location, sampling ratio, or seed is selected for
better agreement. The trial count is fixed before inspecting the new results.

Keep the original M2 generator, true fractions $(0.20,0.50)$, sampling ratios
$\delta\in\{9,11,13\}$, unpenalized 10-fold CV, and the same first-order theory
construction. Increase the dimension while keeping the **relative** segment
constraints fixed: the minimum training segment size scales as $1.2p$, including
the endpoint segments. The exact admissible-partition marginal weights are
recomputed at each dimension. Final full-data constraints use the corresponding
fraction and integer ceiling.

| Dimension $p$ | Sample sizes $n$ | Trials per setting | Minimum training segment size |
|---:|---|---:|---:|
'''+design_rows+r'''

The $p=200$ follow-up extends the original run: its first 100 datasets are reused
exactly, followed by additional trial IDs. It is therefore **not independent** of
the original 100-trial estimate. Different dimensions use their documented seed
streams; no trial pairing across dimensions is asserted. Empirical and theoretical
outcomes are paired within each dataset.

The $p=400$ follow-up is an explicitly approved exception to the earlier
$n<3000$ limit: its sample sizes are 3600, 4400 and 5200. Its minimum training
segment size is 480, retaining the same $m_s/p\ge1.2$ requirement. No other
model, fitting, selection or theory settings are changed.

Larger trial counts reduce uncertainty in the plotted means, but do not remove
bias in a first-order approximation. Increasing dimension may reduce partition
fluctuations at fixed sampling ratios; any persistent discrepancy remains visible.
The 20 final SE draws per dataset are retained: increasing them would address
Monte Carlo error in the final-location prediction, not the CV-selection step.'''
    cells=[md(intro),code('''# Self-contained follow-up. Completed checkpoints are reused.
import os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from tools import m2_cv_theory_convergence as convergence
from tools import werm_cv_theory_experiment as cv_experiment
import tools.consistent_plotting as cv_style
import matplotlib.pyplot as plt
CV_DIMENSIONS = '''+repr(list(dimensions))+'''
CV_TRIALS = '''+str(trials)+'''
CV_RUNS = {}
for p in CV_DIMENSIONS:
    root,spec,weights = convergence.prepare(p=p,trials=CV_TRIALS)
    if not (root/'results.pkl').exists():
        cv_experiment.run(root,spec,weights,jobs=int(os.environ.get('WERM_CV_THEORY_JOBS','12')))
    CV_RUNS[p] = (root,spec)
CV_OUTPUT = CV_RUNS[max(CV_DIMENSIONS)][0]
cv_style.OUTPUT_DIR = CV_OUTPUT/'figures'
cv_style.SOURCE_HASHES.clear();cv_style.FIGURE_AUDIT.clear()
CV_ORIGINAL = cv_style.load_saved(convergence.PARENT/'results.pkl')
CV_FOLLOWUPS = {p:cv_style.load_saved(root/'results.pkl') for p,(root,spec) in CV_RUNS.items()}
CV_LABELS = (r'\\textbf{WERM-CV}',r'\\textbf{Theory}')
CV_COLORS = ('black','#4477AA')
print('Follow-up result directories:',{p:str(v[0]) for p,v in CV_RUNS.items()})'''),
    md(r'''### Aggregate agreement and its Monte Carlo uncertainty

Each row compares the empirical and theory means over datasets. Differences are
**Theory minus WERM-CV**. Their standard errors use the paired per-dataset
differences, with the theory's 20 final-location draws averaged within each
dataset. This treats the signal matrix as the independent sampling unit rather
than counting all theory draws as independent datasets.

Count agreement reports the fraction of individual datasets with the same CV
selection; it is distinct from agreement of mean-count curves. Hausdorff means
are normalized by $n$. If empty estimates occur, their numbers and the number of
finite pairs are retained explicitly.'''),
    code('''def cv_records(payload,delta):
    return [r for r in payload['records'] if r['experiment']=='M2' and r['delta']==delta]

def cv_vectors(records):
    empirical_count=np.array([r['empirical']['selected_by_bound'][4]['selected_segments']-1 for r in records],float)
    theory_count=np.array([r['theory']['selected_by_bound'][4]['selected_segments']-1 for r in records],float)
    empirical_haus=np.array([r['empirical']['selected_by_bound'][4]['haus'] for r in records],float)
    theory_haus=[]
    for r in records:
        h=np.array([f['haus'] for f in r['theory']['selected_by_bound'][4]['final_fits']],float)
        theory_haus.append(h[np.isfinite(h)].mean() if np.isfinite(h).any() else np.nan)
    return empirical_count,theory_count,empirical_haus,np.array(theory_haus)

def finite_mean(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return values.mean() if len(values) else np.nan

def paired_summary(a,b):
    valid=np.isfinite(a)&np.isfinite(b);d=(b-a)[valid]
    return (d.mean() if len(d) else np.nan,
            d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else np.nan,len(d))

CV_COMPARISON_ROWS=[]
for label,payload in [('Original 100 trials',CV_ORIGINAL)]+[(f'p={p}, {CV_TRIALS} trials',v) for p,v in CV_FOLLOWUPS.items()]:
    for delta in (9.,11.,13.):
        records=cv_records(payload,delta)
        ec,tc,eh,th=cv_vectors(records)
        count_diff,count_se,count_pairs=paired_summary(ec,tc)
        haus_diff,haus_se,haus_pairs=paired_summary(eh,th)
        empirical_scores=np.array([r['empirical']['normalized_cv_scores'] for r in records])
        theory_scores=np.array([r['theory']['normalized_cv_scores'] for r in records])
        CV_COMPARISON_ROWS.append(dict(run=label,p=records[0]['p'],n=records[0]['n'],delta=delta,
            trials=len(records),empirical_count=ec.mean(),theory_count=tc.mean(),
            count_difference=count_diff,count_difference_se=count_se,
            individual_count_agreement=np.mean(ec==tc),
            empirical_haus=finite_mean(eh),theory_haus=finite_mean(th),
            haus_difference=haus_diff,haus_difference_se=haus_se,finite_haus_pairs=haus_pairs,
            empirical_empty=int(np.sum(ec==0)),theory_empty=int(np.sum(tc==0)),
            mean_absolute_candidate_score_error=np.mean(np.abs(empirical_scores-theory_scores))))
CV_COMPARISON=pd.DataFrame(CV_COMPARISON_ROWS)
CV_COMPARISON.to_csv(CV_OUTPUT/'convergence_comparison.csv',index=False)
with pd.option_context('display.max_columns',None,'display.width',180):
    display(CV_COMPARISON.round(6))'''),
    md('''### Follow-up curves at each dimension

Each dimension is displayed separately. WERM-CV retains the canonical black
curve and empirical quartile bars. Theory is a blue diamond scatter overlay,
with no connecting lines or error bars. Theory location means average the final
draws within each dataset first, then average over datasets. These figures do
not replace or conceal the original 100-trial curves above.'''),
    code('''def cv_followup_plot(p,payload):
    deltas=(9.,11.,13.)
    vectors=[cv_vectors(cv_records(payload,d)) for d in deltas]
    for metric,emp_index,theory_index,ylabel in [('count',0,1,'No. change points'),
            ('haus',2,3,r'Hausdorff dist. / $n$')]:
        with plt.rc_context(cv_style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            cv_style.error_series(ax,deltas,[v[emp_index] for v in vectors],
                CV_LABELS[0],CV_COLORS[0],'D','-',scale=1)
            ax.scatter(deltas,[finite_mean(v[theory_index]) for v in vectors],
                label=CV_LABELS[1],marker='d',color=CV_COLORS[1],s=120,
                edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
            ax.set(xlabel=r'$\\delta=n/p$',ylabel=ylabel,xticks=deltas)
            ax.set_title(rf'$p={p}$, {CV_TRIALS} trials',fontsize=14)
            if metric=='count':
                ax.axhline(2,color='gray',lw=2.5,alpha=.8,zorder=0)
                ax.set_yticks(range(4))
            handles,labels=ax.get_legend_handles_labels()
            ax.legend([handles[labels.index(l)] for l in CV_LABELS],CV_LABELS,
                loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();cv_style.export_plot(fig,f'M2_p{p}_{metric}_convergence')
for p,payload in CV_FOLLOWUPS.items():
    cv_followup_plot(p,payload)
cv_style.verify_sources()'''),
    md('''### Candidate-score check at the largest dimension

These score curves show every candidate, not only the selected one. The
diagnostics separate observed validation noise, uniform-refit risk approximation,
and the effect of replacing empirical training partitions with the theoretical
limiting partitions. Empirical coefficients and intervals enter only the diagnostic
calculations; the main Theory prediction and count selection never use them.'''),
    code('''CV_STAGE_ROWS=[]
for p,payload in CV_FOLLOWUPS.items():
    for delta in (9.,11.,13.):
        records=cv_records(payload,delta)
        observed=np.array([r['empirical']['normalized_cv_scores'] for r in records])
        predicted=np.array([r['theory']['normalized_cv_scores'] for r in records])
        conditional=np.array([r['conditional_validation_risk'] for r in records])
        refit=np.array([r['refit_se_at_empirical_partitions'] for r in records])
        sorted_risks=np.sort(predicted,axis=1)
        CV_STAGE_ROWS.append(dict(p=p,delta=delta,
            observed_vs_conditional_mae=np.abs(observed-conditional).mean(),
            conditional_vs_refit_se_mae=np.abs(conditional-refit).mean(),
            empirical_vs_theory_partition_risk_mae=np.abs(refit-predicted).mean(),
            median_theory_selection_gap=np.median(sorted_risks[:,1]-sorted_risks[:,0]),
            min_empirical_segment_ratio=min(r['diagnostics']['min_empirical_segment_ratio'] for r in records),
            min_theory_segment_ratio=min(r['diagnostics']['min_theory_segment_ratio'] for r in records),
            max_se_residual=max(r['diagnostics']['max_se_residual'] for r in records)))
        if p!=max(CV_DIMENSIONS):continue
        with plt.rc_context(cv_style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            k=np.arange(1,5)
            cv_style.error_series(ax,k,observed.T,CV_LABELS[0],CV_COLORS[0],'D','-',scale=1)
            ax.scatter(k,predicted.mean(axis=0),label=CV_LABELS[1],marker='d',color=CV_COLORS[1],
                s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
            ax.set(xlabel=r'Candidate segments $k$',ylabel='Absolute prediction error',xticks=k)
            ax.set_title(rf'$p={p},\\ \\delta={delta:g}$',fontsize=14)
            handles,labels=ax.get_legend_handles_labels()
            ax.legend([handles[labels.index(l)] for l in CV_LABELS],CV_LABELS,
                loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();cv_style.export_plot(fig,f'M2_p{p}_scores_{delta:g}_convergence')
CV_STAGE_DIAGNOSTICS=pd.DataFrame(CV_STAGE_ROWS)
CV_STAGE_DIAGNOSTICS.to_csv(CV_OUTPUT/'convergence_stage_diagnostics.csv',index=False)
display(CV_STAGE_DIAGNOSTICS.round(6))
assert CV_STAGE_DIAGNOSTICS.min_empirical_segment_ratio.min()>=1.2-1e-12
assert CV_STAGE_DIAGNOSTICS.min_theory_segment_ratio.min()>=1.2-1e-12
assert CV_STAGE_DIAGNOSTICS.max_se_residual.max()<1e-9
assert all(len(v['records'])==3*CV_TRIALS for v in CV_FOLLOWUPS.values())
cv_style.verify_sources()
print('Original M2 and E_L cells and outputs were preserved.')''')]
    for c in cells:c['metadata']['tags']=[TAG]
    notebook['cells']=original+json.loads(nbf.writes(nbf.v4.new_notebook(cells=cells)))['cells']
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    assert json.loads(path.read_text())['cells'][:len(original)]==original
    return path,len(original)


if __name__=='__main__':
    print(append())
