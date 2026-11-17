"""Append the frozen-gamma location exploration without replacing prior studies."""
import json
from pathlib import Path
import nbformat as nbf

TAG = 'el-location-exploration-gamma6-v1'


def append():
    path=Path('E_L_upper_bound.ipynb')
    notebook=json.loads(path.read_text())
    original=[c for c in notebook['cells'] if TAG not in c.get('metadata',{}).get('tags',[])]
    backup=Path('results/E_L_location_exploration/notebook_before_exploration.ipynb')
    if not backup.exists():
        backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_bytes(path.read_bytes())
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md(r'''## Exploratory truth locations at frozen $\gamma=6$

The preceding studies remain unchanged. Here we ask whether changing the true
locations, or using two rather than three true changes, reduces over-selection
when the segment upper bound $L$ exceeds $L^\star$.

Keep $n=1000$, $p=100$, noise standard deviation 0.5, the same Gaussian random-walk
coefficient distribution, the same frozen upper-bound weights, and exactly
$P(K)=6\log\binom{n-1}{K-1}$. No coefficient tuning or CV is used.
For two changes, use the first three coefficient vectors of the same four-vector
draw; the unused fourth vector is still drawn so that covariates and noise remain
paired across all candidate layouts. SE retains the corresponding three or four
true regimes and runs its own penalized search.

**Pilot design:** all two- and three-location combinations of
$\{0.15,0.30,0.45,0.60,0.75,0.90\}$, plus $(0.20,0.50,0.80)$,
$(0.25,0.45,0.70)$ and $(0.30,0.55)$: 38 layouts, each with 20 paired trials.
Rank separately within each true count by empirical mean absolute count error
averaged over $L>L^\star$, then by higher exact-count frequency, then by lower
conditional Hausdorff error, then lexicographically. All pilot results are shown.

**Independent assessment:** freeze the best layout for each true count before
100 fresh paired trials, alongside the original three-change layout. Use 20
independent SE draws per signal matrix (2,000 per setting). Trial IDs 10000–10019
are for pilots; 20000–20099 are for assessment. Assessment outcomes and SE matches
never choose or retune locations. This is an exploratory, selected example of
layout sensitivity, not a general count-consistency claim.'''),
    code('''# Self-contained: load completed artifacts, or compute missing trials.
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from tools import el_location_exploration as exploration
EXP_ROOT, EXP_SPEC, EXP_WEIGHTS = exploration.prepare()
EXP_FILE = EXP_ROOT/'assessment.pkl'
EXPLORATION = (pickle.loads(EXP_FILE.read_bytes()) if EXP_FILE.exists()
               else exploration.run(EXP_ROOT,EXP_SPEC,EXP_WEIGHTS))
EXP_SUMMARY = exploration.summarize(EXPLORATION['records'],theory=True)
print('Frozen gamma:',EXP_SPEC['gamma'])
print('Artifacts:',EXP_ROOT)
print('Pilot-selected layouts:',EXPLORATION['frozen']['choices'])
PILOT_RANKING = pd.read_csv(EXP_ROOT/'pilot_ranking.csv')
with pd.option_context('display.max_rows',None):
    display(PILOT_RANKING.round(4))'''),
    md('''### Fresh assessment: count stability and localization

The table below includes all upper bounds at or above the true segment count.
`correct_count` is the fraction selecting the true number of changes; `over_count`
and `under_count` report over- and under-selection. `max_count` is the fraction
selecting the allowed maximum $L-1$. Hausdorff error is divided by $n$ and excludes
empty estimates, which are reported separately.

Mean panels retain the empirical curve and quartile bars. SE means are standalone
blue diamonds with white edges and no error bars or connecting lines, as in
the SE estimation plotting notebook. All 2,000 SE draws enter these means.
Histograms for the selected layouts use upper bounds 4, 5 and 6, with the first
SE draw per signal matrix to match the 100 empirical realizations.'''),
    code('''display(EXP_SUMMARY[EXP_SUMMARY.L>=EXP_SUMMARY.true_changes+1].round(4))
pilot_ids=set(EXP_SPEC['pilot_trials']); assessment_ids=set(EXP_SPEC['assessment_trials'])
assert not pilot_ids & assessment_ids
assert EXP_SPEC['gamma']==6.
assert len(EXPLORATION['records'])==300
assert all(len(f['theory'])==20 for r in EXPLORATION['records'] for f in r['fits'].values())
states=[f['se'] for r in EXPLORATION['records'] for f in r['fits'].values()]
residuals={key:max(s[key] for s in states)
           for key in ('b_residual','bias_residual','covariance_residual')}
assert max(residuals.values())<1e-9
print('Maximum SE residuals:',residuals)
print('Full pilot results by L:',EXP_ROOT/'pilot_by_upper_bound.csv')'''),
    code(r'''import matplotlib.pyplot as plt
import tools.consistent_plotting as exp_style
exp_style.OUTPUT_DIR=EXP_ROOT/'figures'
exp_style.SOURCE_HASHES.clear();exp_style.FIGURE_AUDIT.clear()
saved=exp_style.load_saved(EXP_FILE)
EXP_LABELS=(r'\textbf{WERM}',r'\textbf{Theory}')
EXP_COLORS=('black','#4477AA')
layouts=[exploration.BASELINE]+[tuple(saved['frozen']['choices'][str(k)]) for k in (3,2)]

def exploration_groups(records,source,metric):
    return [np.array([f[metric] for r in records for f in
        ([r['fits'][L]['empirical']] if source=='empirical' else r['fits'][L]['theory'])])
        for L in exploration.COUNTS]

for locations in layouts:
    records=[r for r in saved['records'] if r['locations']==locations]
    prefix='truth_'+exploration.layout_name(locations)
    truth_label=', '.join(f'{v:g}' for v in locations)
    for metric,ylabel in [('count','No. change points'),('haus',r'Hausdorff dist. / $n$')]:
        with plt.rc_context(exp_style.CANONICAL_STYLE):
            fig,ax=plt.subplots(figsize=(5,4));fig._canonical_m1=True
            groups=exploration_groups(records,'empirical',metric)
            exp_style.error_series(ax,exploration.COUNTS,groups,EXP_LABELS[0],
                                   EXP_COLORS[0],'D','-',scale=1)
            groups=exploration_groups(records,'theory',metric)
            means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in groups]
            ax.scatter(exploration.COUNTS,means,label=EXP_LABELS[1],marker='d',
                color=EXP_COLORS[1],s=120,edgecolor='white',linewidth=2.5,alpha=.99,zorder=5)
            ax.set(xlabel=r'Upper bound $L$',ylabel=ylabel,xticks=exploration.COUNTS)
            ax.set_title(r'True locations: $'+truth_label+'$',fontsize=14)
            if metric=='count':
                ax.axhline(len(locations),color='gray',lw=2.5,alpha=.8,zorder=0)
                ax.set_yticks(range(6))
            handles,labels=ax.get_legend_handles_labels()
            order=[labels.index(label) for label in EXP_LABELS]
            ax.legend([handles[i] for i in order],EXP_LABELS,loc='best',prop={'size':14},framealpha=.4)
            fig.tight_layout();exp_style.export_plot(fig,prefix+'_'+metric)
    if locations!=exploration.BASELINE:
        pools=[]
        for L in (4,5,6):
            pools.append([np.concatenate([r['fits'][L]['empirical']['cp']/EXP_SPEC['n'] for r in records]),
                          np.concatenate([r['fits'][L]['theory'][0]['cp']/EXP_SPEC['n'] for r in records])])
        exp_style.histogram_grid(pools,EXP_LABELS,EXP_COLORS,locations,prefix+'_histograms',
            row_labels=[rf'$L={L}$' for L in (4,5,6)],density_label_x=.005)
exp_style.verify_sources()'''),
    md('''Selection provenance is in `frozen_selection.json`; every pilot layout is in
`pilot_ranking.csv` and `pilot_by_upper_bound.csv`; independent empirical/SE
assessment is in `assessment_summary.csv` and `assessment.pkl`.
Implementation: `tools/el_location_exploration.py`. The original three-change
search-penalty and fixed-count studies above are retained for comparison.''')]
    for cell in cells:
        cell['metadata']['tags']=[TAG]
    notebook['cells']=original+json.loads(nbf.writes(nbf.v4.new_notebook(cells=cells)))['cells']
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    assert json.loads(path.read_text())['cells'][:len(original)]==original
    return path,len(original)


if __name__=='__main__':
    print(append())
