"""Build the three-candidate paper-format notebook without changing originals."""
from pathlib import Path
import copy
import nbformat as nb

NAME='Myocardial Infarction Experiment - final_final.ipynb'


def build():
    old=nb.read('Myocardial Infarction Experiment - final redo.ipynb',4)
    intro=nb.v4.new_markdown_cell('''# Myocardial infarction — final_final

Three fixed, exploratory analyses only: the previously reproduced rank-3 moment-update analysis;
the best numerically rechecked whole-data Gram candidate; and the best numerically rechecked
unbalanced segmented Gram candidate. “Best” means smallest mean conditional TV to rank 3
in the completed Gram sweep, **not** largest probability of a change. Selection was post hoc.
All use P(0 chgpts)=P(1 chgpt)=1/2 and uniform admissible location conditional on one change.
The same prior determines regression weights. No claim of calibrated change detection is made.

Plot functions and sensitivity-table formatting are copied from the final redo notebook:
LaTeX/serif, black stems, the same figure sizes, tick locations and label sizes. Each analysis
has rho, count probabilities, a conditional posterior, numerical diagnostics and independent
±10%/±20% eigenvalue perturbations. All reported integration calculations use 1024 draws.
For repeated eigenvalues (rank 3), the chosen eigenbasis is conventional, not uniquely identified.
Baseline fits are reproduced; matching archived integration states are reused. New sensitivity
states are checkpointed, with up to 36 workers. No original results are overwritten.

The two Gram baseline sections appear before their sensitivity sections so their principal
figures are available immediately. Paper-ready PDF/PNG and CSV/LaTeX tables are exported
under `results/myocardial_final_final_v1/paper_assets` in addition to inline display.
''')
    cells=[intro]+copy.deepcopy(old.cells)
    cells[1].source=cells[1].source.replace('# Myocardial infarction — final redo (updated rank 3)','# A. Rank-3 moment-update analysis')
    cells.append(nb.v4.new_code_cell(Path('tools/myocardial_final_final_gram.py').read_text()+
        '\nfrom IPython.display import display\nfrom itertools import product\n'
        "assets=FINAL_ROOT/'paper_assets';assets.mkdir(parents=True,exist_ok=True)\n"
        "gram_data={};gram_records={}\n"))
    plot=old.cells[7].source
    sens_table=old.cells[9].source.replace('compute_scenarios(data,remaining)','gram_scenarios(data,remaining)')
    sens_plot=old.cells[11].source.replace("print('Completed fresh rank-3 reproduction and sensitivity analysis.')",'')
    for mode,title,description in [
        ('whole','B. Whole-data Gram','Exact unpenalized LS auxiliary fits; logistic FISTA, 30 iterations, no explicit penalty; intercepts in both stages; Delta=5. rho=B^T B/p, p=110. Selected config283_p. No moment updates or covariance clipping.'),
        ('segmented','C. Estimated-segment Gram','LS auxiliary and segment refits with L1 strength 0.0001; logistic posterior with the same L1 strength, convergence-seeking solve; intercepts in both stages; Delta=2. rho=Bseg^T Bseg/n, n=1692. Selected config189_n. Unit within-segment weights (not balanced). Pilot selected by auxiliary loss, including the null; segment fits use its estimated split. No moment updates or covariance clipping.')]:
        cells.append(nb.v4.new_markdown_cell('## '+title+'\n\n'+description+
            '\n\nSquared loss uses residual squared (not half squared); weighted-mean losses plus the stated penalty; intercepts are unpenalized and excluded from Gram matrices. Same mean-imputed, centered, whitened global features as A. Gram estimates need not satisfy the signal-strength moment equations.'))
        cells.append(nb.v4.new_code_cell(f"mode='{mode}'\ndata=reconstruct_gram(mode)\ngram_data[mode]=data\n"
            "print('rho_hat:');display(pd.DataFrame(data['rho']))\n"
            "print('Eigenvalues:',np.linalg.eigvalsh(data['rho']))\n"
            "print('Pilot index:',data['pilot'],'first age after split:',data['ages'][data['pilot']])\n"
            "print('Posterior fitting diagnostics:');display(pd.DataFrame(data['fit']['checks']))\n"
            "records=gram_scenarios(data,[(1.,1.)]);gram_records[mode]=records\nbaseline=records[((1.,1.),10,123)]\n"
            "probability_table=pd.DataFrame([{'Draws':1024,'Seed':123,'P(0 chgpts)':baseline['absence'],'P(1 chgpt)':baseline['presence'],'Max. state residual':baseline['residuals'].max()}])\n"
            "display(probability_table.style.format({'P(0 chgpts)':'{:.6f}','P(1 chgpt)':'{:.6f}','Max. state residual':'{:.3g}'}))\n"
            "probability_table.to_csv(assets/f'{mode}_probabilities.csv',index=False)\n"
            "probability_table.to_latex(assets/f'{mode}_probabilities.tex',index=False,float_format='%.6f')"))
        cells.append(nb.v4.new_code_cell(plot.replace('plt.show()',"fig.savefig(assets/f'{mode}_conditional.pdf',bbox_inches='tight',transparent=True)\nfig.savefig(assets/f'{mode}_conditional.png',bbox_inches='tight',dpi=300)\nplt.show()")))
        cells.append(nb.v4.new_code_cell('gram_diagnostics(data,baseline)'))
    for mode in ('whole','segmented'):
        cells.append(nb.v4.new_markdown_cell(f'## {mode.capitalize()} Gram — paired eigenvalue sensitivity\n\nIndependently multiply the ascending eigenvalues by 0.8, 0.9, 1, 1.1 and 1.2. Hold eigenvectors, fits, prior and data fixed; recompute each location-specific posterior state using common integration draws.'))
        cells.append(nb.v4.new_code_cell(f"mode='{mode}'\ndata=gram_data[mode];records=gram_records[mode];baseline=records[((1.,1.),10,123)]\n"+sens_table+
            "\nsummary.to_csv(assets/f'{mode}_sensitivity.csv',index=False)\ntable.to_latex(assets/f'{mode}_sensitivity.tex',float_format='%.6f')"))
        cells.append(nb.v4.new_code_cell(sens_plot.replace('    plt.show()',"    fig.savefig(assets/f'{mode}_sensitivity.pdf',bbox_inches='tight',transparent=True)\n    fig.savefig(assets/f'{mode}_sensitivity.png',bbox_inches='tight',dpi=300)\n    plt.show()")))
    # Export the existing rank-3 analysis with the identical plot functions too.
    cells.append(nb.v4.new_code_cell("rank_root=Path('results/myocardial_final_redo_e3d965cc4b9b')\n"
        "baseline=pickle.load((rank_root/'scenario_100_100_q10_seed123.pkl').open('rb'))\n"
        "fig,ax=plot_posterior_compact(baseline['ages'],baseline['conditional']);ax.set_xlim(26,95.1)\n"
        "fig.savefig(assets/'rank3_conditional.pdf',bbox_inches='tight',transparent=True);fig.savefig(assets/'rank3_conditional.png',bbox_inches='tight',dpi=300);plt.close(fig)\n"
        "mode='rank3';ages=baseline['ages']\n"
        "scenarios={(level_order[i],level_order[j]):pickle.load((rank_root/f'scenario_{round(100*u)}_{round(100*v)}_q10_seed123.pkl').open('rb')) for i,u in enumerate(LEVELS) for j,v in enumerate(LEVELS)}\n"
        "plot_sensitivity('conditional','Conditional posterior probability')\n"
        "for filename in ['baseline_probabilities_1024','sensitivity_summary_1024']:\n    t=pd.read_csv(rank_root/(filename+'.csv'));t.to_csv(assets/('rank3_'+filename+'.csv'),index=False);t.to_latex(assets/('rank3_'+filename+'.tex'),index=False,float_format='%.6f')\n"
        "print('All three canonical analyses and publication exports complete:',assets)"))
    for c in cells:
        if c.cell_type=='code':compile(c.source,NAME,'exec')
    out=nb.v4.new_notebook(cells=cells,metadata=old.metadata)
    nb.validate(out);nb.write(out,NAME)
    print(NAME)

if __name__=='__main__':build()
