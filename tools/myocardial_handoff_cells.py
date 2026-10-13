"""Executable final validation and selection cells for the intercept notebook."""
import nbformat as nbf


def intercept_cells(ROOT):
    cells=[nbf.v4.new_markdown_cell(r'''## Independent draws and final bounded penalty resolution

The two closest eligible initial settings used auxiliary logistic L1 strength $10^{-4}$ and
10/30 rho updates. Check independent seed 123 with 1,024 draws. Reuse both sets of states for
the same 22 logarithmic penalties as the main notebook, plus L2 strengths
$6\times10^{-7},8\times10^{-7},1.2\times10^{-6},1.5\times10^{-6},2\times10^{-6}$ around the
promising initial value. These are **outcome-informed exploratory refinements**, not independent
evidence for the selected setting. Keep all 54 settings and both numerical realizations.
Rank settings meeting both probability targets in both realizations by average TV to the archive.
''')]
    source=(ROOT/'tools/myocardial_independent_draw_check.py').read_text().split("if __name__=='__main__':")[0]
    source=source.replace('from tools.myocardial_exploration import *','')
    cells.append(nbf.v4.new_code_cell(source+'\ncheck_intercept(output)'))
    source=(ROOT/'tools/myocardial_intercept_penalty_sweep.py').read_text().split("if __name__=='__main__':")[0]
    source=source.replace('from tools.myocardial_intercept_exploration import *','')
    cells.append(nbf.v4.new_code_cell(source+'\nresolved=intercept_penalty_sweep(output)'))
    cells.append(nbf.v4.new_code_cell(r'''
display(pd.read_csv(output/'independent_draw_check.csv'))
rows=[]
for rec in resolved:
    row={k:v for k,v in rec.items() if k!='conditional'}
    row['TV_to_reference']=.5*np.abs(rec['conditional']-reference_vector).sum()
    rows.append(row)
resolution=pd.DataFrame(rows)
resolution.to_csv(output/'penalty_resolution'/'summary_with_reference.csv',index=False)
ranked=resolution.groupby(['case','posterior_fit']).agg(P1_min=('P_1','min'),P1_max=('P_1','max'),
    mass_above55_min=('mass_above55','min'),mean_TV=('TV_to_reference','mean'),max_TV=('TV_to_reference','max'),
    moment_residual=('moment_residual','first'),max_fit_kkt=('max_fit_kkt','max'),FP_residual_max=('FP_residual_max','max')).reset_index()
ranked['meets_both_in_both']=(ranked.P1_min>.5)&(ranked.mass_above55_min>.5)
ranked=ranked.sort_values('mean_TV')
ranked.to_csv(output/'penalty_resolution'/'ranked_settings.csv',index=False)
print('Full final grid:')
with pd.option_context('display.max_rows',120):display(resolution)
print('Eligible in both realizations; closest curves first (exploratory):')
display(ranked[ranked.meets_both_in_both])
print('Conditional prior mass above 55:',np.mean(ages[locations]>55))
leaders=ranked[ranked.meets_both_in_both].head(3)
for row in leaders.itertuples():
    pairs=[r for r in resolved if r['case']==row.case and r['posterior_fit']==row.posterior_fit]
    print(row.case,row.posterior_fit,'P(1) range:',row.P1_min,row.P1_max,
          '; minimum P(age>55 | 1):',row.mass_above55_min,'; mean TV:',row.mean_TV)
    fig,ax=plt.subplots(figsize=(12,3))
    ax.plot(ref_ages,ref_mass,color='.6',ls='--',lw=1.5,label='Archived conditional')
    for rec in pairs:
        color='black' if rec['seed_label']=='seed42_q9' else '#377eb8'
        ax.plot(ages[locations],rec['conditional'],color=color,lw=1.6,label=rec['seed_label'])
    ax.axvline(55,color='.7',ls=':',lw=1);ax.set_xlabel('Age');ax.set_ylabel('Probability')
    ax.set_xticks([26,40,50,60,75,81,92]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.legend(fontsize=10,frameon=False)
    fig.tight_layout();plt.show()
if len(leaders):
    winner=leaders.iloc[0]
    chosen=next(r for r in resolved if r['case']==winner.case and r['posterior_fit']==winner.posterior_fit)
    source=pickle.load((output/(winner.case+'.pkl')).open('rb'))
    recipe=dict(selection='Exploratory: targets in both draws, then lowest mean TV to archive',
        centered_features=True,unpenalized_intercepts=True,Delta=1,count_prior=[.5,.5],
        auxiliary_loss='logistic',auxiliary_penalty='l1',auxiliary_penalty_strength=1e-4,
        rho_updates=int(source['scenario']['iteration']),rho_damping=.25,
        rho=source['scenario']['rho'].tolist(),rho_moment_residual=float(winner.moment_residual),
        posterior_loss='logistic',posterior_penalty=chosen['penalty'],posterior_penalty_strength=chosen['strength'],
        response_model='global-prevalence calibrated logistic offsets',response_rate=float(y.mean()),
        P1_range=[float(winner.P1_min),float(winner.P1_max)],mass_above55_min=float(winner.mass_above55_min),
        mean_TV_to_archive=float(winner.mean_TV),state_draws=[512,1024],state_seeds=[42,123],state_max_nfev=100)
    (output/'selected_exploratory_setup.json').write_text(json.dumps(recipe,indent=2))
    print('Selected exploratory recipe, not a calibration claim:')
    print(json.dumps(recipe,indent=2))
'''))
    return cells
