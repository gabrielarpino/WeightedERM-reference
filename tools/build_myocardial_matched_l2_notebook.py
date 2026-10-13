"""Create the queued full-solve matched-L2 notebook and combined summary."""
from pathlib import Path
import nbformat as nbf

PATH=Path('Myocardial Infarction Experiment - fine matched L2.ipynb')


def build():
    intro=r'''# Fine matched-L2 sweep: fully solved logistic fits

Both auxiliary and posterior-input fits minimize the **same** weighted mean logistic
loss plus $\lambda\|\beta\|_2^2$, with the **same strength**, no intercept, and the
same coefficient solution reused in both stages. This is separate from early stopping.
The 35 strengths are eight per decade from $10^{-8}$ to $10^{-4}$, plus
$3\times10^{-4}$ and $10^{-3}$. Coefficients are solved by damped Newton until the
maximum gradient magnitude is at most $10^{-10}$ and the strong-convexity
objective-gap bound $\|\nabla f\|^2/(4\lambda)$ is at most $10^{-8}$.
Failure to meet these criteria raises an error; a finite safety budget is not treated
as a converged fit. The solver was tested against an independent BFGS solution.

All other choices are explicit and unchanged: centered/whitened 110 predictors,
1,692 observations, no intercept, $\Delta=1$, equal prior odds for zero/one change,
uniform location over the 61 admissible distinct-age splits, rho start $I_2$,
damping 0.25, and 1, 3, 10, 30 rho updates. This makes **140 primary comparisons**.
State calculations initially use 256 Sobol draws/seed 42 and at most 60 evaluations.
The four closest curves and two highest-count-probability curves within 0.10 TV of
the best are checked with 512 draws/seed 42 and 1,024 draws/seed 123, at most 100
state evaluations, and 160-node Gaussian quadrature. No P(1)>0.5 cutoff is imposed.

**Full solve refers only to coefficient optimization.** Rho estimation remains a
finite-update heuristic, and the existing approximately unpenalized state equations
are used with the weakly penalized fits. Those approximation/model limitations are
not removed by accurate coefficient solves. Outcome-guided comparison with the
archived curve remains exploratory, not independently calibrated model selection.
'''
    source=Path('tools/myocardial_matched_l2_fine.py').read_text().split("if __name__=='__main__':")[0]
    execute='''base=Path('results/myocardial_finite_iteration_443c17cf2df6')
matched_root=run_full_matched(base)
full_summary=pd.read_csv(matched_root/'summary.csv')
full_validation=pd.read_csv(matched_root/'validation.csv')
full_ranked=pd.read_csv(matched_root/'ranked.csv')
display(full_ranked)
'''
    plot=r'''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
                     'axes.labelsize':12,'figure.dpi':150})
print('All matched strengths: primary numerical accuracy')
fig,axes=plt.subplots(1,3,figsize=(12,3.2))
for updates,data in full_summary.groupby('rho_updates'):
    data=data.sort_values('strength')
    for ax,key in zip(axes,('P_1','TV_to_reference','mass_above55')):
        ax.semilogx(data.strength,data[key],'-D',ms=3,label=f'{updates} rho updates')
for ax,label in zip(axes,('P(1 chgpt)','TV from archived curve','P(age > 55 | 1 chgpt)')):
    ax.set_xlabel(r'Matched L2 strength $\lambda$');ax.set_ylabel(label)
    ax.yaxis.set_major_locator(MaxNLocator(3))
axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()
reference=reference_distribution();a=np.array(sorted(reference));p=np.array([reference[x] for x in a])
for row in full_ranked.itertuples():
    print(f'Both stages: L2={row.strength:g}; {row.case}; '
          f'P(1)={row.P1_min:.4g}–{row.P1_max:.4g}; mean TV={row.mean_TV:.4g}')
    fig,ax=plt.subplots(figsize=(6,2.8))
    ax.plot(a,p,color='0.65',label='Archived conditional posterior')
    for tag,color,label in [('q9_seed42','black','512 draws'),('q10_seed123','#4178a8','1,024 draws')]:
        rec=pickle.load((matched_root/(row.case+'_'+tag+'_score.pkl')).open('rb'))
        ax.plot(rec['ages'],rec['conditional'],color=color,label=label)
    ax.axvline(55,color='0.6',ls=':',lw=.8);ax.set_xlabel('Age');ax.set_ylabel('Probability')
    ax.yaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();plt.show()
display(full_validation)
'''
    combined='''print('Combined numerically checked candidates: early stopping, initializations, and full matched L2')
earlier=pd.read_csv(base/'combined_checked_candidates.csv')
new=[]
for row in full_ranked.to_dict('records'):
    recs=[pickle.load((matched_root/(row['case']+'_'+tag+'_score.pkl')).open('rb'))
          for tag in ('q9_seed42','q10_seed123')]
    row.update(stage='fully_solved_matched_L2',Delta=1,initialization='identity',
        auxiliary=recs[0]['auxiliary'],rho_updates=recs[0]['rho_updates'],
        numerical_curve_TV=.5*np.abs(recs[0]['conditional']-recs[1]['conditional']).sum())
    new.append(row)
all_candidates=pd.concat([earlier,pd.DataFrame(new)],ignore_index=True).sort_values('mean_TV')
all_candidates.to_csv(base/'combined_with_full_l2.csv',index=False)
display(all_candidates)
print('Best curve matches in each class (not a statistical acceptance rule):')
display(all_candidates.groupby('group',sort=False).head(3))
print('Ranges across numerical checks are not confidence intervals. Small TV does not establish calibration.')
'''
    historical='''## Earlier promising alternatives (retained in the final comparison)

| Earlier fit setup | Intercept | P(1) across numerical checks | Main trade-off |
|---|---|---|---|
| LS auxiliary; logistic L1 1e-4 posterior fit; 30 rho updates | No | 0.402–0.453 | Mode 79; original-like but count probability below 0.5 |
| LS auxiliary; logistic L2 1e-6 posterior fit; 3 rho updates | No | 0.99988–0.99998 | Mode 87; too concentrated/late relative to reference |
| LS auxiliary; logistic L2 1e-5 posterior fit; 30 rho updates | No | 0.481–0.484 | Mode 75; diffuse |
| Logistic L1 1e-4 auxiliary; logistic L2 1.5e-6 posterior fit; 30 updates | Yes | 0.718–0.747 | Mode 81; mean reference TV 0.291; mismatched penalties |
| Logistic L1 1e-4 auxiliary; logistic L1 1e-5 posterior fit; 30 updates | Yes | 0.654–0.680 | Mode 81; mean reference TV 0.367; mismatched strengths |

These historical candidates remain exploratory and are not relabeled as satisfying
the no-intercept or exactly matched-penalty preferences when they do not.
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(source),
        nbf.v4.new_code_cell(execute),nbf.v4.new_code_cell(plot),nbf.v4.new_code_cell(combined),
        nbf.v4.new_markdown_cell(historical)])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook_cell','exec')
    nbf.write(nb,PATH);print(PATH)


if __name__=='__main__':build()
