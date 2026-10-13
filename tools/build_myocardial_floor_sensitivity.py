from pathlib import Path
import nbformat as nb

path=Path('Myocardial Infarction Experiment - smaller rho floors.ipynb')
original=nb.read('Myocardial Infarction Experiment - final redo.ipynb',4)
style=original.cells[7].source.split("fig,ax=plot_posterior_compact(baseline")[0]
cells=[nb.v4.new_markdown_cell(r'''# Smaller eigenvalue floors — rank-3 sensitivity

Change only the minimum eigenvalue in the rho estimation procedure:
1e-4 (baseline), 1e-6, 1e-8, and zero. Keep the identity start, damping .25,
20 updates, maximum eigenvalue100, pseudoinverse cutoff1e-6, LS auxiliary fits,
16-step unpenalized logistic posterior fits, no intercept, Delta=2 and the same
count/location prior. Posterior evaluation: 1,024 scrambled-Sobol draws,
seed123, max100 state evaluations, quadrature160, 36 jobs. Saved baseline fits
are reused; auxiliary rho paths are recomputed. The final-redo notebook is unchanged.

Each raw covariance update and its eigenvalues are saved. When both eigenvalues
are negative, clipping them to the same floor destroys their off-diagonal
structure regardless of how small the floor is. In that case the damped path is
exactly rho_k = [floor + (1-floor) (0.75)^k] I. This is checked numerically for
each run. Zero floor is safe here only because the finite damped iterate remains
positive definite; it is not a proposal to evaluate a singular limiting rho.

The residuals measure equation mismatch, not signal-strength estimation accuracy.
Posterior changes under this sensitivity analysis do not establish calibration.
'''),nb.v4.new_code_cell(Path('tools/myocardial_floor_sensitivity.py').read_text()),
nb.v4.new_code_cell('''from IPython.display import display
root,summary,curves=run()
display(summary.style.format(precision=9))
print('Every update clipped two negative eigenvalues:',summary.all_raw_targets_negative.all())
'''),nb.v4.new_code_cell(style+'''
for result in curves:
    print(f"Eigenvalue floor: {result['floor']:g}; P(1)={result['presence']:.6f}; rho:")
    print(result['rho'])
    fig,ax=plot_posterior_compact(result['ages'],result['conditional'])
    ax.set_xlim(26,95.1)
    plt.show()
fig,ax=plt.subplots(figsize=(8,3))
for result,color,ls in zip(curves,['black','#0072B2','#D55E00','#009E73'],['-','--','-.',':']):
    ax.plot(result['ages'],result['conditional'],color=color,ls=ls,lw=1.5,
            label=f"floor={result['floor']:g}, P(1)={result['presence']:.4f}")
ax.set(xlabel='Age (years)',ylabel='Conditional probability',xlim=(26,95.1))
ax.legend(frameon=False,fontsize=9);fig.tight_layout();plt.show()
''')]
out=nb.v4.new_notebook(cells=cells)
out.metadata['kernelspec']=dict(name='python3',language='python',display_name='Python 3')
for c in cells:
    if c.cell_type=='code':compile(c.source,str(path),'exec')
nb.write(out,path);print(path)
