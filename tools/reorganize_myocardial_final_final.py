"""Keep each final analysis together and state achieved moment residuals explicitly."""
from pathlib import Path
import nbformat as nb

PATH=Path('Myocardial Infarction Experiment - final_final.ipynb')

def residual_cell(case):
    if case=='A':
        calculation='residual_value=rho_residual\nrelative_value=relative_rho_residual\n'
        caveat='This is an achieved mismatch after 20 updates, not a prescribed tolerance or an error bound on rho_hat.'
    else:
        calculation="prepared=data['prepared']\nV_aux=prepared['auxiliary']['V']\nS_aux=V_aux.T@V_aux/len(V_aux)\naux_state=state_at(prepared,data['rho'],prepared['psi'],power=10,seed=123,loss=prepared['loss'])\nresidual_value=float(np.linalg.norm(aux_state['moment']-S_aux,'fro'))\nrelative_value=residual_value/np.linalg.norm(S_aux,'fro')\n"
        caveat='The Gram estimator is not constructed to solve these moment equations: this is a diagnostic mismatch, not a solver tolerance or an error bound on rho_hat.'
    source=calculation+f"""
display(Markdown('**Case {case}: Frobenius-norm moment-equation residual**'))
residual_table=pd.DataFrame([{{'Frobenius-norm moment-equation residual':residual_value,
    'Relative Frobenius-norm moment-equation residual':relative_value}}])
display(residual_table.T.rename(columns={{0:'Value'}}).style.format('{{:.9g}}'))
display(Markdown(f'The achieved Frobenius-norm moment-equation residual is **{{residual_value:.6g}}**, '
    f'or **{{100*relative_value:.2f}}%** relative to the empirical moment Frobenius norm. '
    '{caveat}'))
residual_assets=Path('results/myocardial_final_final_v1/paper_assets')
residual_assets.mkdir(parents=True,exist_ok=True)
residual_table.to_csv(residual_assets/'case_{case}_moment_residual.csv',index=False)
residual_table.to_latex(residual_assets/'case_{case}_moment_residual.tex',index=False,float_format='%.9g')
"""
    return nb.v4.new_code_cell(source)

def main():
    n=nb.read(PATH,4)
    if n.metadata.get('grouped_abc_residuals'):return
    c=n.cells
    assert len(c)==33
    c[0].source=c[0].source.replace('The two Gram baseline sections appear before their sensitivity sections so their principal\nfigures are available immediately.',
        'Each analysis has its baseline, diagnostics, achieved Frobenius-norm moment-equation residual, sensitivity study and exports grouped within its own section.')
    c[0].source+='\n\nThe moment diagnostic is $\\|\\Lambda^T\\hat\\rho\\Lambda+K-S\\|_F$, where $S=V_{\\mathrm{aux}}^TV_{\\mathrm{aux}}/n$. Its relative version divides by $\\|S\\|_F$. It is distinct from the maximum numerical state-equation residual used in posterior evaluation.'
    # The original export cell relied on the sensitivity function last defined in C.
    # Give the relocated A export its own identical function so execution is ordered.
    export=c[32]
    plot_definition=c[28].source.split("ages=baseline['ages']")[0]
    export.source=plot_definition+'\n'+export.source
    export.source=export.source.replace("print('All three canonical analyses and publication exports complete:',assets)","print('Case A exports complete:',assets)")
    c[18].source=c[18].source.replace('## B.','# B.')
    c[22].source=c[22].source.replace('## C.','# C.')
    n.cells=c[:15]+[residual_cell('A')]+c[15:18]+[export]+c[18:22]+[residual_cell('B')]+c[26:29]+c[22:26]+[residual_cell('C')]+c[29:32]
    n.cells.append(nb.v4.new_code_cell("print('All three grouped canonical analyses and publication exports complete:',assets)"))
    for cell in n.cells:
        if cell.cell_type=='code':compile(cell.source,str(PATH),'exec')
    n.metadata['grouped_abc_residuals']=True
    nb.validate(n);tmp=PATH.with_suffix('.reordered.ipynb');nb.write(n,tmp);tmp.replace(PATH)
    print('Grouped A/B/C; added explicit achieved moment residuals.')

if __name__=='__main__':main()
