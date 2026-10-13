"""Restore original age-axis geometry without adding invalid posterior states."""
from pathlib import Path
import nbformat as nb


def match_original_spacing(cells):
    note=('\n\nThe original age-26 stem represented split index 0 (a no-change boundary), '
          'which is excluded when conditioning on one change point. Delta=2 also '
          'excludes the age-27 split at index 1. No artificial stems are added. '
          'Age 26 is the left axis boundary; no artificial stem is added there. '
          'Figure size, font settings, label/tick sizes, '
          'DPI, and stem/marker styling match the original plotting function.')
    if 'original age-26 stem' not in cells[6].source:cells[6].source+=note
    cells[6].source=cells[6].source.replace(
        'The x-axis retains the original limits 22.7–95.3, including the space before the age-26 tick. ',
        'Age 26 is the left axis boundary; no artificial stem is added there. ')
    cells[7].source=cells[7].source.replace(
        '# Original autoscale: ages 26–92, padded by 5% at each end.\n# Fix the same geometry even though the conditional support starts at 30.\nax.set_xlim(22.7, 95.3)',
        '# Restore the previous conditional-plot limits, with age 26 at the left boundary.\nax.set_xlim(26, 95.1)')
    if 'ax.set_xlim(26, 95.1)' not in cells[7].source:
        cells[7].source=cells[7].source.replace('plt.show()',
            '# Restore the previous conditional-plot limits, with age 26 at the left boundary.\n'
            'ax.set_xlim(26, 95.1)\nplt.show()')
    return cells


def update():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    path=Path('Myocardial Infarction Experiment - final redo.ipynb')
    notebook=nb.read(path,4);notebook.cells=match_original_spacing(notebook.cells)
    setup=nb.v4.new_code_cell("import pickle\nbaseline=pickle.load(open('results/myocardial_final_redo_e3d965cc4b9b/scenario_100_100_q10_seed123.pkl','rb'))")
    plot=nb.v4.new_code_cell(notebook.cells[7].source)
    verify=nb.v4.new_code_cell("""import numpy as np
np.testing.assert_allclose(ax.get_xlim(),[26,95.1])
np.testing.assert_allclose(fig.get_size_inches(),[12,2])
assert fig.dpi==72 and ax.xaxis.label.get_fontsize()==22
assert all(t.get_fontsize()==22 for t in ax.get_xticklabels()+ax.get_yticklabels())
assert plt.rcParams['text.usetex'] and plt.rcParams['font.serif']==['Times New Roman']
assert len(ax.containers[0].markerline.get_xdata())==len(baseline['ages'])
""")
    scratch=nb.v4.new_notebook(cells=[setup,plot,verify])
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[str(Path.cwd()/'.venv-phase1/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(scratch,km=km,timeout=120,resources={'metadata':{'path':str(Path.cwd())}}).execute()
    notebook.cells[7].outputs=scratch.cells[1].outputs
    for output in notebook.cells[7].outputs:
        if 'execution_count' in output:output.execution_count=notebook.cells[7].execution_count
    nb.validate(notebook)
    temp=path.with_suffix('.spacing.ipynb');nb.write(notebook,temp);temp.replace(path)
    print('Re-rendered with age 26 as left boundary; geometry/font assertions passed. Posterior values unchanged.')


if __name__=='__main__':update()
