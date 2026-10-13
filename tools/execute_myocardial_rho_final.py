"""Execute the notebook's own code, persisting each completed cell and any errors.

No experimental logic lives here. Run using .venv-phase1/bin/python.
"""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'Myocardial Infarction Experiment - rho sensitivity final.ipynb'

def main():
    nb = nbformat.read(PATH, 4)
    # A rerun must not show old results beneath an in-progress/new computation.
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    def save(**kwargs):
        temporary = PATH.with_suffix('.executing.ipynb')
        nbformat.write(nb, temporary)
        temporary.replace(PATH)
        if 'cell_index' in kwargs:
            print('Completed cell', kwargs['cell_index'], flush=True)
    save()
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [str(ROOT / '.venv-phase1/bin/python'), '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(nb, km=km, timeout=None, resources={'metadata': {'path': str(ROOT)}},
                            on_cell_executed=save)
    try:
        client.execute()
    finally:
        save()
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    print('FINAL NOTEBOOK COMPLETE', flush=True)

if __name__ == '__main__':
    main()
