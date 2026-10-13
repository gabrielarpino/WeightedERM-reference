"""Render saved unpenalized results, preserving them across active runner saves."""
from pathlib import Path
import json,time,os
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from tools.add_whole_gram_high_presence import SOURCE as TEMPLATE

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'Myocardial Infarction Experiment - segmented coefficient Gram top 20.ipynb'
TAG='segmented-unpenalized-top20'
CACHE=ROOT/'results/segmented_unpenalized_rendered_cells.ipynb'
SOURCE=TEMPLATE.replace("/whole')","/segmented')").replace(
    "selected=summary.loc[summary.P_1>=0.5].sort_values('TV_to_rank3').head(10).copy()",
    "unpenalized=summary.loc[summary.penalty=='none'].copy()\n"
    "unpenalized['loss_priority']=(unpenalized.aux_loss!='logistic').astype(int)\n"
    "selected=unpenalized.sort_values(['loss_priority','TV_to_rank3','id']).head(20).copy()\n"
    "assert len(selected)==20 and selected.penalty.eq('none').all()\n"
    "print('Logistic auxiliary/segment fits prioritized; within each loss, ascending screening TV. No P(1) threshold.')"
).replace('Higher-P(1) #','Unpenalized #')

def merge(cells):
    notebook=nbformat.read(PATH,4)
    old=[c for c in notebook.cells if TAG in c.metadata.get('tags',[])]
    if old==cells:return
    notebook.cells=[c for c in notebook.cells if TAG not in c.metadata.get('tags',[])]+cells
    nbformat.validate(notebook)
    staged=PATH.with_suffix('.unpenalized.tmp.ipynb')
    nbformat.write(notebook,staged);staged.replace(PATH)

def main(watch=False):
    if not watch:
        cells=[nbformat.v4.new_markdown_cell(
            '## Top 20 unpenalized results — logistic auxiliary/segment fits preferred\n\n'
            'Completed **unbalanced segmented Gram** study. Restrict to zero explicit L1/L2 penalty; '
            'prioritize logistic auxiliary and segment refits, then rank by smallest screening TV '
            'distance to the earlier rank-3 curve. There are 216 logistic candidates, so all twenty '
            'selected rows use logistic. Finite iteration budgets act as implicit regularization. '
            'The solver, budget and intercept choice are matched across the initial auxiliary, '
            'segment and posterior regressions. No P(1) cutoff is imposed; probabilities are displayed '
            'rather than optimized. Screening uses 512 draws; existing 1024-draw rechecks are overlaid '
            'and tabulated where available. Missing rechecks are explicitly marked. This is exploratory '
            'shape-based selection, not evidence of statistical calibration.'),nbformat.v4.new_code_cell(SOURCE)]
        for c in cells:c.metadata['tags']=[TAG]
        rendered=nbformat.v4.new_notebook(cells=cells)
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[str(ROOT/'.venv-phase1/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}']
        try:NotebookClient(rendered,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
        assert sum('image/png' in o.get('data',{}) for c in rendered.cells for o in c.get('outputs',[]))==20
        nbformat.write(rendered,CACHE)
        merge(rendered.cells)
        print('Rendered and appended 20 unpenalized posterior plots and table.',flush=True)
    else:
        cells=nbformat.read(CACHE,4).cells
        while True:
            merge(cells)
            status=json.loads((ROOT/'results/myocardial_estimator_queue_status.json').read_text())
            if status['state'] in ('complete','finished_with_errors'):break
            try:os.kill(3705452,0)
            except ProcessLookupError:break
            time.sleep(30)
        merge(cells)
        print('Final notebook additions preserved.',flush=True)

if __name__=='__main__':
    import sys
    main('--watch' in sys.argv)
