"""Append saved LS-only screening results without repeating any experiment."""
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from tools import add_segmented_unpenalized as common
from tools.add_whole_gram_high_presence import SOURCE as TEMPLATE

SOURCE=TEMPLATE.replace("/whole')","/segmented')").replace(
    "selected=summary.loc[summary.P_1>=0.5].sort_values('TV_to_rank3').head(10).copy()",
    "selected=summary.loc[(summary.penalty=='none') & (summary.aux_loss=='squared')].sort_values(['TV_to_rank3','id']).head(20).copy()\n"
    "assert len(selected)==20 and selected.penalty.eq('none').all() and selected.aux_loss.eq('squared').all()"
).replace('Higher-P(1) #','Unpenalized LS #')

def main():
    cells=[nbformat.v4.new_markdown_cell(
        '## Top 20 unpenalized least-squares auxiliary/segment results\n\n'
        'Completed **unbalanced segmented Gram** study. Both the initial auxiliary fit and '
        'the estimated-segment refits use exact, unpenalized least squares. The posterior '
        'WERM regression uses unpenalized finite-iteration logistic; the displayed solver '
        'and iteration budget refer to that logistic fit, not to LS. Select the twenty '
        'smallest **512-draw screening TV distances** to the earlier rank-3 curve, with no '
        'P(1) cutoff. All these LS screening configurations have P(1) numerically near one; '
        'that is not evidence of calibration. Available 1024-draw rechecks are tabulated '
        'and overlaid, and missing checks are explicitly marked. The best screening '
        'candidate had TV about 0.890, worsening to about 1.000 on recheck; the ranking '
        'must not be interpreted as a validated good match.'),
        nbformat.v4.new_code_cell(SOURCE)]
    for c in cells:c.metadata['tags']=[common.TAG,'segmented-ls-unpenalized-top20']
    rendered=nbformat.v4.new_notebook(cells=cells)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[str(common.ROOT/'.venv-phase1/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(rendered,km=km,timeout=180,resources={'metadata':{'path':str(common.ROOT)}}).execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    assert sum('image/png' in o.get('data',{}) for c in rendered.cells for o in c.get('outputs',[]))==20
    cached=nbformat.read(common.CACHE,4)
    cached.cells=[c for c in cached.cells if 'segmented-ls-unpenalized-top20' not in c.metadata.get('tags',[])]+rendered.cells
    nbformat.write(cached,common.CACHE)
    common.merge(cached.cells)
    print('Added LS-only table and 20 rendered plots; retained logistic section.',flush=True)

if __name__=='__main__':main()
