"""Wait for the existing notebook, run the next one, then append its outputs.

Never edits the notebook while its current executor is writing it.
"""
import sys,time,copy
from pathlib import Path
import nbformat
from tools import execute_myocardial_rho_final as runner


def main(pid):
    process=Path(f'/proc/{pid}/cmdline')
    print('Fine matched-L2 full solve queued after the active notebook.',flush=True)
    while process.exists():
        try:
            if b'finite iteration exploration.ipynb' not in process.read_bytes():break
        except FileNotFoundError:break
        time.sleep(10)
    main_path=Path('Myocardial Infarction Experiment - finite iteration exploration.ipynb')
    current=nbformat.read(main_path,4)
    assert not any(o.output_type=='error' for c in current.cells if c.cell_type=='code' for o in c.outputs), 'Previous notebook failed; inspect it before advancing.'
    assert all(c.execution_count is not None for c in current.cells if c.cell_type=='code'), 'Previous notebook did not execute every experiment cell.'
    assert Path('results/myocardial_finite_iteration_443c17cf2df6/combined_checked_candidates.csv').exists(), 'Previous combined summary not ready.'
    runner.PATH=Path.cwd()/'Myocardial Infarction Experiment - fine matched L2.ipynb'
    runner.main()
    # The active notebook is finished, so it is now safe to append the new
    # visible code and executed plots, preserving every previous output.
    current=nbformat.read(main_path,4);extra=nbformat.read(runner.PATH,4)
    tag='## Queued follow-up: fine matched-L2 full solves'
    if not any(c.cell_type=='markdown' and c.source==tag for c in current.cells):
        current.cells.append(nbformat.v4.new_markdown_cell(tag))
        current.cells.extend(copy.deepcopy(extra.cells))
        temporary=main_path.with_suffix('.appending.ipynb')
        nbformat.write(current,temporary);temporary.replace(main_path)
    print('Matched L2 complete; plots and combined final summary appended to the main notebook.',flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
