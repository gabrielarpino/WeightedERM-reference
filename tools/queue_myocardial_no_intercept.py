"""Run the new notebook only after the existing intercept notebook completes."""
import json
import os
from pathlib import Path
import time
import nbformat
from tools import execute_myocardial_rho_final as executor

PREDECESSOR = Path('Myocardial Infarction Experiment - finite intercept best ones.ipynb')
SUCCESSOR = Path('Myocardial Infarction Experiment - no intercept solver best ones.ipynb')
STATUS = Path('results/myocardial_no_intercept_queue_status.json')


def status(state, **kwargs):
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(state=state,time=time.time(),**kwargs),indent=2))
    temporary.replace(STATUS)
    print(state, kwargs, flush=True)


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def main(pid):
    identity=process_identity(pid)
    status('waiting_for_intercept',pid=pid,identity=identity)
    while identity is not None and process_identity(pid)==identity:
        time.sleep(30)
    nb=nbformat.read(PREDECESSOR,as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code' and c.source.strip()]
    errors=[o for c in cells for o in c.outputs if o.output_type=='error']
    if errors or not cells or any(c.execution_count is None for c in cells):
        status('blocked_predecessor_incomplete',errors=[str(o) for o in errors])
        return
    status('running_no_intercept',notebook=str(SUCCESSOR))
    executor.PATH=SUCCESSOR.resolve()
    try:executor.main()
    except Exception as exc:
        status('failed_no_intercept',error=repr(exc));raise
    status('complete',notebook=str(SUCCESSOR))


if __name__=='__main__':
    import sys
    main(int(sys.argv[1]))
