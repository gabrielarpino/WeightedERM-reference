"""Execute the three requested estimator studies in series, without monitoring."""
from pathlib import Path
import json,time
from tools.build_myocardial_estimator_queue_notebooks import NAMES
from tools import execute_myocardial_rho_final as runner

STATUS=Path('results/myocardial_estimator_queue_status.json')


def update(**values):
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),**values),indent=2));tmp.replace(STATUS)


def main():
    completed=[];failed={}
    for name in NAMES:
        runner.PATH=(Path.cwd()/name)
        for attempt in range(1,4):
            update(state='running',notebook=name,attempt=attempt,completed=completed,failed=failed)
            try:
                runner.main()
                completed.append(name)
                break
            except Exception as exc:
                print('Attempt failed:',name,attempt,repr(exc),flush=True)
                if attempt==3:failed[name]=repr(exc)
    update(state='complete' if not failed else 'finished_with_errors',completed=completed,failed=failed)


if __name__=='__main__':main()
