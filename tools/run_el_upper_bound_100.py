"""Execute the self-contained 100-trial notebook, validate, publish PDFs only."""
from pathlib import Path
import os,sys,json,time,hashlib,pickle,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['EL_TRIALS']='100';os.environ['EL_JOBS']='36'
import nbformat
import numpy as np
from nbclient import NotebookClient
from jupyter_client import KernelManager
OUT=ROOT/'results/E_L_upper_bound_100_run';OUT.mkdir(exist_ok=True)
def status(stage,**kw):
    (OUT/'status.json').write_text(json.dumps(dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),indent=2))
    print(stage,kw,flush=True)
def run(args,cwd=ROOT):
    subprocess.run(args,cwd=cwd,check=True)
def execute(cells):
    mini=nbformat.v4.new_notebook(cells=cells)
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
    manager.transport='ipc';sock=tempfile.mkdtemp(prefix='el100-kernel-');manager.ip=str(Path(sock)/'socket')
    try:
        NotebookClient(mini,timeout=None,km=manager,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
        shutil.rmtree(sock,ignore_errors=True)
    return mini.cells
try:
    path=ROOT/'E_L_upper_bound.ipynb';nb=nbformat.read(path,4)
    # Verify completed checkpoints before reusing any of them.
    old=pickle.load((OUT/'study_20_trials.pkl').open('rb'))['rows'][(1000,100)]
    for row in old:
        cached=pickle.load((ROOT/f"results/E_L_upperbound_gap10_n1000_p100/n1000_p100_trial{row['trial']:03d}.pkl").open('rb'))
        assert (cached['n'],cached['p'],cached['trial'])==(1000,100,row['trial'])
        for L in range(1,7):
            np.testing.assert_array_equal(cached['fits'][L]['cp'],row['fits'][L]['cp'])
            np.testing.assert_array_equal(cached['fits'][L]['scores'],row['fits'][L]['scores'])
    status('smoke_trial_20',cached_trials=len(old))
    definitions=nb.cells[1].source.split("if not (EL_ROOT/'replay_cache_verification.pkl').exists():")[0]
    smoke=execute([nbformat.v4.new_code_cell(definitions+"\nsmoke=el_trial(1000,100,20)\nassert smoke['trial']==20\nprint('Fresh-trial smoke passed for all six bounds.')")])
    nbformat.write(nbformat.v4.new_notebook(cells=smoke),OUT/'smoke.ipynb')
    status('running_100_trials',jobs=36)
    executed=execute(nb.cells[:4])
    current=nbformat.read(path,4)
    assert [c.source for c in current.cells[:4]]==[c.source for c in nb.cells[:4]],'Concurrent notebook edit'
    current.cells[:4]=executed;nbformat.write(current,path)
    study=pickle.load((ROOT/'results/E_L_upper_bound_n1000_p100_gap10/study.pkl').open('rb'))
    rows=study['rows'][(1000,100)]
    assert len(rows)==100 and sorted(r['trial'] for r in rows)==list(range(100))
    for before,after in zip(old,rows):
        assert before['trial']==after['trial']
        for L in range(1,7):
            np.testing.assert_array_equal(before['fits'][L]['cp'],after['fits'][L]['cp'])
            np.testing.assert_array_equal(before['fits'][L]['scores'],after['fits'][L]['scores'])
    summary={}
    for L in range(1,7):
        fits=[r['fits'][L] for r in rows];h=np.array([f['haus'] for f in fits]);h=h[np.isfinite(h)]
        summary[L]=dict(trials=100,correct_count=sum(f['correct_count'] for f in fits),empty=sum(f['empty'] for f in fits),median_haus=float(np.median(h)) if len(h) else None,mean_haus=float(np.mean(h)) if len(h) else None)
    (OUT/'summary_100.json').write_text(json.dumps(summary,indent=2))
    status('rendering_consistent_plots')
    run([sys.executable,'tools/update_consistent_notebook_plots.py','E_L_upper_bound.ipynb'])
    run([sys.executable,'tools/package_consistent_plots.py'])
    status('publishing_figures_only')
    paper=ROOT.parent/'WERM-paper'
    run(['git','pull','--ff-only'],paper)
    tex=paper/'JRSSB-revision-rebuttal/JRSSB_revision.tex';texhash=hashlib.sha256(tex.read_bytes()).hexdigest()
    paths=[]
    for name in ('upper_L_count_line','upper_L_haus_box','upper_L_histograms'):
        rel=Path('JRSSB-revision-rebuttal/figures/consistent_20261103/E_L_upper_bound')/(name+'.pdf')
        assert str(rel) in tex.read_text()
        shutil.copyfile(ROOT/'results/consistent_plots_20261102/E_L_upper_bound'/rel.name,paper/rel)
        paths.append(str(rel))
    assert hashlib.sha256(tex.read_bytes()).hexdigest()==texhash
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=paper,text=True).strip()
    assert not staged,'Unexpected pre-staged paper edits; stop before committing'
    run(['git','add','--',*paths],paper)
    assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=paper,text=True).splitlines())==set(paths)
    run(['git','commit','-m','Update upper-bound sensitivity figures to 100 paired trials'],paper)
    run(['git','push','origin','master'],paper)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=paper,text=True).strip()
    assert hashlib.sha256(tex.read_bytes()).hexdigest()==texhash
    run([sys.executable,'tools/map_consistent_plots_to_paper.py'])
    status('complete',commit=commit,trials=100,overleaf_text_unchanged=True,
           review_note='Manuscript text intentionally retains 20-trial descriptions and old numerical summaries; see summary_100.json.')
except BaseException as exc:
    status('failed',error=repr(exc));raise
