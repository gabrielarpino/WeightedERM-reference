"""Paired T=20 AMP follow-up, reusing the unchanged T=10 WERM fits.

The core data generator, AMP recurrence, priors and WERM code are untouched.
Every T=20 AMP run starts from the saved T=10 seed; its first ten diagnostic
entries are checked against the original trajectory. No earlier iterate is
selected in place of the requested final iterate.
"""
from tools import werm_amp_experiment as experiment

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
from io import StringIO
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd


def load_reference_roots():
    parent = experiment.ROOT/'results/WERM_AMP_comparison'
    return {20:parent/'7a5275632c7b',10:parent/'e8c4b4c6af5f'}


def _matching_specifications(reference, current):
    before,after = dict(reference),dict(current)
    assert before.pop('amp_iterations') == 10
    assert after.pop('amp_iterations') == 20
    assert before == after, 'Only AMP iteration count may change'


def prepare_iterations():
    roots = {}
    for gap,reference in load_reference_roots().items():
        old = json.loads((reference/'specification.json').read_text())
        experiment.verify_sources(old)
        root = experiment.prepare({'separation_fraction':1/gap,
                                    'unknown_truth':[.25,.5],'amp_iterations':20})
        spec = json.loads((root/'specification.json').read_text())
        _matching_specifications(old,spec)
        for filename in ('weights.pkl','covariances.pkl'):
            a = pickle.loads((reference/filename).read_bytes())
            b = pickle.loads((root/filename).read_bytes())
            assert a.keys() == b.keys()
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
        provenance = dict(reference_root=str(reference),amp_iterations=20,
            werm='copied without refitting from paired T=10 records',
            amp='rerun from the same initialization and Monte Carlo seed',
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path = root/'iteration_provenance.json'
        if path.exists():
            assert json.loads(path.read_text()) == provenance, 'Iteration driver changed'
        else:
            path.write_text(json.dumps(provenance,indent=2)+'\n')
        roots[gap] = root
    return roots


def run_amp_trial(root, reference_path):
    root,reference_path = Path(root),Path(reference_path)
    output_path = root/reference_path.name
    if output_path.exists():
        return str(output_path)
    started = time.perf_counter()
    old_bytes = reference_path.read_bytes()
    old = pickle.loads(old_bytes)
    spec = json.loads((root/'specification.json').read_text())
    weights = pickle.loads((root/'weights.pkl').read_bytes())
    matrices = pickle.loads((root/'covariances.pkl').read_bytes())
    X,y,truth,latent_hash = experiment.generate_data(spec,old['study'],
        old['covariance'],old['delta'],old['trial'],matrices)
    data_hash = hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
    assert data_hash == old['data_hash'] and latent_hash == old['latent_hash']
    np.testing.assert_array_equal(truth,old['truth'])
    seed = old['methods']['AMP']['seed']
    n,p = X.shape
    try:
        from tools.werm_amp_adapter import fit_amp
        phi = weights[n,spec['max_segments']] if old['study']=='known' else weights[n,'mixture']
        with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            amp = fit_amp(X,y,delta=n/p,noise_std=spec['noise_std'],
                prior_cov=(n/p)*np.eye(spec['max_segments']),phi=phi,
                gap=experiment.minimum_gap(n,spec),seed=seed,
                iterations=20,mc_samples=spec['amp_mc_samples'],
                fixed_count=old['study']=='known')
    except Exception:
        amp = dict(status='failed',seed=seed,iterations_requested=20,
                   error=traceback.format_exc())
    old_diagnostics = old['methods']['AMP']['diagnostics']
    new_diagnostics = amp.get('diagnostics',[])[:len(old_diagnostics)]
    prefix_complete = len(new_diagnostics)==len(old_diagnostics)
    prefix_exact = prefix_complete and new_diagnostics==old_diagnostics
    if prefix_complete and not prefix_exact:
        for a,b in zip(old_diagnostics,new_diagnostics):
            assert a.keys()==b.keys()
            np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-12,atol=1e-12)
    if amp['status']=='ok':
        assert prefix_complete and amp['iterations_completed']==20
    record = copy.deepcopy(old)
    record['spec_digest'] = root.name
    record['methods']['AMP'] = experiment.add_metrics(amp,truth,n)
    record['amp_seconds'] = time.perf_counter()-started
    record['seconds'] = record['amp_seconds']
    record['seconds_scope'] = 'new AMP computation only; WERM reused'
    record['iteration_comparison'] = dict(reference_file=str(reference_path),
        reference_sha256=hashlib.sha256(old_bytes).hexdigest(),
        werm_reused=True,first_ten_complete=prefix_complete,first_ten_exact=prefix_exact)
    assert hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()==data_hash
    for method in old['methods']:
        if method!='AMP':
            assert pickle.dumps(record['methods'][method])==pickle.dumps(old['methods'][method])
    experiment.atomic_pickle(output_path,record)
    return str(output_path)


def _save_summary(root):
    payload = experiment.load_results(root)
    experiment.summarize(payload).to_csv(root/'summary.csv',index=False)
    (root/'status.json').write_text(json.dumps(experiment.status(root),indent=2)+'\n')
    return payload


def run_iterations(roots=None,jobs=16):
    roots = prepare_iterations() if roots is None else {int(k):Path(v) for k,v in roots.items()}
    references = load_reference_roots()
    tasks = []
    for gap,root in roots.items():
        spec = json.loads((root/'specification.json').read_text())
        experiment.verify_sources(spec)
        _matching_specifications(json.loads((references[gap]/'specification.json').read_text()),spec)
        for path in sorted(references[gap].glob('*__t*.pkl')):
            if not (root/path.name).exists():
                tasks.append((str(root),str(path)))
    # Interleave gaps and trials while retaining deterministic individual seeds.
    tasks.sort(key=lambda pair:(Path(pair[1]).name.split('__t')[-1],Path(pair[1]).name,pair[0]))
    print(f'T=20: {len(tasks)} remaining AMP fits; {jobs} workers',flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),
                             initializer=experiment._worker_init) as pool:
        futures = [pool.submit(run_amp_trial,*task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            path = Path(future.result())
            record = pickle.loads(path.read_bytes())
            fit = record['methods']['AMP']
            print(f'{i}/{len(tasks)} {path.parent.name}/{path.stem}: {fit["status"]}'
                  +(' '+fit.get('error','') if fit['status']!='ok' else ''),flush=True)
            if i%48==0 or i==len(tasks):
                for root in roots.values():
                    _save_summary(root)
    for root in roots.values():
        payload = _save_summary(root)
        experiment.atomic_pickle(root/'study.pkl',payload)
    print(f'AMP follow-up completed in {time.perf_counter()-started:.1f}s',flush=True)
    return roots


def audit_iterations(roots):
    from tools.werm_amp_iteration_plots import paired_summary
    references = load_reference_roots()
    reports,tables = {},[]
    for gap,root in roots.items():
        old = experiment.load_results(references[gap])
        new = experiment.load_results(root)
        _matching_specifications(old['specification'],new['specification'])
        experiment.verify_sources(new['specification'])
        assert experiment.status(root)['complete']
        before = {(r['study'],r['covariance'],r['delta'],r['trial']):r for r in old['records']}
        after = {(r['study'],r['covariance'],r['delta'],r['trial']):r for r in new['records']}
        assert before.keys()==after.keys() and len(after)==2400
        successes,empty,prefix_exact = 0,0,0
        for key,r in after.items():
            b = before[key]
            assert r['data_hash']==b['data_hash'] and r['latent_hash']==b['latent_hash']
            assert r['methods']['AMP']['seed']==b['methods']['AMP']['seed']
            np.testing.assert_array_equal(r['truth'],b['truth'])
            for method in r['methods']:
                if method!='AMP':
                    assert pickle.dumps(r['methods'][method])==pickle.dumps(b['methods'][method])
            amp = r['methods']['AMP']
            prefix_exact+=r['iteration_comparison']['first_ten_exact']
            if amp['status']=='ok':
                successes+=1
                empty+=amp['empty']
                assert amp['iterations_requested']==amp['iterations_completed']==20
                assert amp['design_transform']=='none'
                assert np.all(np.diff(amp['cp'])>=experiment.minimum_gap(r['n'],new['specification']))
                if r['study']=='known': assert amp['count']==3
        report = dict(gap_divisor=gap,paired_datasets=len(after),amp_successes=successes,
            amp_failures=len(after)-successes,amp_empty_estimates=empty,
            exact_first_ten_trajectories=int(prefix_exact),
            werm_results_identical=True,data_and_seeds_identical=True,
            only_parameter_change='amp_iterations: 10 -> 20')
        reports[gap]=report
        (root/'iteration_pairing_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        table = paired_summary(old,new)
        table.insert(0,'gap_divisor',gap)
        table.to_csv(root/'paired_iteration_summary.csv',index=False)
        tables.append(table)
    print(json.dumps(reports,indent=2),flush=True)
    return reports


def plot_amp_iteration_comparison(*args,**kwargs):
    from tools.werm_amp_iteration_plots import plot_amp_iteration_comparison as plot
    return plot(*args,**kwargs)


def paired_summary(*args,**kwargs):
    from tools.werm_amp_iteration_plots import paired_summary as summary
    return summary(*args,**kwargs)


def refresh_notebook(roots):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    from tools.build_werm_amp_notebook import build
    path = build(t20_roots=roots)
    notebook = nbformat.read(path,as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher',
        '-f','{connection_file}'],display_name='Python (weighted-erm)',language='python')
    try:
        NotebookClient(notebook,timeout=1800,km=manager,skip_cells_with_tag='existing-plots',
                       resources={'metadata':{'path':str(experiment.ROOT)}}).execute()
        temporary = path.with_suffix('.ipynb.tmp')
        nbformat.write(notebook,temporary)
        temporary.replace(path)
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    print('Notebook refreshed:',path,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=16)
    parser.add_argument('--plot-only',action='store_true')
    args = parser.parse_args()
    roots = prepare_iterations()
    print('T=20 results:',{gap:str(root) for gap,root in roots.items()},flush=True)
    if not args.plot_only:
        run_iterations(roots,args.jobs)
    audit_iterations(roots)
    refresh_notebook(roots)


if __name__=='__main__':
    main()
