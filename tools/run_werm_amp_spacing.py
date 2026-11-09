"""Repeat the frozen WERM/AMP comparison at gaps n/20 and n/10.

The two spacings share corrected AMP label handling, unknown-count truth
(.25,.50), and the same data, seeds and all other parameters.
Launches two checkpointed runs on disjoint CPU sets, then checks pairing and
refreshes the notebook without launching additional fits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tools import werm_amp_experiment as experiment


def audit_pairing(roots, reference=None):
    """Verify common datasets, AMP seeds, gaps and known counts in every fit."""
    import numpy as np
    import pandas as pd
    payloads = {d:experiment.load_results(p) for d,p in roots.items()}
    specifications = [p['specification'] for p in payloads.values()]
    original = dict(specifications[0])
    original.pop('separation_fraction')
    tables, records = [], {}
    for divisor,payload in payloads.items():
        spec = payload['specification']
        comparison = dict(spec)
        comparison.pop('separation_fraction')
        assert comparison == original
        assert spec['separation_fraction'] == 1/divisor
        assert spec['amp_iterations'] == 10 and not spec['whitening']
        assert experiment.status(roots[divisor])['complete']
        experiment.verify_sources(spec)
        indexed = {}
        for r in payload['records']:
            key = (r['study'],r['covariance'],r['delta'],r['trial'])
            assert key not in indexed
            indexed[key] = r
            np.testing.assert_array_equal(r['truth'],np.rint(
                r['n']*np.asarray(spec[r['study']+'_truth'])).astype(int))
            for method,fit in r['methods'].items():
                if fit['status'] != 'ok':
                    continue  # Failed fits remain visible in summaries.
                assert np.all(np.diff(fit['cp']) >= experiment.minimum_gap(r['n'],spec))
                if r['study'] == 'known':
                    assert fit['count'] == 3
                if method == 'AMP':
                    assert fit['iterations_completed'] == 10
                    assert fit['design_transform'] == 'none'
        records[divisor] = indexed
        table = experiment.summarize(payload)
        table.insert(0,'spacing_divisor',divisor)
        tables.append(table)
    a,b = list(records.values())
    assert a.keys() == b.keys()
    for key in a:
        assert a[key]['data_hash'] == b[key]['data_hash']
        assert a[key]['latent_hash'] == b[key]['latent_hash']
        assert a[key]['methods']['AMP']['seed'] == b[key]['methods']['AMP']['seed']
    compared_reference = 0
    compared_reference_latents = 0
    if reference and Path(reference).is_dir():
        for r in experiment.load_results(reference)['records']:
            key = (r['study'],r['covariance'],r['delta'],r['trial'])
            if key in a:
                # The original reference had unknown truth (.20,.50);
                # its known-count data are unchanged, while unknown-count
                # rows share the latent draws but use the requested new labels.
                assert r['latent_hash'] == a[key]['latent_hash']
                compared_reference_latents += 1
                if r['study'] == 'known':
                    assert r['data_hash'] == a[key]['data_hash']
                    compared_reference += 1
                assert r['methods']['AMP']['seed'] == a[key]['methods']['AMP']['seed']
    report = dict(paired_datasets=len(a),reference_datasets_verified=compared_reference,
                  reference_latent_draws_verified=compared_reference_latents,
                  data_identical_across_spacings=True,amp_seeds_identical=True,
                  only_specification_change='separation_fraction',
                  amp_receives_design_covariance=False,
                  runs={str(d):dict(root=str(p),**experiment.status(p))
                        for d,p in roots.items()})
    summary = pd.concat(tables,ignore_index=True)
    for root in roots.values():
        (root/'spacing_pairing_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        summary.to_csv(root/'spacing_comparison_summary.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report


def refresh_notebook(roots):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    from tools.build_werm_amp_notebook import build
    path = build(spacing_roots=roots)
    notebook = nbformat.read(path,as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher',
        '-f','{connection_file}'],display_name='Python (weighted-erm)',language='python')
    try:
        NotebookClient(notebook,timeout=1800,km=manager,skip_cells_with_tag='baseline-plots',
            resources={'metadata':{'path':str(experiment.ROOT)}}).execute()
        temporary = path.with_suffix('.ipynb.tmp')
        nbformat.write(notebook,temporary)
        temporary.replace(path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print('Notebook refreshed:',path,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-per-spacing',type=int,default=8)
    parser.add_argument('--audit-and-plot-only',action='store_true')
    args = parser.parse_args()
    roots = {d:experiment.prepare({'separation_fraction':1/d,
                                  'unknown_truth':[.25,.5]}) for d in (20,10)}
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    jobs = min(args.jobs_per_spacing,max(1,len(cpus)//2)) if cpus else args.jobs_per_spacing
    if jobs < 1:
        raise ValueError('At least one worker per spacing is required')
    children = []
    if not args.audit_and_plot_only:
        for i,(divisor,root) in enumerate(roots.items()):
            command = [sys.executable,'-m','tools.werm_amp_experiment',
                       '--root',str(root),'--jobs',str(jobs)]
            allocated = cpus[i*jobs:(i+1)*jobs]
            if allocated:
                # Set affinity inside the new interpreter, before scientific
                # imports; all its workers inherit the disjoint CPU allowance.
                bootstrap = ('import os,sys; os.sched_setaffinity(0,'
                             +repr(set(allocated))+'); os.execv(sys.executable,sys.argv[1:])')
                command = [sys.executable,'-c',bootstrap,*command]
            log = (root/'spacing_run.log').open('a')
            child = subprocess.Popen(command,cwd=experiment.ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((divisor,child,log))
            print(f'Gap n/{divisor}: {root}; controller PID {child.pid}; workers {jobs}',flush=True)
        errors = []
        for divisor,child,log in children:
            code = child.wait()
            log.close()
            if code:
                errors.append((divisor,code))
        if errors:
            raise RuntimeError(f'Experiment process errors: {errors}; inspect spacing_run.log')
    audit_pairing(roots,experiment.ROOT/'results/WERM_AMP_comparison/8490596ee1d3')
    refresh_notebook(roots)


if __name__ == '__main__':
    main()
