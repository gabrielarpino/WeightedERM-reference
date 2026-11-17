"""Untuned dimension/trial-count follow-up to the initial M2 CV-SE study."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

from tools import werm_cv_theory_experiment as base
import numpy as np

PARENT=Path('results/E-WERM-CV-theory/ed5fc8b279bb')


def prepare(p=200,trials=300,theory_draws=20):
    if p<200 or p%10:
        raise ValueError('Use a feature dimension >=200 divisible by ten')
    previous=json.loads((PARENT/'specification.json').read_text())
    for name,digest in previous['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    spec=copy.deepcopy(previous)
    spec.update(schema=2,parent=str(PARENT),trials=trials,theory_draws=theory_draws,
        min_training_segment_rows=1.2*p,
        purpose='Dimension/trial convergence; unchanged M2 generator, ratios, truth and CV-SE procedure')
    spec['settings']=copy.deepcopy(previous['settings'][:3])
    for s in spec['settings']:
        s['p']=p;s['n']=int(s['delta']*p)
        # Preserve exactly the same relative constraints as the original study.
        assert np.ceil(s['minimum_segment_fraction']*(s['n']*9//10))/p>=1.2-1e-12
    spec['sources']['tools/m2_cv_theory_convergence.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results/M2_CV_theory_convergence')/f'p{p}_{digest}'
    root.mkdir(parents=True,exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    weights={}
    for s in spec['settings']:
        for n in (s['n'],s['n']*9//10):
            gap=max(base.empirical.minimum_gap(n,s['delta_fraction']),int(np.ceil(s['minimum_segment_fraction']*n)))
            for k in range(1,s['max_segments']+1):
                weights[n,k,gap]=base.uniform_marginals(n,k,gap)
    if not (root/'weights.pkl').exists():base.original_el.atomic_pickle(root/'weights.pkl',weights)
    if p==200 and theory_draws==previous['theory_draws']:
        for i in range(3):
            for t in range(min(trials,previous['trials'])):
                name=f'setting{i}_trial{t:03d}.pkl'
                if not (root/name).exists():(root/name).write_bytes((PARENT/name).read_bytes())
    return root,spec,weights


if __name__=='__main__':
    import argparse,time
    parser=argparse.ArgumentParser()
    parser.add_argument('--p',type=int,default=200)
    parser.add_argument('--trials',type=int,default=300)
    parser.add_argument('--jobs',type=int,default=12)
    args=parser.parse_args()
    root,spec,weights=prepare(args.p,args.trials)
    print('Results:',root,flush=True)
    start=time.perf_counter()
    result=base.run(root,spec,weights,jobs=args.jobs)
    print('Completed',len(result['records']),'paired trials in',time.perf_counter()-start,'seconds.',flush=True)
