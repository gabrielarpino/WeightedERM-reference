"""Unknown-count rerun at (0.30, 0.55), preserving the original known-count study."""
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

from tools import m2_adjusted_experiment as base
import numpy as np
from joblib import Parallel, delayed, parallel_config

PARENT = Path('results/M2_adjusted_comparison/ecd9d5c2529d')
UNKNOWN_FRACTIONS = (.3, .55)


def generate_data(setting, trial, p=100):
    """Same original B, X and noise draws; only the response regime labels move."""
    delta = base.DELTAS[setting]
    rng = base.trial_rng('M2c-alpha-0.3', setting, trial)
    n = int(p*delta)
    prior = base.SparseDiffSignal(8, np.sqrt(400*delta), .3, 4)
    B = prior.sample(p, rng=rng)
    X = rng.normal(0, np.sqrt(1/n), (n,p))
    noise = rng.normal(size=n)*np.sqrt(.1)
    truth = (n*np.asarray(UNKNOWN_FRACTIONS)).astype(int)
    labels = np.searchsorted(truth, np.arange(n), side='right')
    theta = (X*np.sqrt(n))@(B/np.sqrt(n))
    return X, theta[np.arange(n), labels]+noise, truth


def prepare():
    previous = json.loads((PARENT/'specification.json').read_text())
    for name, digest in previous['sources'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Original estimator source changed: {name}')
    spec = copy.deepcopy(previous)
    spec['schema'] = 2
    spec['experiment'] = 'unknown-count locations 0.30 and 0.55'
    spec['fractions']['unknown'] = list(UNKNOWN_FRACTIONS)
    spec['parent'] = str(PARENT)
    spec['sources']['tools/m2_adjusted_relocated.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    weight_bytes = (PARENT/'weights.pkl').read_bytes()
    spec['weights_sha256'] = hashlib.sha256(weight_bytes).hexdigest()
    digest = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root = PARENT.parent/f'unknown_030_055_{digest}'
    root.mkdir(parents=True,exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    weights_file = root/'weights.pkl'
    if not weights_file.exists():
        weights_file.write_bytes(weight_bytes)
    assert weights_file.read_bytes() == weight_bytes
    return root,spec,pickle.loads(weight_bytes)


def run_trial(setting,trial,root,spec,weights):
    path=root/f'unknown_delta{base.DELTAS[setting]:g}_trial{trial:03d}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    X,y,truth=generate_data(setting,trial,spec['p'])
    record=dict(mode='unknown',setting=setting,trial=trial,n=len(y),truth=truth,
                data_hash=base.data_hash(X,y),methods={})
    for adjusted,name in enumerate(base.METHODS):
        started=time.perf_counter()
        try:
            fitted=base.select_count(X,y,bool(adjusted),weights)
            fitted.update(base.evaluate_changepoints(fitted['cp'],truth,len(y)))
            fitted['error']=None
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            fitted=dict(cp=None,status='failed',haus=np.nan,size=np.nan,error=repr(error))
        fitted['runtime']=time.perf_counter()-started
        record['methods'][name]=fitted
    base.atomic_pickle(path,record)
    return record


def run(root,spec,weights,jobs=12):
    with parallel_config(backend='loky',inner_max_num_threads=1):
        records=Parallel(n_jobs=jobs,verbose=10)(delayed(run_trial)(i,j,root,spec,weights)
            for i in range(len(base.DELTAS)) for j in range(spec['trials']))
    result=dict(spec=spec,records=records,mode='unknown')
    base.atomic_pickle(root/'unknown.pkl',result)
    base.summarize(result).to_csv(root/'unknown_summary.csv',index=False)
    return result


if __name__=='__main__':
    root,spec,weights=prepare()
    print('Results:',root,flush=True)
    result=run(root,spec,weights)
    print(base.summarize(result).to_string(index=False),flush=True)
