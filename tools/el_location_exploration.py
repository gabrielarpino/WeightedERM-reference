"""Explore truth locations at frozen gamma=6, then assess on fresh trials.

All pilot settings are retained. Selection uses empirical pilot count errors only;
SE agreement and assessment outcomes do not choose the locations.
"""
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from tools import el_state_evolution as base

PARENT = Path('results/E_L_SE_penalty_fixed/982fa979e869')
GAMMA = 6.
COUNTS = tuple(range(1, 7))
BASELINE = (.2, .5, .8)
GRID = (.15, .30, .45, .60, .75, .90)
CANDIDATES = sorted(set(itertools.combinations(GRID, 2)) |
                    set(itertools.combinations(GRID, 3)) |
                    {BASELINE, (.25, .45, .7), (.3, .55)})


def layout_name(locations):
    return '_'.join(f'{v:.2f}'.replace('.', '') for v in locations)


def generate_data(locations, trial):
    # Always generate the same four-state coefficient path first. Two-change
    # experiments use its first three states, preserving X and noise pairing.
    X, original_y, beta, _, _, original_mu = base.generate_data(trial)
    truth = np.rint(base.N*np.asarray(locations)).astype(int)
    if len(truth) not in (2, 3) or np.any(np.diff(truth) < base.N//10):
        raise ValueError('Use two or three internally separated changes')
    labels = np.searchsorted(truth, np.arange(base.N), side='right')
    beta = beta[:, :len(truth)+1]
    mu = np.einsum('ij,ij->i', X, beta[:, labels].T)
    y = original_y if tuple(locations) == BASELINE else mu + (original_y-original_mu)
    return X, y, beta, truth, labels


def prepare():
    parent_spec = json.loads((PARENT/'specification.json').read_text())
    for name, digest in parent_spec['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    sources = dict(parent_spec['sources'])
    sources['tools/el_location_exploration.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    weight_bytes = (PARENT/'weights.pkl').read_bytes()
    spec = dict(schema=1, parent=str(PARENT), gamma=GAMMA, n=base.N, p=base.P,
        noise_std=base.SIGMA, seed=base.SEED, candidates=CANDIDATES, upper_bounds=COUNTS,
        pilot_trials=list(range(10000,10020)), assessment_trials=list(range(20000,20100)),
        theory_draws=20, weights_sha256=hashlib.sha256(weight_bytes).hexdigest(),
        selection='Within each true count: minimum empirical mean absolute count error '
                  'over L > Lstar; then maximum exact-count frequency; then minimum '
                  'conditional Hausdorff error; then lexicographic locations. Pilot only.',
        sources=sources)
    digest = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root = Path('results/E_L_location_exploration')/digest
    root.mkdir(parents=True,exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    if not (root/'weights.pkl').exists():
        (root/'weights.pkl').write_bytes(weight_bytes)
    assert (root/'weights.pkl').read_bytes() == weight_bytes
    return root, spec, pickle.loads(weight_bytes)


def run_trial(locations, trial, draws, root, weights):
    stage = 'assessment' if draws else 'pilot'
    path = root/f'{stage}_{layout_name(locations)}_{trial}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    X,y,beta,truth,labels = generate_data(locations,trial)
    strength = beta.T@beta/base.N
    record = dict(locations=tuple(locations), trial=trial, truth=truth,
                  data_hash=base.data_hash(X,y), fits={})
    if draws:
        rng = np.random.default_rng(np.random.SeedSequence([base.SEED,trial,917341]))
        standard_signal = rng.normal(size=(draws,base.N,4))
        Z = standard_signal[:,:,:len(truth)+1]@np.linalg.cholesky(strength).T
        response = Z[:,np.arange(base.N),labels]+rng.normal(0,base.SIGMA,(draws,base.N))
        standard_noise = rng.normal(size=(draws,base.N,6))
    for count in COUNTS:
        phi = weights['penalty',count]
        theta = base.empirical_fit(X,y,phi)
        empirical = base.add_metrics(base.search(theta,y,base.N//10,True,GAMMA),truth,base.N)
        result = dict(empirical=empirical,theory=[])
        if draws:
            se = base.squared_se(phi,strength,labels,base.N/base.P,base.SIGMA**2)
            for d in range(draws):
                V = Z[d]@se['lam']+standard_noise[d,:,:count]@se['covariance_root'].T
                predicted = (1-se['omega'])*V+se['omega']*response[d,:,None]
                fit = base.search(predicted,response[d],base.N//10,True,GAMMA)
                result['theory'].append(base.add_metrics(fit,truth,base.N))
            result['se'] = {k:v for k,v in se.items() if k not in ('omega','covariance_root')}
        record['fits'][count] = result
    base.atomic_pickle(path,record)
    return record


def summarize(records, theory=False):
    rows=[]
    for locations in sorted(set(r['locations'] for r in records)):
        subset=[r for r in records if r['locations']==locations]
        for count in COUNTS:
            for source in (('empirical','theory') if theory else ('empirical',)):
                fits=([r['fits'][count]['empirical'] for r in subset] if source=='empirical'
                      else [f for r in subset for f in r['fits'][count]['theory']])
                selected=np.array([f['count'] for f in fits])
                h=np.array([f['haus'] for f in fits]);h=h[np.isfinite(h)]
                rows.append(dict(locations=locations,true_changes=len(locations),L=count,
                    source=source,draws=len(fits),mean_count=selected.mean(),
                    correct_count=np.mean(selected==len(locations)),
                    over_count=np.mean(selected>len(locations)),
                    under_count=np.mean(selected<len(locations)),
                    mean_absolute_count_error=np.mean(np.abs(selected-len(locations))),
                    max_count=np.mean(selected==count-1),empty=int(np.sum(selected==0)),
                    mean_haus=h.mean() if len(h) else np.nan))
    return pd.DataFrame(rows)


def rank_pilots(records):
    rows=[]
    for locations in CANDIDATES:
        fits=[r['fits'][count]['empirical'] for r in records if r['locations']==locations
              for count in COUNTS if count>len(locations)+1]
        selected=np.array([f['count'] for f in fits])
        h=np.array([f['haus'] for f in fits]);h=h[np.isfinite(h)]
        rows.append(dict(locations=locations,true_changes=len(locations),
            mean_absolute_count_error=float(np.mean(np.abs(selected-len(locations)))),
            correct_count=float(np.mean(selected==len(locations))),
            over_count=float(np.mean(selected>len(locations))),
            mean_haus=float(np.mean(h)) if len(h) else float('inf')))
    return sorted(rows,key=lambda r:(r['mean_absolute_count_error'],-r['correct_count'],
                                     r['mean_haus'],r['locations']))


def run(root,spec,weights,jobs=12):
    with parallel_config(backend='loky',inner_max_num_threads=1):
        pilots=Parallel(n_jobs=jobs,verbose=10)(delayed(run_trial)(loc,t,0,root,weights)
            for loc in CANDIDATES for t in spec['pilot_trials'])
    base.atomic_pickle(root/'pilot.pkl',dict(specification=spec,records=pilots))
    summarize(pilots).to_csv(root/'pilot_by_upper_bound.csv',index=False)
    ranking=rank_pilots(pilots)
    pd.DataFrame(ranking).to_csv(root/'pilot_ranking.csv',index=False)
    choices={str(k):next(r['locations'] for r in ranking if r['true_changes']==k) for k in (2,3)}
    frozen=dict(choices=choices,selection=spec['selection'],gamma=GAMMA)
    path=root/'frozen_selection.json'
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(frozen))
    else:
        path.write_text(json.dumps(frozen,indent=2))
    print('Frozen pilot choices:',choices,flush=True)
    assessment_locations=sorted(set([BASELINE,*choices.values()]))
    assert not set(spec['pilot_trials']) & set(spec['assessment_trials'])
    with parallel_config(backend='loky',inner_max_num_threads=1):
        assessment=Parallel(n_jobs=jobs,verbose=10)(delayed(run_trial)(loc,t,spec['theory_draws'],root,weights)
            for loc in assessment_locations for t in spec['assessment_trials'])
    assert not {r['data_hash'] for r in pilots} & {r['data_hash'] for r in assessment}
    payload=dict(specification=spec,frozen=frozen,records=assessment)
    base.atomic_pickle(root/'assessment.pkl',payload)
    table=summarize(assessment,theory=True)
    table.to_csv(root/'assessment_summary.csv',index=False)
    print(table[table.L>=4].to_string(index=False),flush=True)
    return payload


if __name__=='__main__':
    root,spec,weights=prepare()
    print('Results:',root,flush=True)
    run(root,spec,weights)
