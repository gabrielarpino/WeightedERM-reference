"""Reproducible empirical WERM-CV versus composed first-order SE experiment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.special import gammaln, betaln
from joblib import Parallel, delayed, parallel_config
from WERM.signal_priors import SparseDiffSignal
from WERM.experiment_utils import trial_rng, data_hash, evaluate_changepoints
from tools import el_state_evolution as original_el
from tools import werm_cv_empirical as empirical
from tools import werm_cv_theory as theory

SETTINGS=[dict(experiment='M2',n=n,p=200,max_segments=4,delta_fraction=1/40,
               noise_variance=.1,locations=[.2,.5]) for n in (1800,2200,2600)] + [
    dict(experiment='EL',n=2400,p=200,max_segments=6,delta_fraction=1/10,
         noise_variance=.25,locations=[.2,.5,.8])]
for setting in SETTINGS:
    setting['delta']=setting['n']/setting['p']
    setting['minimum_segment_fraction']=240/(.9*setting['n'])


def uniform_marginals(n,k,d):
    """Exact marginals for uniform integer segment lengths >=d, including ends.

    After removing d from each length, the remaining weak composition is
    uniform. Its first j coordinates sum to BetaBinomial(n-k*d,j,k-j).
    """
    if k<1 or d<1 or k*d>n:
        raise ValueError('Infeasible minimum segment length')
    if k==1:
        return np.ones((1,n))
    extra=n-k*d;z=np.arange(extra+1,dtype=float);i=np.arange(n)
    cumulative=[np.ones(n)]
    for j in range(1,k):
        logpmf=(gammaln(extra+1)-gammaln(z+1)-gammaln(extra-z+1)
                +betaln(z+j,extra-z+k-j)-betaln(j,k-j))
        pmf=np.exp(logpmf);pmf/=pmf.sum()
        cdf=np.cumsum(pmf);cdf[-1]=1.
        at=i-j*d
        cumulative.append(np.where(at<0,0.,cdf[np.clip(at,0,extra)]))
    cumulative.append(np.zeros(n))
    phi=np.maximum(np.asarray(cumulative[:-1])-np.asarray(cumulative[1:]),0.)
    phi/=phi.sum(axis=0)
    return phi


def prepare(trials=100,theory_draws=20):
    sources=['tools/werm_cv_theory_experiment.py','tools/werm_cv_theory.py',
        'tools/werm_cv_empirical.py','tools/werm_cv_partition.py','tools/el_state_evolution.py',
        'WERM/proximal.py','WERM/signal_strength.py','WERM/signal_priors.py','WERM/experiment_utils.py']
    spec=dict(schema=1,settings=SETTINGS,trials=trials,theory_draws=theory_draws,folds=10,
        min_training_segment_rows=240,coefficient_penalty='none',partition_penalty=0,
        scorer='absolute error',fitter='squared error',
        prior='exact uniform integer compositions with every segment >= minimum',
        theory='deterministic expected-loss partition; uniform-refit SE; absolute risk; CV argmin; full SE',
        sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results/E-WERM-CV-theory')/digest;root.mkdir(parents=True,exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    path=root/'weights.pkl'
    if path.exists():
        weights=pickle.loads(path.read_bytes())
    else:
        weights={}
        for s in SETTINGS:
            for n in (s['n'],s['n']*9//10):
                gap=max(empirical.minimum_gap(n,s['delta_fraction']),int(np.ceil(s['minimum_segment_fraction']*n)))
                for k in range(1,s['max_segments']+1):
                    weights[n,k,gap]=uniform_marginals(n,k,gap)
        original_el.atomic_pickle(path,weights)
    return root,spec,weights


def generate_data(setting,trial):
    n,p=setting['n'],setting['p']
    if setting['experiment']=='EL':
        X,y,beta,truth,labels,_=original_el.generate_data(trial,n,p)
        return X,y,beta,truth,labels
    rng=trial_rng('E-WERM-CV-theory-M2',n,trial)
    beta=SparseDiffSignal(8,np.sqrt(400*n/p),.3,4).sample(p,rng=rng)
    X=rng.normal(0,1/np.sqrt(n),(n,p))
    noise=rng.normal(size=n)*np.sqrt(.1)
    truth=(n*np.array(setting['locations'])).astype(int)
    labels=np.searchsorted(truth,np.arange(n),side='right')
    theta=(X*np.sqrt(n))@(beta/np.sqrt(n))
    return X,theta[np.arange(n),labels]+noise,beta[:,:3],truth,labels


def metrics(cp,truth,n):
    result=evaluate_changepoints(cp,truth,n)
    result['count']=len(cp);result['cp']=cp
    return result


def run_trial(index,trial,root,spec,weights):
    path=root/f'setting{index}_trial{trial:03d}.pkl'
    if path.exists():return pickle.loads(path.read_bytes())
    started=time.perf_counter();s=spec['settings'][index]
    X,y,beta,truth,labels=generate_data(s,trial)
    gamma=beta.T@beta/s['n']
    options=dict(max_segments=s['max_segments'],weights=weights,
        delta_fraction=s['delta_fraction'],minimum_segment_fraction=s['minimum_segment_fraction'])
    fitted=empirical.run_cv(X,y,**options,return_coefficients=True,
                           loss_scale=2. if s['experiment']=='M2' else 1.)
    conditional=empirical.gaussian_validation_risk(fitted,beta,labels,s['noise_variance'])
    predicted=theory.predict_cv(gamma,labels,s['p'],s['noise_variance'],**options)
    # A diagnostic using empirical intervals, explicitly excluded from SE selection.
    refit_oracle=np.zeros_like(fitted['fold_scores']);indices=np.arange(s['n'])
    for (k,v),f in fitted['fold_results'].items():
        train=indices[indices%10!=v];test=indices[v::10]
        refit_oracle[k-1,v]=theory.uniform_refit_risk(gamma,labels,train,test,
            f['train_cp'],s['p'],s['noise_variance'])['score']
        for key in ('segment_coefficients','predictions','absolute_errors'):
            f.pop(key,None)
    for f in fitted['full_fits'].values():f.pop('coefficients',None)
    for L,f in fitted['selected_by_bound'].items():f.update(metrics(f['cp'],truth,s['n']))
    cached={}
    for L,f in predicted['selected_by_bound'].items():
        k=f['selected_segments']
        if k not in cached:
            rng=np.random.default_rng(np.random.SeedSequence([20261107,index,trial,k,491]))
            cached[k]=theory.final_surrogates(gamma,labels,s['p'],s['noise_variance'],k,
                weights,s['delta_fraction'],spec['theory_draws'],rng,s['minimum_segment_fraction'])
            for fit in cached[k]['final_fits']:fit.update(metrics(fit['cp'],truth,s['n']))
        f.update(cached[k])
    diagnostics=dict(
        min_empirical_segment_ratio=min(f['segment_sample_ratios'].min() for f in fitted['fold_results'].values()),
        min_theory_segment_ratio=min(f['segment_ratios'].min() for f in predicted['fold_results'].values()),
        max_se_residual=max([f['se_residual'] for f in predicted['fold_results'].values()]+
                            [f['se_residual'] for f in cached.values()]),
        max_refit_kkt=max(f['max_refit_kkt'] for f in fitted['fold_results'].values()),
        max_partition_kkt=max(f['max_partition_kkt'] for f in fitted['fold_results'].values()),
        cv_score_mae=float(np.mean(abs(fitted['normalized_cv_scores']-predicted['normalized_cv_scores']))))
    assert diagnostics['min_empirical_segment_ratio']>1
    assert diagnostics['min_theory_segment_ratio']>1
    assert diagnostics['max_se_residual']<1e-9
    record=dict(**s,setting=index,trial=trial,truth=truth,gamma=gamma,data_hash=data_hash(X,y),
        empirical=fitted,theory=predicted,diagnostics=diagnostics,
        conditional_validation_risk=conditional,
        refit_se_at_empirical_partitions=refit_oracle.sum(axis=1)/s['n'],
        seconds=time.perf_counter()-started)
    original_el.atomic_pickle(path,record)
    return record


def run(root,spec,weights,jobs=12):
    with parallel_config(backend='loky',inner_max_num_threads=1):
        records=Parallel(n_jobs=jobs,verbose=10)(delayed(run_trial)(i,t,root,spec,weights)
            for i in range(len(spec['settings'])) for t in range(spec['trials']))
    payload=dict(specification=spec,records=records)
    original_el.atomic_pickle(root/'results.pkl',payload)
    return payload


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--jobs',type=int,default=12);parser.add_argument('--theory-draws',type=int,default=20)
    args=parser.parse_args();root,spec,weights=prepare(args.trials,args.theory_draws)
    print('Results:',root,flush=True)
    result=run(root,spec,weights,args.jobs)
    print('Completed',len(result['records']),'paired trials.',flush=True)
