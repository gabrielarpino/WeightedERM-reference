"""Reproducible exploratory grid, also embedded verbatim in its notebook."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import pickle, hashlib, json, time, traceback
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from joblib import Parallel, delayed, parallel_config
from WERM.signal_strength import (configuration_prior, empirical_adjustment,
    squared_logistic_state, gaussian_logistic_loglikelihood)
from WERM.myocardial_approximate import approximate_rho, logistic_state
from WERM.proximal import fit_weighted

DELTAS=(1,85,169)
FIT_SETTINGS=(('l2',1e-8),('l2',1e-6),('l1',1e-6),('l1',1e-4))
ITERATIONS=(1,3,10,30)
N_JOBS=36


def load_data():
    from scipy.linalg import eigh
    raw=pd.read_csv('results/myocardial_raw_uci579.csv')
    data=raw.loc[raw.AGE.notna()].copy()
    data=data.iloc[np.argsort(data.AGE.to_numpy())]
    ages=data.AGE.to_numpy(float);y=(data.ZSN.to_numpy()>0).astype(int)
    X=data[list(raw.columns[2:112])].to_numpy(float)
    X=np.where(np.isnan(X),np.nanmean(X,axis=0),X);X-=X.mean(axis=0)
    n,p=X.shape
    values,vectors=eigh(X.T@X/n+1e-6*np.eye(p))
    X=X@((vectors/np.sqrt(values))@vectors.T)/np.sqrt(n)
    _,indices=np.unique(ages,return_index=True)
    return X,y,ages,indices[indices>0]


def atomic_save(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:pickle.dump(value,f)
    temp.replace(path)


def fit_logistic(X,y,phi,penalty,strength):
    columns=[];checks=[]
    for w in phi:
        wn=w/w.sum()
        if penalty=='l2':
            def objective(beta):
                z=X@beta
                return float(wn@(np.logaddexp(0,z)-y*z)+strength*(beta@beta)), X.T@(wn*(expit(z)-y))+2*strength*beta
            fit=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',
                         options=dict(maxiter=5000,ftol=1e-14,gtol=1e-10,maxcor=30))
            beta=fit.x;err=np.max(abs(objective(beta)[1]))
            checks.append(dict(kkt=float(err),iterations=fit.nit,success=bool(fit.success)))
        else:
            fit=fit_weighted(X,y,loss='logistic',sample_weight=w,penalty='l1',
                             penalty_strength=strength,max_iter=20000,tol=1e-8)
            beta=fit.coef
            checks.append(dict(kkt=fit.kkt_violation,iterations=fit.n_iter,success=fit.converged))
        columns.append(beta)
    theta=X@np.column_stack(columns)
    V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
    return dict(theta=theta,V=V,b=b,checks=checks,penalty=penalty,strength=strength)


def pilot(theta,y,eta,loss):
    errors=(theta-y[:,None])**2 if loss=='squared' else np.logaddexp(0,theta)-y[:,None]*theta
    scores=[errors[:,0].sum() if e<0 else errors[:e,0].sum()+errors[e:,1].sum() for e in eta]
    e=int(eta[np.argmin(scores)])
    # Signal strength with two latent signals is not identified at a null pilot.
    return e, np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)


def project_covariance(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.clip(values,1e-4,100.))@vectors.T


def rho_path(S,phi,psi,delta,loss):
    rho=np.eye(2);records=[]
    for iteration in range(1,max(ITERATIONS)+1):
        if loss=='squared':state=squared_logistic_state(rho,phi,psi,delta)
        else:state=logistic_state(rho,phi,psi,delta,sample_power=7,max_nfev=30)
        lam=state['lambda'];K=state['kappa'];inverse=np.linalg.pinv(lam,rcond=1e-6)
        target=project_covariance(inverse.T@(S-K)@inverse)
        rho=project_covariance(.75*rho+.25*target)
        if iteration in ITERATIONS:
            check=squared_logistic_state(rho,phi,psi,delta) if loss=='squared' else logistic_state(rho,phi,psi,delta,sample_power=8,max_nfev=60)
            records.append(dict(rho=rho.copy(),iteration=iteration,
                moment_residual=float(np.linalg.norm(check['lambda'].T@rho@check['lambda']+check['kappa']-S)),
                state_residual=check.get('training_residual',0.)))
    return records


def prepare(delta_min,root,X,y,ages,locations):
    path=root/f'fits_delta{delta_min}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    eta,prior,phi=configuration_prior(len(y),locations[(locations>=delta_min)&(locations<=len(y)-delta_min)])
    fits={f'{kind}_{strength:g}':fit_logistic(X,y,phi,kind,strength) for kind,strength in FIT_SETTINGS}
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    theta=X@B;V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='squared')
    auxiliary={'least_squares':dict(theta=theta,V=V,b=b),**fits}
    scenarios=[]
    for name,fit in auxiliary.items():
        loss='squared' if name=='least_squares' else 'logistic'
        e,psi=pilot(fit['theta'],y,eta,loss);S=fit['V'].T@fit['V']/len(y)
        for record in rho_path(S,phi,psi,len(y)/X.shape[1],loss):
            scenarios.append(dict(record,auxiliary=name,pilot=e,loss=loss))
        if loss=='squared':
            record=approximate_rho(S,phi,psi,len(y)/X.shape[1])
            scenarios.append(dict(rho=record['rho'],iteration=-1,moment_residual=record['residual_norm'],
                                  state_residual=0.,auxiliary=name,pilot=e,loss=loss))
    result=dict(eta=eta,prior=prior,phi=phi,fits=fits,scenarios=scenarios,delta_min=delta_min)
    atomic_save(path,result);return result


def evaluate_scenario(setting,scenario,X,y,ages,root,power=7,budget=40):
    label=f"d{setting['delta_min']}_{scenario['auxiliary']}_it{scenario['iteration']}_q{power}_b{budget}"
    path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    begin=time.monotonic();rho=scenario['rho'];eta=setting['eta'];phi=setting['phi']
    logweights={name:[] for name in setting['fits']};residuals=[];states=[]
    for e in eta:
        psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
        state=logistic_state(rho,phi,psi,len(y)/X.shape[1],sample_power=power,max_nfev=budget)
        residuals.append(state['training_residual']);states.append(state)
        for name,fit in setting['fits'].items():
            logweights[name].append(gaussian_logistic_loglikelihood(fit['V'],y,rho,
                state['lambda'],state['kappa'],psi,quadrature_order=80))
    results={}
    for name,values in logweights.items():
        values=np.asarray(values);joint=values+np.log(setting['prior'])
        probs=np.exp(joint-logsumexp(joint));conditional=np.exp(values[1:]-logsumexp(values[1:]))
        results[name]=dict(log_likelihood=values,probabilities=probs,conditional=conditional,
                          presence=float(probs[1:].sum()),peak_age=float(ages[eta[1:][np.argmax(conditional)]]))
    result=dict(label=label,scenario=scenario,delta_min=setting['delta_min'],eta=eta,
                results=results,residuals=residuals,states=states,elapsed=time.monotonic()-begin)
    atomic_save(path,result);print('Completed',label,flush=True);return result


def run(smoke=False):
    X,y,ages,locations=load_data()
    specification=dict(deltas=DELTAS,fits=FIT_SETTINGS,iterations=ITERATIONS,
        damping=.25,eigenvalue_bounds=[1e-4,100],seed=42,primary_prior='half single null, half uniform admissible age grid',
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
               ['tools/myocardial_exploration.py','WERM/myocardial_approximate.py','WERM/signal_strength.py','results/myocardial_raw_uci579.csv']})
    fingerprint=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_exploration_'+fingerprint);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(specification,indent=2))
    print('Checkpoint directory:',root,flush=True)
    if smoke:
        eta,prior,phi=configuration_prior(len(y),locations)
        fit=fit_logistic(X,y,phi,'l2',1e-6)
        state=logistic_state(np.eye(2),phi,(np.arange(len(y))>=locations[30]).astype(int),len(y)/X.shape[1],sample_power=7,max_nfev=40)
        print('Smoke:',fit['checks'],state['training_residual'],flush=True)
        return root
    with parallel_config(backend='loky',inner_max_num_threads=1):
        settings=Parallel(n_jobs=3)(delayed(prepare)(d,root,X,y,ages,locations) for d in DELTAS)
        print('All auxiliary trajectories computed.',flush=True)
        results=Parallel(n_jobs=N_JOBS)(delayed(evaluate_scenario)(s,r,X,y,ages,root) for s in settings for r in s['scenarios'])
    rows=[]
    for result in results:
        for fit,stats in result['results'].items():
            rows.append(dict(label=result['label'],delta_min=result['delta_min'],auxiliary=result['scenario']['auxiliary'],
                rho_iterations=result['scenario']['iteration'],posterior_fit=fit,presence=stats['presence'],peak_age=stats['peak_age'],
                moment_residual=result['scenario']['moment_residual'],fp_residual_max=max(result['residuals'])))
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    print('Full coarse exploration complete:',len(rows),'comparisons.',flush=True)
    return root


if __name__=='__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
