"""Matched-intercept early-stopping sweep, with explicit model/provenance."""
from tools.myocardial_exploration import *
from tools.myocardial_finite_iteration_exploration import reference_distribution
from tools.myocardial_intercept_exploration import offset_rho_path
from WERM.myocardial_intercept import (fit_with_intercept,squared_intercept_state,
    logistic_intercept_state,intercept_loglikelihood)
from WERM.myocardial_finite_intercept import finite_intercept_fit

METHODS=('fista','lbfgs','gd')
FIT_STEPS=(1,3,10,30,100)
MIN_SEGMENTS=(1,2,5)
RHO_STEPS=(1,3,10,20,30)


def experiment_root():
    files=['tools/myocardial_finite_intercept_sweep.py','WERM/myocardial_finite_intercept.py',
           'WERM/myocardial_intercept.py','WERM/signal_strength.py','WERM/myocardial_numerics.py',
           'WERM/proximal.py','results/myocardial_raw_uci579.csv']
    specification=dict(methods=METHODS,fit_steps=FIT_STEPS,minimum_segments=MIN_SEGMENTS,rho_steps=RHO_STEPS,
        intercept=True,penalty='none',count_prior=[.5,.5],intercept_column='1/sqrt(n)',
        coefficient_start='zero slopes, weighted intercept-only logistic MLE',
        response_model='prevalence-calibrated intercept model; finite-fit state equations are heuristic',
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    digest=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_finite_intercept_'+digest);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(specification,indent=2));return root


def setting_for(delta,y,locations):
    eta,prior,phi=configuration_prior(len(y),locations[(locations>=delta)&(locations<=len(y)-delta)])
    return dict(delta_min=delta,eta=eta,prior=prior,phi=phi,rate=float(y.mean()))


def fit_task(delta,name,X,y,setting,root):
    path=root/f'fit_d{delta}_{name}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    try:
        if name=='least_squares':
            fit=fit_with_intercept(X,y,setting['phi'],'none',0.,loss='squared')
            fit.update(method='least_squares',budget=0,fit_intercept=True)
        else:
            method,budget=name.split('_');fit=finite_intercept_fit(X,y,setting['phi'],method,int(budget))
        fit['adjusted_score_mean']=fit['V'].mean(axis=0)
        result=dict(name=name,delta_min=delta,fit=fit)
    except Exception as error:result=dict(name=name,delta_min=delta,error=repr(error),traceback=traceback.format_exc())
    atomic_save(path,result);return result


def rho_task(fit_rec,setting,X,y,initial_scale,root):
    name=fit_rec['name'];delta=setting['delta_min'];prefix=f'd{delta}_{name}_start{initial_scale:g}'
    path=root/('rho_'+prefix+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    try:
        fit=fit_rec['fit'];loss='squared' if name=='least_squares' else 'logistic'
        e,psi=pilot(fit['theta'],y,setting['eta'],loss);S=fit['V'].T@fit['V']/len(y)
        rho=initial_scale*np.eye(2);records=[]
        updates=RHO_STEPS if initial_scale==1 else (3,10,30)
        def state_at(matrix,power=7,budget=40):
            return squared_intercept_state(matrix,setting['phi'],psi,len(y)/X.shape[1],setting['rate']) if loss=='squared' else logistic_intercept_state(
                matrix,setting['phi'],psi,len(y)/X.shape[1],setting['rate'],sample_power=power,max_nfev=budget)
        for it in range(1,max(updates)+1):
            state=state_at(rho);inv=np.linalg.pinv(state['lambda'],rcond=1e-6)
            rho=project_covariance(.75*rho+.25*project_covariance(inv.T@(S-state['kappa'])@inv))
            if it in updates:
                check=state_at(rho,8,60);residual=float(np.linalg.norm(check['lambda'].T@rho@check['lambda']+check['kappa']-S))
                records.append(dict(label=prefix+f'_rho{it}',rho=rho.copy(),rho_updates=it,
                    auxiliary=name,initial_scale=initial_scale,pilot=e,loss=loss,delta_min=delta,
                    moment_residual=residual,relative_moment_residual=residual/max(np.linalg.norm(S),1e-15),
                    auxiliary_adjusted_mean_norm=float(np.linalg.norm(fit['adjusted_score_mean']))))
        if loss=='squared' and initial_scale==1:
            direct=offset_rho_path(S,setting['phi'],psi,len(y)/X.shape[1],setting['rate'],'squared')[-1]
            records.append(dict(label=prefix+'_rho_direct',rho=direct['rho'],rho_updates=-1,
                auxiliary=name,initial_scale=1,pilot=e,loss=loss,delta_min=delta,
                moment_residual=direct['moment_residual'],relative_moment_residual=direct['moment_residual']/max(np.linalg.norm(S),1e-15),
                auxiliary_adjusted_mean_norm=float(np.linalg.norm(fit['adjusted_score_mean']))))
        result=dict(records=records)
    except Exception as error:result=dict(error=repr(error),case=prefix,traceback=traceback.format_exc())
    atomic_save(path,result);print('Rho trajectory finished:',prefix,flush=True);return result


def state_task(rec,setting,n,p,root,power,seed,budget):
    tag=f'q{power}_seed{seed}_b{budget}';path=root/(rec['label']+'_'+tag+'_states.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    partial=path.with_name(path.stem+'_partial.pkl')
    states=pickle.load(partial.open('rb')) if partial.exists() else []
    try:
        for e in setting['eta'][len(states):]:
            psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            states.append(logistic_intercept_state(rec['rho'],setting['phi'],psi,n/p,setting['rate'],
                sample_power=power,max_nfev=budget,seed=seed))
            if len(states)%8==0:atomic_save(partial,states)
        result=dict(states=states)
    except Exception as error:result=dict(error=repr(error),case=rec['label'],tag=tag,traceback=traceback.format_exc())
    atomic_save(path,result);print('Posterior states:',rec['label'],tag,flush=True);return result


def score_task(rec,bundle,setting,fit_rec,X,y,ages,reference,root,tag):
    name=fit_rec['name'];label=rec['label']+'__'+name+'__'+tag;path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    try:
        fit=fit_rec['fit'];ll=[]
        for e,state in zip(setting['eta'],bundle['states']):
            psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
            ll.append(intercept_loglikelihood(fit['V'],y,rec['rho'],state,psi))
        ll=np.asarray(ll);joint=ll+np.log(setting['prior']);probs=np.exp(joint-logsumexp(joint))
        curve=np.exp(ll[1:]-logsumexp(ll[1:]));a=ages[setting['eta'][1:]];mapping=dict(zip(a,curve))
        tv=.5*sum(abs(reference[k]-mapping.get(k,0.)) for k in reference)
        assert np.isfinite(ll).all() and np.isfinite(curve).all()
        result=dict(label=label,case=rec['label'],posterior_fit=name,auxiliary=rec['auxiliary'],
            group='LS_auxiliary' if rec['auxiliary']=='least_squares' else 'matched_logistic',
            method=fit['method'],fit_budget=fit['budget'],intercept=True,explicit_penalty=0.,
            Delta=rec['delta_min'],rho_updates=rec['rho_updates'],initial_scale=rec['initial_scale'],
            pilot_index=rec['pilot'],pilot_age=float(ages[rec['pilot']]) if rec['pilot']>=0 else np.nan,
            tag=tag,P_1=float(probs[1:].sum()),conditional=curve,ages=a,
            mass_above55=float(curve[a>55].sum()),peak_age=float(a[np.argmax(curve)]),
            TV_to_reference=tv,max_location_probability=float(curve.max()),
            moment_residual=rec['moment_residual'],relative_moment_residual=rec['relative_moment_residual'],
            posterior_adjusted_mean_norm=float(np.linalg.norm(fit['adjusted_score_mean'])),
            auxiliary_adjusted_mean_norm=rec['auxiliary_adjusted_mean_norm'],
            intercept_score_max=max(abs(c['intercept_score']) for c in fit['checks']),
            fit_kkt_max=max(c['kkt'] for c in fit['checks']),
            actual_iterations_max=max(c['iterations'] for c in fit['checks']),
            FP_residual_max=max(s['training_residual'] for s in bundle['states']))
    except Exception as error:result=dict(error=repr(error),case=rec['label'],posterior_fit=name,tag=tag,traceback=traceback.format_exc())
    atomic_save(path,result);return result


def scalar_table(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k not in ('conditional','ages')} for r in rows if 'error' not in r])


def run_sweep(smoke=False):
    X,y,ages,locations=load_data();reference=reference_distribution();root=experiment_root()
    print('Finite-intercept experiment directory:',root,flush=True)
    settings={d:setting_for(d,y,locations) for d in MIN_SEGMENTS}
    names=[f'{method}_{budget}' for method in METHODS for budget in FIT_STEPS]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        fitted=Parallel(n_jobs=3 if smoke else 36)(delayed(fit_task)(d,name,X,y,settings[d],root)
                    for d in ((1,) if smoke else MIN_SEGMENTS) for name in names+['least_squares'])
    errors=[r for r in fitted if 'error' in r];fits={(r['delta_min'],r['name']):r for r in fitted if 'error' not in r}
    if smoke:
        assert not errors,errors
        for r in fitted:
            fit=r['fit'];assert np.isfinite(fit['V']).all()
            if r['name']!='least_squares':assert max(c['iterations'] for c in fit['checks'])<=fit['budget']
        e=settings[1]['eta'][len(settings[1]['eta'])//2];psi=(np.arange(len(y))>=e).astype(int)
        state=logistic_intercept_state(np.eye(2),settings[1]['phi'],psi,len(y)/X.shape[1],settings[1]['rate'],sample_power=7,max_nfev=40)
        ll=intercept_loglikelihood(fits[(1,'fista_10')]['fit']['V'],y,np.eye(2),state,psi)
        assert np.isfinite(ll);print('Real-data smoke passed: all 15 logistic fits, LS, and intercept posterior.',flush=True);return root
    rho_jobs=[(r,1.) for r in fits.values()]
    rho_jobs += [(fits[(1,f'{method}_{budget}')],scale) for method in METHODS for budget in (10,30)
                 for scale in (.1,10.) if (1,f'{method}_{budget}') in fits]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        paths=Parallel(n_jobs=36)(delayed(rho_task)(r,settings[r['delta_min']],X,y,scale,root) for r,scale in rho_jobs)
        records=[r for path in paths if 'error' not in path for r in path['records']]
        errors.extend(p for p in paths if 'error' in p)
        bundles=Parallel(n_jobs=36)(delayed(state_task)(r,settings[r['delta_min']],len(y),X.shape[1],root,8,42,60) for r in records)
        errors.extend(b for b in bundles if 'error' in b)
        jobs=[(r,bundle,fits[(r['delta_min'],name)]) for r,bundle in zip(records,bundles) if 'error' not in bundle
              for name in (names if r['auxiliary']=='least_squares' else [r['auxiliary']]) if (r['delta_min'],name) in fits]
        scores=Parallel(n_jobs=36)(delayed(score_task)(r,b,settings[r['delta_min']],fit,X,y,ages,reference,root,'primary') for r,b,fit in jobs)
    errors.extend(r for r in scores if 'error' in r);summary=scalar_table(scores)
    summary.to_csv(root/'summary.csv',index=False)
    selected_parts=[summary.sort_values('TV_to_reference').head(10)]
    for group,data in summary.groupby('group'):
        selected_parts.extend([data.sort_values('TV_to_reference').head(10),
            data[data.TV_to_reference<=data.TV_to_reference.min()+.1].sort_values('P_1',ascending=False).head(2)])
    selected=pd.concat(selected_parts).drop_duplicates('label');selected.to_csv(root/'selected_for_validation.csv',index=False)
    by_case={r['label']:r for r in records}
    validation_jobs=[(by_case[c],power,seed) for c in selected.case.unique() for power,seed in ((9,42),(10,123))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        refined=Parallel(n_jobs=36)(delayed(state_task)(r,settings[r['delta_min']],len(y),X.shape[1],root,power,seed,100)
                                  for r,power,seed in validation_jobs)
        checks=Parallel(n_jobs=36)(delayed(score_task)(r,b,settings[r['delta_min']],fits[(r['delta_min'],row.posterior_fit)],
            X,y,ages,reference,root,f'q{power}_seed{seed}') for (r,power,seed),b in zip(validation_jobs,refined) if 'error' not in b
            for row in selected[selected.case==r['label']].itertuples())
    errors.extend(b for b in refined if 'error' in b);errors.extend(c for c in checks if 'error' in c)
    validation=scalar_table(checks);validation.to_csv(root/'validation.csv',index=False)
    rows=[]
    for (case,posterior),data in validation.groupby(['case','posterior_fit']):
        if len(data)!=2:continue
        original=data.iloc[0].to_dict();curves=[pickle.load((root/(r.label+'.pkl')).open('rb'))['conditional'] for r in data.itertuples()]
        original.update(P1_min=data.P_1.min(),P1_max=data.P_1.max(),mean_TV=data.TV_to_reference.mean(),max_TV=data.TV_to_reference.max(),
            numerical_curve_TV=float(.5*np.abs(curves[0]-curves[1]).sum()),mass_above55_min=data.mass_above55.min(),
            peak_min=data.peak_age.min(),peak_max=data.peak_age.max())
        rows.append(original)
    ranked=pd.DataFrame(rows).sort_values(['mean_TV','max_TV']);ranked.to_csv(root/'ranked.csv',index=False)
    ranked.head(10).to_csv(root/'top_ten.csv',index=False)
    ranked[ranked.group=='matched_logistic'].head(10).to_csv(root/'top_ten_matched_logistic.csv',index=False)
    (root/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    print('FINITE INTERCEPT SWEEP COMPLETE:',len(summary),'primary comparisons;',len(validation),'checks;',len(errors),'recorded failures.',flush=True)
    print(ranked[['group','method','fit_budget','Delta','rho_updates','initial_scale','P1_min','P1_max','mean_TV']].head(10).to_string(index=False),flush=True)
    return root


if __name__=='__main__':
    import sys
    run_sweep(smoke='--smoke' in sys.argv)
