"""Fixed-point hyperparameter search: no external covariance mixtures."""
from tools.myocardial_final_redo import *
from tools.myocardial_direct_moment_experiment import ls_fit,state_at
from WERM.myocardial_finite_intercept import finite_intercept_fit
from WERM.myocardial_no_intercept_solvers import fit_column
from tools.myocardial_best_ones import curve_metrics

DELTAS_SEARCH=(1,2,5,10,20)
STEPS=(1,3,5,8,12,16,20,24,30,40)
FITS=[('fista',b) for b in (8,12,16,20,30,50)]+[('lbfgs',b) for b in (3,5,10,20)]+[('gd',b) for b in (8,16,30,50)]
BASE=Path('results/myocardial_final_redo_e3d965cc4b9b')


def correlation(rho):
    return float(rho[0,1]/np.sqrt(rho[0,0]*rho[1,1]))


def update_rho(rho,S,state):
    inverse=np.linalg.pinv(state['lambda'],rcond=1e-6)
    raw=inverse.T@(S-state['kappa'])@inverse
    eigenvalues=np.linalg.eigvalsh((raw+raw.T)/2)
    result=project_covariance(.75*rho+.25*project_covariance(raw))
    return result,eigenvalues,int(np.sum(eigenvalues<1e-4))


def prepare(delta,intercept,X,y,locations,root):
    name=f'd{delta}_int{int(intercept)}';path=root/f'fits_{name}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    eta,prior,phi=configuration_prior(len(y),locations[(locations>=delta)&(locations<=len(y)-delta)])
    fits={'LS':ls_fit(X,y,phi,intercept)}
    for method,budget in FITS:
        if intercept:fit=finite_intercept_fit(X,y,phi,method,budget)
        else:
            cols=[fit_column(X,y,w,method,budget) for w in phi]
            B=np.column_stack([c['coef'] for c in cols]);theta=X@B
            V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
            fit=dict(theta=theta,V=V,b=b,checks=cols)
        fits[f'{method}{budget}']=fit
    result=dict(name=name,Delta=delta,intercept=intercept,setting=dict(eta=eta,prior=prior,phi=phi),
        fits=fits,delta=len(y)/X.shape[1],rate=float(y.mean()),spec=dict(intercept=intercept))
    atomic_save(path,result);return result


def rho_path_search(setting,auxiliary,y,root,power=9,seed=123,max_step=40):
    label=f"{setting['name']}_{auxiliary}_q{power}_seed{seed}_steps{max_step}"
    path=root/f'rho_{label}.pkl';partial=root/f'rho_{label}_partial.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    try:
        fit=setting['fits'][auxiliary];loss='squared' if auxiliary=='LS' else 'logistic'
        pilot_eta,psi=pilot(fit['theta'],y,setting['setting']['eta'],loss)
        S=fit['V'].T@fit['V']/len(y);prepared=dict(setting,loss=loss)
        saved=pickle.load(partial.open('rb')) if partial.exists() else dict(rho=np.eye(2),iteration=0,records=[],trace=[])
        rho=saved['rho'];records=saved['records'];trace=saved['trace']
        for it in range(saved['iteration']+1,max_step+1):
            state=state_at(prepared,rho,psi,power=power,seed=seed)
            rho,raw_eigs,clipped=update_rho(rho,S,state)
            trace.append(dict(iteration=it,raw_eigenvalues=raw_eigs,clipped=clipped,correlation=correlation(rho)))
            if it in STEPS:
                check=state_at(prepared,rho,psi,power=power,seed=seed)
                moment=check.get('moment')
                if moment is None:moment=check['lambda'].T@rho@check['lambda']+check['kappa']
                residual=float(np.linalg.norm(moment-S,'fro'))
                records.append(dict(case=f"{setting['name']}_{auxiliary}_rho{it}",auxiliary=auxiliary,
                    rho_updates=it,rho=rho.copy(),correlation=correlation(rho),pilot=pilot_eta,
                    moment_residual=residual,relative_residual=residual/np.linalg.norm(S,'fro'),
                    clipped_updates=sum(t['clipped']>0 for t in trace),both_clipped_updates=sum(t['clipped']==2 for t in trace),
                    inner_residual=check.get('training_residual',0.),rho_power=power,rho_seed=seed))
            atomic_save(partial,dict(rho=rho,iteration=it,records=records,trace=trace))
        result=dict(records=records,trace=trace,setting_name=setting['name'],auxiliary=auxiliary)
    except Exception:result=dict(error=traceback.format_exc(),label=label)
    atomic_save(path,result);print('Fixed-point path completed:',label,flush=True);return result


def posterior(setting,record,names,y,ages,target,root,power=9,seed=123):
    label=f"{record['case']}_rhop{record['rho_power']}s{record['rho_seed']}_postp{power}s{seed}"
    path=root/f'post_{label}.pkl';partial=root/f'post_{label}_partial.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    try:
        eta=setting['setting']['eta'];n=len(y)
        states=pickle.load(partial.open('rb')) if partial.exists() else []
        prepared=dict(setting,loss='logistic')
        for e in eta[len(states):]:
            psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            states.append(state_at(prepared,record['rho'],psi,power=power,seed=seed))
            if len(states)%8==0:atomic_save(partial,states)
        atomic_save(partial,states);results=[]
        from WERM.myocardial_intercept import intercept_loglikelihood
        for name in names:
            fit=setting['fits'][name];values=[]
            for e,state in zip(eta,states):
                psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
                if setting['intercept']:ll=intercept_loglikelihood(fit['V'],y,record['rho'],state,psi,order=160)
                else:ll=gaussian_logistic_loglikelihood(fit['V'],y,record['rho'],state['lambda'],state['kappa'],psi,quadrature_order=160)
                values.append(ll)
            result=normalize_posterior(values,setting['setting']['prior']);a=ages[eta[1:]]
            metrics=curve_metrics(a,result['conditional'],target)
            result.update(record,id=record['case']+'_'+name,posterior=name,intercept=setting['intercept'],
                Delta=setting['Delta'],setting_name=setting['name'],ages=a,TV_to_rank3=metrics['TV'],
                P_1=result['presence'],peak_age=metrics['peak'],mass_above55=metrics['above55'],
                matched=(name==record['auxiliary']),posterior_power=power,posterior_seed=seed,
                max_state_residual=max(s['training_residual'] for s in states),checkpoint=str(path))
            results.append(result)
        out=dict(results=results)
    except Exception:out=dict(error=traceback.format_exc(),label=label)
    atomic_save(path,out);return out


def tabulate(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k not in ('rho','conditional','ages','probabilities','log_likelihood')} for r in rows])


def run(smoke=False):
    X,y,ages,locations=load_data()
    baseline=pickle.load((BASE/'scenario_100_100_q10_seed123.pkl').open('rb'))
    target=dict(zip(baseline['ages'],baseline['conditional']))
    spec=dict(deltas=DELTAS_SEARCH,fit_grid=FITS,rho_steps=STEPS,intercepts=[False,True],
        rho_start='identity',damping=.25,eigenvalue_bounds=[1e-4,100],search_draws=512,
        final_draws=1024,posterior_auxiliary_policy='LS: all logistic fits; logistic: matched plus FISTA16',
        screening='all snapshots with abs correlation >=0.001 plus exact rank3 control',
        fingerprint=hashlib.sha256(Path('tools/myocardial_fixedpoint_search.py').read_bytes()).hexdigest())
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_fixedpoint_rank3_'+digest);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    if smoke:
        setting=prepare(2,False,X,y,locations,root)
        path=rho_path_search(setting,'LS',y,root,max_step=20)
        # Store partial smoke trajectory separately, not as a complete 40-step path.
        assert 'error' not in path,path
        rec=next(r for r in path['records'] if r['rho_updates']==20)
        original=pickle.load((BASE/'fresh_fit_and_rho.pkl').open('rb'))
        np.testing.assert_allclose(rec['rho'],original['rho'],atol=1e-12)
        print('Rank3 fixed-point reproduction smoke passed.',root,flush=True);return root
    with parallel_config(backend='loky',inner_max_num_threads=1):
        settings=Parallel(n_jobs=10)(delayed(prepare)(d,i,X,y,locations,root) for d in DELTAS_SEARCH for i in (False,True))
        paths=Parallel(n_jobs=36)(delayed(rho_path_search)(s,name,y,root) for s in settings for name in s['fits'])
    by_name={s['name']:s for s in settings};errors=[p for p in paths if 'error' in p]
    jobs=[];inventory=[]
    for path in paths:
        if 'error' in path:continue
        setting=by_name[path['setting_name']]
        for rec in path['records']:
            inventory.append(dict(rec,setting_name=setting['name'],Delta=setting['Delta'],intercept=setting['intercept']))
            control=setting['name']=='d2_int0' and rec['auxiliary']=='LS' and rec['rho_updates']==20
            if abs(rec['correlation'])<.001 and not control:continue
            names=[n for n in setting['fits'] if n!='LS'] if rec['auxiliary']=='LS' else list(dict.fromkeys([rec['auxiliary'],'fista16']))
            jobs.append((setting,rec,names))
    tabulate(inventory).to_csv(root/'rho_inventory.csv',index=False)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        bundles=Parallel(n_jobs=36)(delayed(posterior)(s,r,n,y,ages,target,root) for s,r,n in jobs)
    errors += [b for b in bundles if 'error' in b]
    rows=[r for b in bundles if 'error' not in b for r in b['results']]
    summary=tabulate(rows);summary.to_csv(root/'summary.csv',index=False)
    eligible=summary[summary.correlation.abs()>=.001].sort_values('TV_to_rank3')
    candidates=pd.concat([eligible.head(10),eligible[eligible.correlation.abs()>=.05].head(10),
        eligible[~eligible.intercept].head(5),eligible[eligible.matched].head(5)]).drop_duplicates('id')
    candidates.to_csv(root/'selected_for_validation.csv',index=False)
    # Re-estimate rho itself using finer integration and a second seed.
    keys=list(dict.fromkeys((r.setting_name,r.auxiliary) for r in candidates.itertuples()))
    with parallel_config(backend='loky',inner_max_num_threads=1):
        new_paths=Parallel(n_jobs=36)(delayed(rho_path_search)(by_name[k],a,y,root,power=10,seed=42) for k,a in keys)
    mapping={(p['setting_name'],p['auxiliary']):p for p in new_paths if 'error' not in p}
    errors += [p for p in new_paths if 'error' in p]
    original_records={r['id']:r for r in rows};validation_jobs=[]
    for row in candidates.itertuples():
        key=(row.setting_name,row.auxiliary)
        if key not in mapping:continue
        reestimated=next(r for r in mapping[key]['records'] if r['rho_updates']==row.rho_updates)
        old=original_records[row.id];s=by_name[row.setting_name]
        validation_jobs.extend([(s,old,[row.posterior],123),(s,reestimated,[row.posterior],42)])
    # Multiple posterior names can share a state checkpoint; group them before
    # parallel execution to avoid competing writes to the same files.
    grouped={}
    for s,r,n,seed in validation_jobs:
        key=(r['case'],r['rho_power'],r['rho_seed'],seed)
        if key not in grouped:grouped[key]=[s,r,[],seed]
        grouped[key][2].extend(n)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        checked=Parallel(n_jobs=36)(delayed(posterior)(s,r,list(dict.fromkeys(names)),y,ages,target,root,10,seed)
            for s,r,names,seed in grouped.values())
    errors += [b for b in checked if 'error' in b]
    validated=[r for b in checked if 'error' not in b for r in b['results']]
    tabulate(validated).to_csv(root/'validation.csv',index=False)
    final=[]
    for ident in candidates.id:
        pair=[r for r in validated if r['id']==ident]
        if len(pair)!=2:continue
        a=next(r for r in pair if r['posterior_seed']==123);b=next(r for r in pair if r['posterior_seed']==42)
        row=tabulate([a]).iloc[0].to_dict()
        row.update(mean_TV_to_rank3=(a['TV_to_rank3']+b['TV_to_rank3'])/2,P1_min=min(a['P_1'],b['P_1']),
            P1_max=max(a['P_1'],b['P_1']),correlation_check=b['correlation'],
            rho_reestimation_difference=float(np.linalg.norm(a['rho']-b['rho'],'fro')),
            check_relative_residual=b['relative_residual'],numerical_TV=.5*np.abs(a['conditional']-b['conditional']).sum(),
            check_checkpoint=b['checkpoint'])
        final.append(row)
    ranked=pd.DataFrame(final).sort_values('mean_TV_to_rank3');ranked.insert(0,'rank',np.arange(1,len(ranked)+1))
    ranked.to_csv(root/'ranked.csv',index=False)
    (root/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    print('FIXED-POINT PARAMETER SEARCH COMPLETE',root,flush=True);return root


if __name__=='__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
