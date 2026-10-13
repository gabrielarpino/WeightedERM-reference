"""Direct constrained rho fitting: previous top 20 and matched logistic variants."""
from tools.myocardial_final_redo import *
from WERM.myocardial_direct_moments import solve_direct, STARTS
from WERM.myocardial_intercept import (fit_with_intercept,squared_intercept_state,
    logistic_intercept_state,intercept_loglikelihood)
from WERM.myocardial_finite_intercept import finite_intercept_fit
from WERM.myocardial_no_intercept_solvers import fit_column
from tools.myocardial_matched_l2_fine import full_fit
from tools.myocardial_best_ones import curve_metrics

TOP=Path('results/myocardial_updated_best_ones_64c8367cda88/top_twenty.csv')
OFFSET=Path('results/myocardial_intercept_exploration_f53f59063391')
OLD=Path('results/myocardial_exploration_4026d0e775d2')
FOCUS=Path('results/myocardial_finite_iteration_443c17cf2df6/focus_354ed8f1e782')


def specifications():
    specs=[]
    for row in pd.read_csv(TOP).to_dict('records'):
        specs.append(dict(id=f"old_rank{row['rank']}",origin='previous_top20',old=row,
            intercept=bool(row['intercept']),Delta=int(row['Delta']),auxiliary=row['auxiliary'],posterior=row['posterior']))
    for intercept in (False,True):
        for method in ('fista','lbfgs','gd'):
            for budget in (20,30,40,50):
                label=f'finite:{method}:{budget}'
                specs.append(dict(id=f'{method}{budget}_intercept{int(intercept)}',origin='matched_finite',
                    intercept=intercept,Delta=2,auxiliary=label,posterior=label,method=method,budget=budget))
        for strength in (1e-8,1e-6,1e-5,4.21696503429e-5,1e-4):
            label=f'l2:{strength:.12g}'
            specs.append(dict(id=f'l2_{strength:.12g}_intercept{int(intercept)}',origin='matched_L2',
                intercept=intercept,Delta=2,auxiliary=label,posterior=label,strength=strength))
    return specs


def ls_fit(X,y,phi,intercept):
    if intercept:return fit_with_intercept(X,y,phi,'none',0.,loss='squared')
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    theta=X@B;V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='squared')
    return dict(theta=theta,V=V,b=b,coef=B)


def archived_fit(label,old,setting,X,y):
    if label=='LS':return ls_fit(X,y,setting['phi'],old['intercept'])
    if 'iterations' in label:
        key=('lbfgs' if 'L-BFGS' in label else 'fista')+label.split()[-2]
        stored=pickle.load((FOCUS/f"fits_delta{old['Delta']}.pkl").open('rb'))
        np.testing.assert_allclose(stored['phi'],setting['phi'],atol=1e-14)
        return stored['fits'][key]
    family=label.split()[1].lower();strength=float(label.split()[2])
    folder=OFFSET/'penalty_resolution' if old['intercept'] else OLD/'denser_penalty_grid'
    path=folder/f'fit_{family}_{strength:.8g}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    # Original intercept auxiliary fits are in the initial offset checkpoint.
    if old['intercept']:
        stored=pickle.load((OFFSET/'offset_fits.pkl').open('rb'))
        return stored['fits'][f'{family}_{strength:g}']
    raise FileNotFoundError(path)


def prepare(spec,X,y,locations,root):
    path=root/(spec['id']+'_fits.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    try:
        d=spec['Delta'];eta,prior,phi=configuration_prior(len(y),locations[(locations>=d)&(locations<=len(y)-d)])
        setting=dict(eta=eta,prior=prior,phi=phi)
        if spec['origin']=='previous_top20':
            aux=archived_fit(spec['auxiliary'],spec['old'],setting,X,y)
            post=archived_fit(spec['posterior'],spec['old'],setting,X,y)
        elif spec['origin']=='matched_L2':
            aux=(fit_with_intercept(X,y,phi,'l2',spec['strength']) if spec['intercept']
                 else full_fit(X,y,phi,spec['strength']))
            post=aux
        else:
            if spec['intercept']:aux=finite_intercept_fit(X,y,phi,spec['method'],spec['budget'])
            else:
                cols=[fit_column(X,y,w,spec['method'],spec['budget']) for w in phi]
                B=np.column_stack([r['coef'] for r in cols]);theta=X@B
                V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
                aux=dict(theta=theta,V=V,b=b,checks=cols)
            post=aux
        loss='squared' if spec['auxiliary']=='LS' else 'logistic'
        e,psi=pilot(aux['theta'],y,eta,loss);S=aux['V'].T@aux['V']/len(y)
        key=hashlib.sha256(S.tobytes()+phi.tobytes()+psi.tobytes()+str((loss,spec['intercept'])).encode()).hexdigest()[:16]
        result=dict(spec=spec,setting=setting,auxiliary=aux,posterior=post,S=S,psi=psi,pilot=e,
            loss=loss,aux_key=key,delta=len(y)/X.shape[1],rate=float(y.mean()))
    except Exception:result=dict(error=traceback.format_exc(),spec=spec)
    atomic_save(path,result);return result


def state_at(prepared,rho,psi,power=10,seed=123,loss=None):
    loss=prepared['loss'] if loss is None else loss
    phi=prepared['setting']['phi'];delta=prepared['delta']
    if prepared['spec']['intercept']:
        if loss=='squared':return squared_intercept_state(rho,phi,psi,delta,prepared['rate'])
        return logistic_intercept_state(rho,phi,psi,delta,prepared['rate'],sample_power=power,max_nfev=100,seed=seed)
    if loss=='squared':return squared_logistic_state(rho,phi,psi,delta)
    import WERM.myocardial_approximate as impl
    old=impl.logistic_prox;impl.logistic_prox=logistic_prox_fast
    try:return impl.logistic_state(rho,phi,psi,delta,sample_power=power,max_nfev=100,seed=seed)
    finally:impl.logistic_prox=old


def fit_rho(prepared,start_index,root):
    path=root/f"rho_{prepared['aux_key']}_start{start_index}.pkl"
    if path.exists():return pickle.load(path.open('rb'))
    try:
        result=solve_direct(lambda rho:state_at(prepared,rho,prepared['psi'],power=9),prepared['S'],STARTS[start_index])
        result.update(aux_key=prepared['aux_key'],start_index=start_index)
    except Exception:result=dict(error=traceback.format_exc(),aux_key=prepared['aux_key'],start_index=start_index)
    atomic_save(path,result)
    print('Direct fit:',prepared['aux_key'],start_index,result.get('relative_residual',result.get('error')),flush=True)
    return result


def validate_rho(prepared,fit,root):
    path=root/f"rho_{prepared['aux_key']}_validation1024.pkl"
    if path.exists():return pickle.load(path.open('rb'))
    state=state_at(prepared,fit['rho'],prepared['psi'],power=10)
    moment=state.get('moment')
    if moment is None:moment=state['lambda'].T@fit['rho']@state['lambda']+state['kappa']
    residual=float(np.linalg.norm(moment-prepared['S'],'fro'))
    result=dict(fit,moment_residual_1024=residual,
        relative_residual_1024=residual/np.linalg.norm(prepared['S'],'fro'),
        inner_residual_1024=state.get('training_residual',0.))
    atomic_save(path,result);return result


def posterior_state(prepared,fit,index,seed,root):
    path=root/f"state_{prepared['aux_key']}_i{index}_seed{seed}.pkl"
    if path.exists():return pickle.load(path.open('rb'))
    try:
        n=prepared['setting']['phi'].shape[1];e=prepared['setting']['eta'][index]
        psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
        state=state_at(prepared,fit['rho'],psi,seed=seed,loss='logistic')
        result=dict(state=state,index=index,seed=seed)
    except Exception:result=dict(error=traceback.format_exc(),aux_key=prepared['aux_key'],index=index,seed=seed)
    atomic_save(path,result);return result


def score(prepared,fit,states,y,ages,reference,seed,root):
    path=root/f"score_{prepared['spec']['id']}_seed{seed}.pkl"
    if path.exists():return pickle.load(path.open('rb'))
    try:
        values=[];setting=prepared['setting'];n=len(y)
        for e,s in zip(setting['eta'],states):
            if 'error' in s:raise RuntimeError(s['error'])
            state=s['state'];psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            if prepared['spec']['intercept']:
                ll=intercept_loglikelihood(prepared['posterior']['V'],y,fit['rho'],state,psi,order=160)
            else:ll=gaussian_logistic_loglikelihood(prepared['posterior']['V'],y,fit['rho'],state['lambda'],state['kappa'],psi,quadrature_order=160)
            values.append(ll)
        result=normalize_posterior(values,setting['prior']);a=ages[setting['eta'][1:]]
        metrics=curve_metrics(a,result['conditional'],reference);rho=fit['rho'];spec=prepared['spec']
        result.update(id=spec['id'],origin=spec['origin'],auxiliary=spec['auxiliary'],posterior=spec['posterior'],
            intercept=spec['intercept'],Delta=spec['Delta'],aux_key=prepared['aux_key'],seed=seed,
            ages=a,rho=rho,rho_11=rho[0,0],rho_12=rho[0,1],rho_22=rho[1,1],
            correlation=rho[0,1]/np.sqrt(rho[0,0]*rho[1,1]),P_1=result['presence'],
            TV=metrics['TV'],peak_age=metrics['peak'],mass_above55=metrics['above55'],
            moment_residual=fit['moment_residual'],relative_residual=fit['relative_residual'],
            optimizer_success=fit['optimizer_success'],boundary=fit['boundary'],optimality=fit['optimality'],
            inner_residual=fit['inner_residual'],moment_residual_1024=fit['moment_residual_1024'],
            relative_residual_1024=fit['relative_residual_1024'],inner_residual_1024=fit['inner_residual_1024'],
            max_state_residual=max(s['state']['training_residual'] for s in states),
            old_rank=spec.get('old',{}).get('rank'),old_P1=spec.get('old',{}).get('check_P1'),
            old_TV=spec.get('old',{}).get('mean_TV'),checkpoint=str(path),
            configuration=prepared['aux_key']+'_'+hashlib.sha256(prepared['posterior']['V'].tobytes()).hexdigest()[:16])
    except Exception:result=dict(error=traceback.format_exc(),id=prepared['spec']['id'],seed=seed)
    atomic_save(path,result);return result


def tabulate(results):
    skip=('conditional','ages','rho','probabilities','log_likelihood')
    return pd.DataFrame([{k:v for k,v in r.items() if k not in skip} for r in results if 'error' not in r])


def run(smoke=False):
    X,y,ages,locations=load_data();reference=reference_distribution();specs=specifications()
    provenance=dict(specifications=specs,starts=[s.tolist() for s in STARTS],rho_search_draws=512,state_draws=1024,
        primary_seed=123,validation_seed=42,coefficient_fit_reuse=True,
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('tools/myocardial_direct_moment_experiment.py','WERM/myocardial_direct_moments.py',
             'WERM/myocardial_approximate.py','WERM/myocardial_intercept.py',str(TOP))})
    digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_direct_moments_'+digest);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(provenance,indent=2))
    # Reuse exactly the unchanged coefficient fits and analytic LS rho fits.
    # Never import a completed logistic rho search made at another resolution.
    previous=Path('results/myocardial_direct_moments_45eda0849170')
    for spec in specs:
        old=previous/(spec['id']+'_fits.pkl');new=root/old.name
        if old.exists() and not new.exists():
            saved=pickle.load(old.open('rb'))
            assert json.dumps(saved['spec'],sort_keys=True)==json.dumps(spec,sort_keys=True)
            atomic_save(new,saved)
            if saved['loss']=='squared':
                for i in range(len(STARTS)):
                    src=previous/f"rho_{saved['aux_key']}_start{i}.pkl"
                    dst=root/src.name
                    if src.exists() and not dst.exists():atomic_save(dst,pickle.load(src.open('rb')))
    with parallel_config(backend='loky',inner_max_num_threads=1):
        prepared=Parallel(n_jobs=36)(delayed(prepare)(s,X,y,locations,root) for s in specs)
    errors=[p for p in prepared if 'error' in p]
    (root/'preparation_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} preparation errors; inspect {root}')
    if smoke:
        representative=next(p for p in prepared if p['spec']['id']=='old_rank3')
        check=solve_direct(lambda rho:state_at(representative,rho,representative['psi']),
            representative['S'],STARTS[0],max_nfev=3)
        assert np.isfinite(check['moment_residual'])
        for intercept in (False,True):
            p=next(p for p in prepared if p['spec']['origin']=='matched_finite' and p['spec']['intercept']==intercept)
            state=state_at(p,.01*np.eye(2),p['psi'],power=9)
            assert np.isfinite(state['training_residual'])
        print('All',len(prepared),'fit settings prepared; constrained LS and both logistic state branches smoke passed.',flush=True)
        return root
    unique={p['aux_key']:p for p in prepared}
    with parallel_config(backend='loky',inner_max_num_threads=1):
        fitted=Parallel(n_jobs=36)(delayed(fit_rho)(p,i,root) for p in unique.values() for i in range(len(STARTS)))
    errors += [r for r in fitted if 'error' in r]
    start_table=pd.DataFrame([{k:v for k,v in f.items() if k not in ('trace','rho','parameters','start')} for f in fitted if 'error' not in f])
    start_table.to_csv(root/'multistart_diagnostics.csv',index=False)
    best={}
    for key in unique:
        valid=[f for f in fitted if f['aux_key']==key and 'error' not in f]
        if valid:best[key]=min(valid,key=lambda f:f['moment_residual'])
    with parallel_config(backend='loky',inner_max_num_threads=1):
        validated=Parallel(n_jobs=36)(delayed(validate_rho)(unique[key],fit,root) for key,fit in best.items())
    best={fit['aux_key']:fit for fit in validated}
    atomic_save(root/'selected_rho_fits.pkl',best)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        jobs=[(key,i) for key in best for i in range(len(unique[key]['setting']['eta']))]
        state_results=Parallel(n_jobs=36)(delayed(posterior_state)(unique[key],best[key],i,123,root) for key,i in jobs)
        states={key:[] for key in best}
        for (key,i),state in zip(jobs,state_results):states[key].append(state)
        scores=Parallel(n_jobs=36)(delayed(score)(p,best[p['aux_key']],states[p['aux_key']],y,ages,reference,123,root)
                                   for p in prepared if p['aux_key'] in best)
    errors += [r for r in scores if 'error' in r]
    summary=tabulate(scores);summary.to_csv(root/'summary.csv',index=False)
    ranked=summary.sort_values('TV').drop_duplicates('configuration')
    ranked.to_csv(root/'ranked_primary.csv',index=False)
    candidates=ranked.head(20);candidates.to_csv(root/'selected_for_validation.csv',index=False)
    selected_keys=set(candidates.aux_key)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        jobs=[(key,i) for key in selected_keys for i in range(len(unique[key]['setting']['eta']))]
        checks=Parallel(n_jobs=36)(delayed(posterior_state)(unique[key],best[key],i,42,root) for key,i in jobs)
        checked_states={key:[] for key in selected_keys}
        for (key,i),state in zip(jobs,checks):checked_states[key].append(state)
        chosen=[p for p in prepared if p['spec']['id'] in set(candidates.id)]
        validation=Parallel(n_jobs=36)(delayed(score)(p,best[p['aux_key']],checked_states[p['aux_key']],y,ages,reference,42,root) for p in chosen)
    errors += [r for r in validation if 'error' in r]
    tabulate(validation).to_csv(root/'validation.csv',index=False)
    tops=[]
    for check in validation:
        if 'error' in check:continue
        primary=next(r for r in scores if r.get('id')==check['id'] and 'error' not in r)
        row=tabulate([primary]).iloc[0].to_dict()
        row.update(mean_TV=(primary['TV']+check['TV'])/2,P1_min=min(primary['P_1'],check['P_1']),
            P1_max=max(primary['P_1'],check['P_1']),numerical_TV=.5*np.abs(primary['conditional']-check['conditional']).sum(),
            check_checkpoint=check['checkpoint'])
        tops.append(row)
    final=pd.DataFrame(tops).sort_values('mean_TV');final.insert(0,'rank',np.arange(1,len(final)+1))
    final.to_csv(root/'top_twenty.csv',index=False)
    (root/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    print('DIRECT MOMENT STUDY COMPLETE',root,'errors:',len(errors),flush=True)
    return root


if __name__=='__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
