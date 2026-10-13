"""Exploratory correlated covariance paths targeting the saved rank-3 curve.

These are declared data-informed shrinkage paths, not newly validated estimators.
"""
from tools.myocardial_final_redo import *

BASE=Path('results/myocardial_final_redo_e3d965cc4b9b')
DIRECT=Path('results/myocardial_direct_moments_fde92cd39485')


def blend_covariance(base,estimate,weight,match_trace):
    if not 0<=weight<=1:raise ValueError('weight outside [0,1]')
    base=np.asarray(base,float);estimate=np.asarray(estimate,float)
    if match_trace:estimate=estimate*np.trace(base)/np.trace(estimate)
    out=(1-weight)*base+weight*estimate
    if np.linalg.eigvalsh(out).min()<=0:raise ValueError('blend is not positive definite')
    return out


def make_candidates(data):
    X,y=data['X'],data['y'];n=len(y)
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in data['phi']])
    weighted=B.T@B/n
    split=data['pilot']
    if not 0<split<n:raise ValueError('segmented Gram requires a non-null estimated pilot')
    segmented=np.column_stack([np.linalg.lstsq(X[:split],y[:split],rcond=None)[0],
                              np.linalg.lstsq(X[split:],y[split:],rcond=None)[0]])
    segmented=segmented.T@segmented/n
    prepared=pickle.load((DIRECT/'old_rank3_fits.pkl').open('rb'))
    direct=pickle.load((DIRECT/'selected_rho_fits.pkl').open('rb'))[prepared['aux_key']]['rho']
    estimates=dict(weighted_LS_Gram=weighted,pilot_segment_LS_Gram=segmented,direct_moment=direct)
    rows=[dict(id='baseline',family='baseline',weight=0.,trace_matched=True,rho=data['rho'])]
    for family,estimate in estimates.items():
        for matched in (True,False):
            weights=(.01,.03,.06,.1,.2,.35,.5,.75,1.) if matched else (.001,.003,.01,.03,.06,.1,.2,.5,1.)
            for w in weights:
                rho=blend_covariance(data['rho'],estimate,w,matched)
                rows.append(dict(id=f'{family}_trace{int(matched)}_w{w:g}',family=family,
                    weight=w,trace_matched=matched,rho=rho))
    return rows,estimates


def evaluate(candidate,data,seed,root):
    path=root/f"{candidate['id']}_seed{seed}.pkl"
    if path.exists():return pickle.load(path.open('rb'))
    folder=root/candidate['id'];folder.mkdir(exist_ok=True)
    local=dict(data,rho=candidate['rho'],root=folder)
    try:
        # One candidate per worker; location checkpoints survive interruption.
        states=[location_task(local,(1.,1.),10,seed,i) for i in range(len(data['eta']))]
        result=normalize_posterior([s['log_likelihood'] for s in states],data['prior'])
        rho=candidate['rho'];X,y,phi=data['X'],data['y'],data['phi'];n,p=X.shape
        B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
        V,_=empirical_adjustment(X@B,y,phi,n/p,loss='squared')
        _,psi=pilot(X@B,y,data['eta'],'squared');S=V.T@V/n
        predicted=squared_logistic_state(rho,phi,psi,n/p)['moment']
        residual=float(np.linalg.norm(predicted-S,'fro'))
        result.update(candidate,seed=seed,ages=data['ages'][data['eta'][1:]],P_1=result['presence'],
            correlation=rho[0,1]/np.sqrt(rho[0,0]*rho[1,1]),rho_11=rho[0,0],rho_12=rho[0,1],rho_22=rho[1,1],
            moment_residual=residual,relative_residual=residual/np.linalg.norm(S,'fro'),
            max_state_residual=max(s['state']['training_residual'] for s in states),checkpoint=str(path))
    except Exception:result=dict(error=traceback.format_exc(),id=candidate['id'],seed=seed)
    atomic_save(path,result);print('Candidate complete:',candidate['id'],seed,flush=True);return result


def scalar_results(results,target):
    rows=[]
    for result in results:
        if 'error' in result:continue
        row={k:v for k,v in result.items() if k not in ('rho','ages','conditional','probabilities','log_likelihood')}
        row.update(TV_to_rank3=.5*np.abs(result['conditional']-target['conditional']).sum(),
            P1_difference=result['P_1']-target['P_1'],peak_age=result['ages'][np.argmax(result['conditional'])],
            mass_above55=result['conditional'][result['ages']>55].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run(smoke=False):
    data=pickle.load((BASE/'fresh_fit_and_rho.pkl').open('rb'))
    candidates,estimates=make_candidates(data)
    digest=hashlib.sha256(Path('tools/myocardial_correlated_target.py').read_bytes()+(BASE/'specification.json').read_bytes()).hexdigest()[:12]
    root=Path('results')/('myocardial_correlated_rank3_'+digest);root.mkdir(exist_ok=True)
    atomic_save(root/'candidate_specifications.pkl',dict(candidates=candidates,estimates=estimates))
    (root/'specification.json').write_text(json.dumps(dict(target='previous overall rank3, 1024 draws',
        source=str(BASE),candidate_count=len(candidates),draws=1024,primary_seed=123,check_seed=42,
        meaningful_correlation_cutoff=.05,selection='TV to rank3 among abs(correlation)>=.05; no P1 filter',
        warning='Outcome-targeted covariance sensitivity paths, not validated rho estimators'),indent=2))
    if smoke:
        for c in candidates:assert np.linalg.eigvalsh(c['rho']).min()>0
        tiny=dict(data,eta=data['eta'][:2],prior=np.array([.5,.5]))
        c=dict(candidates[2],id='smoke_only')
        r=evaluate(c,tiny,123,root)
        assert 'error' not in r,r
        print('All',len(candidates),'covariances valid; two-location posterior smoke passed.');return root
    # Reuse only the exact baseline at the same integration setting.
    folder=root/'baseline';folder.mkdir(exist_ok=True)
    for i in range(len(data['eta'])):
        src=BASE/f'eig100_100_q10_seed123_location{i}.pkl';dst=folder/src.name
        if not dst.exists():atomic_save(dst,pickle.load(src.open('rb')))
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=36)(delayed(evaluate)(c,data,123,root) for c in candidates)
    target=next(r for r in results if r['id']=='baseline')
    archived=pickle.load((BASE/'scenario_100_100_q10_seed123.pkl').open('rb'))
    np.testing.assert_allclose(target['conditional'],archived['conditional'],atol=1e-12)
    np.testing.assert_allclose(target['P_1'],archived['presence'],atol=1e-12)
    summary=scalar_results(results,target);summary.to_csv(root/'summary.csv',index=False)
    eligible=summary[summary.correlation.abs()>=.05].sort_values('TV_to_rank3')
    selected=pd.concat([eligible.head(10),eligible.groupby('family',sort=False).head(2)]).drop_duplicates('id')
    selected.to_csv(root/'selected_for_validation.csv',index=False)
    ids=set(selected.id)|{'baseline'}
    with parallel_config(backend='loky',inner_max_num_threads=1):
        checked=Parallel(n_jobs=36)(delayed(evaluate)(c,data,42,root) for c in candidates if c['id'] in ids)
    check_target=next(r for r in checked if r['id']=='baseline')
    validation=scalar_results(checked,check_target);validation.to_csv(root/'validation.csv',index=False)
    ranked=[]
    for row in selected.to_dict('records'):
        check=next(r for r in checked if r['id']==row['id'])
        if 'error' in check:continue
        original=next(r for r in results if r['id']==row['id'])
        check_tv=.5*np.abs(check['conditional']-check_target['conditional']).sum()
        row.update(mean_TV_to_rank3=(row['TV_to_rank3']+check_tv)/2,
            P1_min=min(original['P_1'],check['P_1']),P1_max=max(original['P_1'],check['P_1']),
            numerical_TV=.5*np.abs(original['conditional']-check['conditional']).sum(),check_checkpoint=check['checkpoint'])
        ranked.append(row)
    final=pd.DataFrame(ranked).sort_values('mean_TV_to_rank3');final.insert(0,'rank',np.arange(1,len(final)+1))
    final.to_csv(root/'ranked.csv',index=False)
    errors=[r for r in results+checked if 'error' in r]
    (root/'errors.json').write_text(json.dumps(errors,indent=2))
    print('CORRELATED RANK3 STUDY COMPLETE:',root,'errors:',len(errors),flush=True)
    return root


if __name__=='__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
