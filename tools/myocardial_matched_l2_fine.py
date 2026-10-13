"""Fine matched-L2 full coefficient solves; separate from early stopping."""
from tools.myocardial_finite_iteration_exploration import *
from tools.myocardial_exact_no_intercept import score as matched_score


def penalty_grid():
    return np.r_[np.logspace(-8,-4,33),3e-4,1e-3]


def solve_logistic_l2(X,y,w,strength):
    """Damped Newton to stationarity, not a chosen finite fitting budget.

    Objective = weighted mean logistic loss + strength * ||beta||^2.
    The strong-convexity gap bound ||gradient||^2/(4*strength) is recorded.
    A safety limit raises an error rather than returning an early-stopped fit.
    """
    if not np.isfinite(strength) or strength<=0:raise ValueError('positive finite L2 strength required')
    X=np.asarray(X,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('invalid sample weights')
    wn=w/w.sum();beta=np.zeros(X.shape[1]);eye=np.eye(X.shape[1])
    def value(b):
        z=X@b
        return float(wn@(np.logaddexp(0,z)-y*z)+strength*(b@b))
    for it in range(101):
        z=X@beta;prob=expit(z);grad=X.T@(wn*(prob-y))+2*strength*beta
        kkt=float(np.max(abs(grad)));gap=float(grad@grad/(4*strength))
        if kkt<=1e-10 and gap<=1e-8:
            return dict(coef=beta,kkt=kkt,gap_bound=gap,iterations=it,objective=value(beta),success=True)
        if it==100:break
        H=X.T@((wn*prob*(1-prob))[:,None]*X)+2*strength*eye
        direction=np.linalg.solve(H,grad);decrease=float(grad@direction);old=value(beta);step=1.
        for backtrack in range(60):
            candidate=beta-step*direction
            if value(candidate)<=old-1e-4*step*decrease+1e-15:
                beta=candidate;break
            step*=.5
        else:raise RuntimeError('Full L2 solve line search failed')
    raise RuntimeError(f'Full L2 solve failed stationarity: KKT={kkt}, gap={gap}')


def full_fit(X,y,phi,strength):
    fits=[solve_logistic_l2(X,y,w,strength) for w in phi]
    B=np.column_stack([f['coef'] for f in fits]);theta=X@B
    V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
    return dict(coef=B,theta=theta,V=V,b=b,penalty='l2',strength=float(strength),
        checks=[{k:v for k,v in f.items() if k!='coef'} for f in fits])


def prepare_full(strength,setting,X,y,root):
    name=f'l2full_{strength:.12g}';path=root/('aux_'+name+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    fit=full_fit(X,y,setting['phi'],strength)
    e,psi=pilot(fit['theta'],y,setting['eta'],'logistic');S=fit['V'].T@fit['V']/len(y)
    import WERM.myocardial_approximate as implementation
    old=implementation.logistic_prox;implementation.logistic_prox=logistic_prox_fast
    try:
        scenarios=[dict(r,auxiliary=name,pilot=e,loss='logistic')
            for r in rho_path(S,setting['phi'],psi,len(y)/X.shape[1],'logistic')]
    finally:implementation.logistic_prox=old
    result=(name,fit,scenarios);atomic_save(path,result)
    print('Full matched L2 fit and rho path ready:',strength,flush=True);return result


def score_full(rec,states,fit,setting,X,y,ages,reference,root,tag):
    path=root/(rec['label']+'_'+tag+'_score.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    result=matched_score(rec,states,fit,setting,X,y,ages,reference,tag)
    result.update(label=rec['label']+'_'+tag,tag=tag,ages=ages[setting['eta'][1:]],
        posterior_fit=rec['scenario']['auxiliary'],group='fully_solved_matched_L2',
        Delta=1,rho_updates=rec['scenario']['iteration'],auxiliary=rec['scenario']['auxiliary'],
        auxiliary_strength=fit['strength'],posterior_strength=fit['strength'],
        gap_bound_max=max(c['gap_bound'] for c in fit['checks']),
        actual_fit_iterations_max=max(c['iterations'] for c in fit['checks']))
    atomic_save(path,result);return result


def run_full_matched(base):
    X,y,ages,locations=load_data();reference_map=reference_distribution()
    reference=np.array([reference_map[a] for a in ages[locations]])
    setting=pickle.load((SOURCE/'fits_delta1.pkl').open('rb'))
    digest=hashlib.sha256(Path('tools/myocardial_matched_l2_fine.py').read_bytes()).hexdigest()[:12]
    root=base/('matched_l2_full_'+digest);root.mkdir(exist_ok=True)
    print('Fine full-solve matched L2 directory:',root,flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        prepared=Parallel(n_jobs=35)(delayed(prepare_full)(v,setting,X,y,root) for v in penalty_grid())
        tasks=[(name,fit,r) for name,fit,scenarios in prepared for r in scenarios]
        records=Parallel(n_jobs=36)(delayed(evaluate_refined)(dict(setting,fits={}),r,X,y,ages,root,8,60) for name,fit,r in tasks)
        results=Parallel(n_jobs=36)(delayed(score_full)(rec,rec['states'],fit,setting,X,y,ages,reference,root,'primary')
                                    for rec,(name,fit,r) in zip(records,tasks))
    def table(rows):return pd.DataFrame([{k:v for k,v in r.items() if k not in ('conditional','ages')} for r in rows])
    summary=table(results);summary.to_csv(root/'summary.csv',index=False)
    # Shape first, with count probability a trade-off, never a hard filter.
    closest=summary.sort_values('TV_to_reference').head(4)
    nearby=summary[summary.TV_to_reference<=summary.TV_to_reference.min()+.1].sort_values('P_1',ascending=False).head(2)
    selected=pd.concat([closest,nearby]).drop_duplicates('case')
    selected.to_csv(root/'selected_for_validation.csv',index=False)
    case_map={r['label']:(r,t[1]) for r,t in zip(records,tasks)}
    jobs=[(case_map[c],power,seed) for c in selected.case for power,seed in ((9,42),(10,123))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        states=Parallel(n_jobs=len(jobs))(delayed(refined_states)(r,setting,len(y),X.shape[1],root,power,seed)
                                        for (r,fit),power,seed in jobs)
        checks=Parallel(n_jobs=36)(delayed(score_full)(r,state,fit,setting,X,y,ages,reference,root,f'q{power}_seed{seed}')
                                  for ((r,fit),power,seed),state in zip(jobs,states))
    validation=table(checks);validation.to_csv(root/'validation.csv',index=False)
    ranked=validation.groupby(['case','posterior_fit','group','strength'],as_index=False).agg(
        P1_min=('P_1','min'),P1_max=('P_1','max'),mean_TV=('TV_to_reference','mean'),max_TV=('TV_to_reference','max'),
        mass_above55_min=('mass_above55','min'),peak_min=('peak_age','min'),peak_max=('peak_age','max'),
        moment_residual=('moment_residual','max'),fit_kkt_max=('fit_kkt_max','max'),gap_bound_max=('gap_bound_max','max')).sort_values('mean_TV')
    ranked.to_csv(root/'ranked.csv',index=False)
    print('FULL-SOLVE MATCHED L2 COMPLETE:',len(summary),'primary comparisons.',flush=True)
    print(ranked.to_string(index=False),flush=True);return root


if __name__=='__main__':
    import sys
    run_full_matched(Path(sys.argv[1]))
