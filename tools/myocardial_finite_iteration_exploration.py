"""Finite-iteration heuristic sensitivity; all settings and diagnostics retained."""
from tools.myocardial_exploration import *
from tools.myocardial_exploration_refinement import evaluate_refined
from WERM.myocardial_early_stopping import early_logistic_fit
from WERM.myocardial_numerics import logistic_prox_fast

FIT_BUDGETS = (1, 3, 10, 30, 100, 300, 1000)
SOURCE = Path('results/myocardial_exploration_4026d0e775d2')


def reference_distribution():
    import jax
    jax.config.update('jax_enable_x64', True)
    ref = pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl','rb'))
    _, _, ages, locations = load_data()
    a=np.asarray(ref['unique_ages']).ravel();p=np.asarray(ref['posterior']).ravel()
    keep=np.isin(a,ages[locations]);a,p=a[keep],p[keep];p=p/p.sum()
    return dict(zip(a,p))


def finite_fit(X,y,phi,budget):
    fits=[early_logistic_fit(X,y,w,budget) for w in phi]
    B=np.column_stack([f.coef for f in fits]);theta=X@B
    V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
    if not np.isfinite(V).all():raise FloatingPointError('Nonfinite adjusted fit')
    return dict(theta=theta,V=V,b=b,coef=B,penalty='none',strength=0.,budget=budget,
        checks=[dict(kkt=f.kkt_violation,iterations=f.n_iter,success=f.converged,
                     objective=f.objective,norm=float(np.linalg.norm(f.coef))) for f in fits])


def auxiliary_task(delta,budget,X,y,root):
    path=root/f'aux_d{delta}_early{budget}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    setting=pickle.load((SOURCE/f'fits_delta{delta}.pkl').open('rb'))
    fit=finite_fit(X,y,setting['phi'],budget)
    pilot_eta,psi=pilot(fit['theta'],y,setting['eta'],'logistic')
    S=fit['V'].T@fit['V']/len(y)
    import WERM.myocardial_approximate as implementation
    old=implementation.logistic_prox;implementation.logistic_prox=logistic_prox_fast
    try:
        scenarios=[dict(r,auxiliary=f'early{budget}',pilot=pilot_eta,loss='logistic',
                        auxiliary_budget=budget) for r in rho_path(S,setting['phi'],psi,len(y)/X.shape[1],'logistic')]
    finally:implementation.logistic_prox=old
    result=dict(delta=delta,fit=fit,scenarios=scenarios)
    atomic_save(path,result);print('Auxiliary complete:',delta,budget,flush=True)
    return result


def score_fit(rec,states,setting,name,fit,X,y,ages,reference,root,tag):
    label=rec['label']+'_'+name+'_'+tag;path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    rho=rec['scenario']['rho'];ll=[]
    for e,state in zip(setting['eta'],states):
        psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
        ll.append(gaussian_logistic_loglikelihood(fit['V'],y,rho,state['lambda'],state['kappa'],psi,quadrature_order=160))
    ll=np.asarray(ll);joint=ll+np.log(setting['prior'])
    probs=np.exp(joint-logsumexp(joint));curve=np.exp(ll[1:]-logsumexp(ll[1:]))
    a=ages[setting['eta'][1:]];mapping=dict(zip(a,curve))
    tv=.5*sum(abs(reference[k]-mapping.get(k,0.)) for k in reference)
    cumulative=np.cumsum(curve);lo=a[min(np.searchsorted(cumulative,.05),len(a)-1)];hi=a[min(np.searchsorted(cumulative,.95),len(a)-1)]
    aux=rec['scenario']['auxiliary'];aux_budget=rec['scenario'].get('auxiliary_budget',0)
    group='least_squares' if aux=='least_squares' else ('logistic_matched' if aux_budget==fit['budget'] else 'logistic_different_budgets')
    result=dict(label=label,case=rec['label'],tag=tag,Delta=setting['delta_min'],auxiliary=aux,
        auxiliary_budget=aux_budget,posterior_fit=name,posterior_budget=fit['budget'],
        rho_updates=rec['scenario']['iteration'],group=group,
        P_1=float(probs[1:].sum()),conditional=curve,ages=a,log_likelihood=ll,
        mass_above55=float(curve[a>55].sum()),peak_age=float(a[np.argmax(curve)]),
        TV_to_reference=tv,interval90_lower=float(lo),interval90_upper=float(hi),
        max_location_probability=float(curve.max()),
        moment_residual=rec['scenario']['moment_residual'],
        FP_residual_max=max(s['training_residual'] for s in states),
        fit_kkt_max=max(c['kkt'] for c in fit['checks']),
        coefficient_norm_max=max(c['norm'] for c in fit['checks']),
        actual_fit_iterations_max=max(c['iterations'] for c in fit['checks']))
    atomic_save(path,result);return result


def refined_states(rec,setting,n,p,root,power,seed):
    path=root/(rec['label']+f'_states_q{power}_seed{seed}.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    import WERM.myocardial_approximate as implementation
    old=implementation.logistic_prox;implementation.logistic_prox=logistic_prox_fast
    try:
        states=[]
        for e in setting['eta']:
            psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            states.append(implementation.logistic_state(rec['scenario']['rho'],setting['phi'],psi,n/p,
                          sample_power=power,max_nfev=100,seed=seed))
    finally:implementation.logistic_prox=old
    atomic_save(path,states);print('Numerical validation complete:',rec['label'],power,seed,flush=True)
    return states


def run(smoke=False):
    X,y,ages,locations=load_data();reference=reference_distribution()
    specification=dict(fit_budgets=FIT_BUDGETS,deltas=DELTAS,rho_updates=ITERATIONS,
        intercept=False,penalty='none',fit_start='zero',step='reciprocal exact global weighted logistic curvature bound',
        auxiliary='least squares or finite FISTA logistic',count_prior=[.5,.5],
        primary=dict(power=8,budget=60,seed=42),validation=[dict(power=9,budget=100,seed=42),dict(power=10,budget=100,seed=123)],
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['tools/myocardial_finite_iteration_exploration.py','WERM/myocardial_early_stopping.py',
             'WERM/proximal.py','WERM/myocardial_approximate.py','WERM/signal_strength.py']})
    fingerprint=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_finite_iteration_'+fingerprint);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(specification,indent=2))
    print('Checkpoint directory:',root,flush=True)
    if smoke:
        setting=pickle.load((SOURCE/'fits_delta1.pkl').open('rb'))
        fit=finite_fit(X,y,setting['phi'],3)
        assert max(c['iterations'] for c in fit['checks'])<=3
        print('Smoke fit:',fit['checks'],flush=True)
        return root
    with parallel_config(backend='loky',inner_max_num_threads=1):
        auxiliary=Parallel(n_jobs=21)(delayed(auxiliary_task)(d,b,X,y,root) for d in DELTAS for b in FIT_BUDGETS)
        settings={}
        for d in DELTAS:
            original=pickle.load((SOURCE/f'fits_delta{d}.pkl').open('rb'))
            fits={f'early{r["fit"]["budget"]}':r['fit'] for r in auxiliary if r['delta']==d}
            scenarios=[r for r in original['scenarios'] if r['auxiliary']=='least_squares']
            scenarios += [s for r in auxiliary if r['delta']==d for s in r['scenarios']]
            settings[d]=dict(original,fits=fits,scenarios=scenarios)
        tasks=[(s,r) for s in settings.values() for r in s['scenarios']]
        # State calculation is shared across all coefficient budgets for a rho.
        state_settings=[dict(s,fits={}) for s,r in tasks]
        records=Parallel(n_jobs=36)(delayed(evaluate_refined)(ss,r,X,y,ages,root,8,60)
                                   for ss,(s,r) in zip(state_settings,tasks))
        rows=Parallel(n_jobs=36)(delayed(score_fit)(rec,rec['states'],settings[rec['delta_min']],name,fit,
            X,y,ages,reference,root,'primary') for rec in records
            for name,fit in settings[rec['delta_min']]['fits'].items())
    summary=pd.DataFrame([{k:v for k,v in r.items() if k not in ('conditional','ages','log_likelihood')} for r in rows])
    summary.to_csv(root/'summary.csv',index=False)
    # No P(1) eligibility threshold. Retain shape leaders and nearby probability
    # trade-offs in each auxiliary class. The selection is explicitly post hoc.
    selections=[]
    for group,data in summary.groupby('group'):
        best=data.sort_values('TV_to_reference').head(3)
        near=data[data.TV_to_reference<=data.TV_to_reference.min()+.15].sort_values('P_1',ascending=False).head(2)
        selections.extend([best,near])
    selected=pd.concat(selections).drop_duplicates('label')
    selected.to_csv(root/'selected_for_validation.csv',index=False)
    by_case={r['label']:r for r in records}
    jobs=[(by_case[c],power,seed) for c in selected.case.unique() for power,seed in ((9,42),(10,123))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        states=Parallel(n_jobs=min(36,len(jobs)))(delayed(refined_states)(r,settings[r['delta_min']],len(y),X.shape[1],root,power,seed)
                                              for r,power,seed in jobs)
        checked=Parallel(n_jobs=36)(delayed(score_fit)(r,state,settings[r['delta_min']],row.posterior_fit,
            settings[r['delta_min']]['fits'][row.posterior_fit],X,y,ages,reference,root,f'q{power}_seed{seed}')
            for (r,power,seed),state in zip(jobs,states) for row in selected[selected.case==r['label']].itertuples())
    validation=pd.DataFrame([{k:v for k,v in r.items() if k not in ('conditional','ages','log_likelihood')} for r in checked])
    validation.to_csv(root/'validation.csv',index=False)
    ranked=validation.groupby(['case','posterior_fit','group'],as_index=False).agg(
        P1_min=('P_1','min'),P1_max=('P_1','max'),mean_TV=('TV_to_reference','mean'),
        max_TV=('TV_to_reference','max'),mass_above55_min=('mass_above55','min'),
        peak_min=('peak_age','min'),peak_max=('peak_age','max'),moment_residual=('moment_residual','max'))
    ranked=ranked.sort_values('mean_TV');ranked.to_csv(root/'ranked.csv',index=False)
    print('FINITE-ITERATION EXPLORATION COMPLETE:',len(summary),'primary;',len(validation),'refined checks.',flush=True)
    print(ranked.to_string(index=False),flush=True)
    return root


if __name__=='__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
