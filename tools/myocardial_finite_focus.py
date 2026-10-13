"""Declared follow-up: smaller Delta, denser LS rho path, optimizer sensitivity.

This post-hoc refinement is retained alongside (not substituted for) the broad
grid. No count-probability threshold is imposed when selecting curve candidates.
"""
from tools.myocardial_finite_iteration_exploration import *

FOCUS_DELTAS=(1,2,5,10)
FOCUS_UPDATES=(4,6,8,12,16,20,24,30,40,60)
FISTA_BUDGETS=(2,4,6,8,12,16,20,25,40,60,100,300)
LBFGS_BUDGETS=(1,2,3,5,10,20,40,80)


def finite_lbfgs_column(X,y,w,budget):
    if isinstance(budget,bool) or int(budget)!=budget or budget<1:
        raise ValueError('positive integer budget required')
    wn=w/w.sum()
    def objective(beta):
        z=X@beta
        return float(wn@(np.logaddexp(0,z)-y*z)), X.T@(wn*(expit(z)-y))
    return minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',
        options=dict(maxiter=int(budget),ftol=0.,gtol=0.,maxcor=10,maxls=20))


def lbfgs_fit(X,y,phi,budget):
    fitted=[finite_lbfgs_column(X,y,w,budget) for w in phi]
    B=np.column_stack([f.x for f in fitted]);theta=X@B
    V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
    if not np.isfinite(V).all():raise FloatingPointError('Nonfinite adjusted fit')
    return dict(theta=theta,V=V,b=b,coef=B,budget=budget,penalty='none',strength=0.,
        checks=[dict(kkt=float(np.max(abs(f.jac))),iterations=f.nit,success=bool(f.success),
                     norm=float(np.linalg.norm(f.x)),objective=float(f.fun)) for f in fitted])


def prepare_focus(delta,X,y,ages,locations,root):
    path=root/f'fits_delta{delta}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    eta,prior,phi=configuration_prior(len(y),locations[(locations>=delta)&(locations<=len(y)-delta)])
    fits={f'fista{b}':finite_fit(X,y,phi,b) for b in FISTA_BUDGETS}
    fits.update({f'lbfgs{b}':lbfgs_fit(X,y,phi,b) for b in LBFGS_BUDGETS})
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    theta=X@B;V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='squared')
    pilot_eta,psi=pilot(theta,y,eta,'squared');S=V.T@V/len(y);rho=np.eye(2);scenarios=[]
    for it in range(1,max(FOCUS_UPDATES)+1):
        state=squared_logistic_state(rho,phi,psi,len(y)/X.shape[1])
        inv=np.linalg.pinv(state['lambda'],rcond=1e-6)
        rho=project_covariance(.75*rho+.25*project_covariance(inv.T@(S-state['kappa'])@inv))
        if it in FOCUS_UPDATES:
            state=squared_logistic_state(rho,phi,psi,len(y)/X.shape[1])
            scenarios.append(dict(rho=rho.copy(),iteration=it,auxiliary='least_squares',loss='squared',
                pilot=pilot_eta,moment_residual=float(np.linalg.norm(state['moment']-S)),state_residual=0.))
    # Alternative implicit regularization in the auxiliary fit too, with exactly
    # matched budget/optimizer indicated in the postprocessing tables.
    if delta==1:
        import WERM.myocardial_approximate as implementation
        old=implementation.logistic_prox;implementation.logistic_prox=logistic_prox_fast
        try:
            for budget in (5,10,20):
                fit=fits[f'lbfgs{budget}'];e,psi=pilot(fit['theta'],y,eta,'logistic')
                S=fit['V'].T@fit['V']/len(y)
                scenarios.extend(dict(r,auxiliary=f'lbfgs{budget}',auxiliary_budget=budget,pilot=e,loss='logistic')
                    for r in rho_path(S,phi,psi,len(y)/X.shape[1],'logistic') if r['iteration'] in (3,10,30))
        finally:implementation.logistic_prox=old
    result=dict(delta_min=delta,eta=eta,prior=prior,phi=phi,fits=fits,scenarios=scenarios)
    atomic_save(path,result);print('Focused fits prepared:',delta,flush=True);return result


def with_group(result):
    result=result.copy()
    # For implicit regularization, matching means BOTH optimizer and budget.
    if result['auxiliary']!='least_squares':
        result['group']='logistic_matched' if result['auxiliary']==result['posterior_fit'] else 'logistic_different_budgets'
    result['posterior_optimizer']='lbfgs' if result['posterior_fit'].startswith('lbfgs') else 'fista'
    return result


def run_focus(base):
    X,y,ages,locations=load_data();reference=reference_distribution()
    digest=hashlib.sha256(Path('tools/myocardial_finite_focus.py').read_bytes()).hexdigest()[:12]
    root=base/('focus_'+digest);root.mkdir(exist_ok=True)
    print('Focused checkpoint directory:',root,flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        settings_list=Parallel(n_jobs=4)(delayed(prepare_focus)(d,X,y,ages,locations,root) for d in FOCUS_DELTAS)
        settings={s['delta_min']:s for s in settings_list}
        tasks=[(s,r) for s in settings_list for r in s['scenarios']]
        records=Parallel(n_jobs=36)(delayed(evaluate_refined)(dict(s,fits={}),r,X,y,ages,root,8,60) for s,r in tasks)
        results=Parallel(n_jobs=36)(delayed(score_fit)(r,r['states'],settings[r['delta_min']],name,fit,
            X,y,ages,reference,root,'primary') for r in records for name,fit in settings[r['delta_min']]['fits'].items())
    summary=pd.DataFrame([{k:v for k,v in with_group(r).items() if k not in ('conditional','ages','log_likelihood')} for r in results])
    summary.to_csv(root/'summary.csv',index=False)
    chosen=[]
    for (group,optimizer),data in summary.groupby(['group','posterior_optimizer']):
        chosen.extend([data.sort_values('TV_to_reference').head(2),
            data[data.TV_to_reference<=data.TV_to_reference.min()+.1].sort_values('P_1',ascending=False).head(1)])
    selected=pd.concat(chosen).drop_duplicates('label');selected.to_csv(root/'selected_for_validation.csv',index=False)
    by_case={r['label']:r for r in records}
    jobs=[(by_case[c],power,seed) for c in selected.case.unique() for power,seed in ((9,42),(10,123))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        states=Parallel(n_jobs=min(36,len(jobs)))(delayed(refined_states)(r,settings[r['delta_min']],len(y),X.shape[1],root,power,seed)
                                              for r,power,seed in jobs)
        checks=Parallel(n_jobs=36)(delayed(score_fit)(r,state,settings[r['delta_min']],row.posterior_fit,
            settings[r['delta_min']]['fits'][row.posterior_fit],X,y,ages,reference,root,f'q{power}_seed{seed}')
            for (r,power,seed),state in zip(jobs,states) for row in selected[selected.case==r['label']].itertuples())
    validation=pd.DataFrame([{k:v for k,v in with_group(r).items() if k not in ('conditional','ages','log_likelihood')} for r in checks])
    validation.to_csv(root/'validation.csv',index=False)
    ranked=validation.groupby(['case','posterior_fit','group'],as_index=False).agg(
        P1_min=('P_1','min'),P1_max=('P_1','max'),mean_TV=('TV_to_reference','mean'),max_TV=('TV_to_reference','max'),
        mass_above55_min=('mass_above55','min'),peak_min=('peak_age','min'),peak_max=('peak_age','max'),
        moment_residual=('moment_residual','max')).sort_values('mean_TV')
    ranked.to_csv(root/'ranked.csv',index=False)
    print('FOCUSED EXPLORATION COMPLETE:',len(summary),'primary comparisons.',flush=True)
    print(ranked.to_string(index=False),flush=True);return root


if __name__=='__main__':
    import sys
    run_focus(Path(sys.argv[1]))
