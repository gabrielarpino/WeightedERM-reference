"""Transparent rho-initialization sensitivity for finite logistic/logistic fits."""
from tools.myocardial_finite_iteration_exploration import *


def initialization_grid():
    return [(f'scale{s:g}_corr{r:g}',s*np.array([[1.,r],[r,1.]]))
            for s in (.1,1.,10.) for r in (-.5,0.,.5)]


def prepare_initialization(base,budget,name,initial,X,y,root):
    path=root/f'aux_early{budget}_{name}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    source=pickle.load((base/f'aux_d1_early{budget}.pkl').open('rb'))
    setting=pickle.load((SOURCE/'fits_delta1.pkl').open('rb'))
    if np.array_equal(initial,np.eye(2)):
        scenarios=[dict(r) for r in source['scenarios'] if r['iteration'] in (1,3,10)]
    else:
        fit=source['fit'];S=fit['V'].T@fit['V']/len(y)
        e,psi=pilot(fit['theta'],y,setting['eta'],'logistic');rho=initial.copy();scenarios=[]
        import WERM.myocardial_approximate as implementation
        old=implementation.logistic_prox;implementation.logistic_prox=logistic_prox_fast
        try:
            for iteration in range(1,11):
                state=implementation.logistic_state(rho,setting['phi'],psi,len(y)/X.shape[1],sample_power=7,max_nfev=30)
                inv=np.linalg.pinv(state['lambda'],rcond=1e-6)
                rho=project_covariance(.75*rho+.25*project_covariance(inv.T@(S-state['kappa'])@inv))
                if iteration in (1,3,10):
                    check=implementation.logistic_state(rho,setting['phi'],psi,len(y)/X.shape[1],sample_power=8,max_nfev=60)
                    scenarios.append(dict(rho=rho.copy(),iteration=iteration,pilot=e,loss='logistic',
                        moment_residual=float(np.linalg.norm(check['lambda'].T@rho@check['lambda']+check['kappa']-S)),
                        state_residual=check['training_residual']))
        finally:implementation.logistic_prox=old
    for r in scenarios:
        r.update(auxiliary=f'early{budget}_init_{name}',auxiliary_budget=budget,
                 initialization_name=name,initial_rho=initial)
    atomic_save(path,scenarios);print('Initialization path ready:',budget,name,flush=True)
    return scenarios


def run_initializations(base):
    X,y,ages,locations=load_data();reference=reference_distribution()
    digest=hashlib.sha256(Path('tools/myocardial_initialization_sweep.py').read_bytes()).hexdigest()[:12]
    root=base/('initializations_'+digest);root.mkdir(exist_ok=True)
    setting=pickle.load((SOURCE/'fits_delta1.pkl').open('rb'))
    setting=dict(setting,fits={f'early{b}':pickle.load((base/f'aux_d1_early{b}.pkl').open('rb'))['fit'] for b in FIT_BUDGETS})
    print('Initialization sensitivity directory:',root,flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        prepared=Parallel(n_jobs=36)(delayed(prepare_initialization)(base,b,name,initial,X,y,root)
                    for b in (1,3,10,30) for name,initial in initialization_grid())
        scenarios=[r for group in prepared for r in group]
        # Reuse identity-start states from the broad sweep only after exact
        # rho/age-grid equality. Other starts get their own checkpoints.
        for r in scenarios:
            if not np.array_equal(r['initial_rho'],np.eye(2)):continue
            old_label=f"d1_early{r['auxiliary_budget']}_it{r['iteration']}_q8_b60"
            old=pickle.load((base/(old_label+'.pkl')).open('rb'))
            assert np.array_equal(old['scenario']['rho'],r['rho'])
            assert np.array_equal(old['eta'],setting['eta'])
            label=f"d1_{r['auxiliary']}_it{r['iteration']}_q8_b60"
            if not (root/(label+'.pkl')).exists():atomic_save(root/(label+'.pkl'),dict(old,label=label,scenario=r))
            for power,seed in ((9,42),(10,123)):
                src=base/(old_label+f'_states_q{power}_seed{seed}.pkl')
                dst=root/(label+f'_states_q{power}_seed{seed}.pkl')
                if src.exists() and not dst.exists():atomic_save(dst,pickle.load(src.open('rb')))
        records=Parallel(n_jobs=36)(delayed(evaluate_refined)(dict(setting,fits={}),r,X,y,ages,root,8,60) for r in scenarios)
        results=Parallel(n_jobs=36)(delayed(score_fit)(r,r['states'],setting,name,fit,X,y,ages,reference,root,'primary')
                    for r in records for name,fit in setting['fits'].items())
    case_map={r['label']:r for r in records}
    def tabulate(results):
        rows=[]
        for result in results:
            row={k:v for k,v in result.items() if k not in ('conditional','ages','log_likelihood')}
            scenario=case_map[row['case']]['scenario'];rho0=scenario['initial_rho']
            row.update(initialization=scenario['initialization_name'],initial_scale=rho0[0,0],initial_correlation=rho0[0,1]/rho0[0,0])
            rows.append(row)
        return pd.DataFrame(rows)
    summary=tabulate(results);summary.to_csv(root/'summary.csv',index=False)
    selected=[]
    for group,data in summary.groupby('group'):
        selected.extend([data.sort_values('TV_to_reference').head(2),
            data[data.TV_to_reference<=data.TV_to_reference.min()+.1].sort_values('P_1',ascending=False).head(1)])
    # Test neighboring starts for the best matched-budget candidate; do not
    # assume an isolated favorable initialization is robust.
    best=summary[summary.group=='logistic_matched'].sort_values('TV_to_reference').iloc[0]
    neighbors=summary[(summary.auxiliary_budget==best.auxiliary_budget)&
        (summary.posterior_budget==best.posterior_budget)&(summary.rho_updates==best.rho_updates)].copy()
    neighbors['distance']=abs(np.log10(neighbors.initial_scale/best.initial_scale))+abs(neighbors.initial_correlation-best.initial_correlation)
    selected.append(neighbors.sort_values('distance').head(3).drop(columns='distance'))
    selected.append(neighbors[(neighbors.initial_scale==1)&(neighbors.initial_correlation==0)].drop(columns='distance'))
    selected=pd.concat(selected).drop_duplicates('label');selected.to_csv(root/'selected_for_validation.csv',index=False)
    jobs=[(case_map[c],power,seed) for c in selected.case.unique() for power,seed in ((9,42),(10,123))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        states=Parallel(n_jobs=min(36,len(jobs)))(delayed(refined_states)(r,setting,len(y),X.shape[1],root,power,seed) for r,power,seed in jobs)
        results=Parallel(n_jobs=36)(delayed(score_fit)(r,state,setting,row.posterior_fit,setting['fits'][row.posterior_fit],
            X,y,ages,reference,root,f'q{power}_seed{seed}') for (r,power,seed),state in zip(jobs,states)
            for row in selected[selected.case==r['label']].itertuples())
    validation=tabulate(results);validation.to_csv(root/'validation.csv',index=False)
    ranked=validation.groupby(['case','posterior_fit','group','initialization'],as_index=False).agg(
        P1_min=('P_1','min'),P1_max=('P_1','max'),mean_TV=('TV_to_reference','mean'),max_TV=('TV_to_reference','max'),
        mass_above55_min=('mass_above55','min'),peak_min=('peak_age','min'),peak_max=('peak_age','max'),
        moment_residual=('moment_residual','max')).sort_values('mean_TV')
    ranked.to_csv(root/'ranked.csv',index=False)
    print('INITIALIZATION STUDY COMPLETE:',len(summary),'primary comparisons.',flush=True)
    print(ranked.to_string(index=False),flush=True);return root


if __name__=='__main__':
    import sys
    run_initializations(Path(sys.argv[1]))
