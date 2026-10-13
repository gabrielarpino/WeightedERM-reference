"""Matched unpenalized solver sensitivity, with an unchanged posterior model."""
from tools.myocardial_finite_iteration_exploration import *
from WERM.myocardial_no_intercept_solvers import fit_column

SWEEP_DELTAS = (1, 5, 10)
RHO_UPDATES = (1, 3, 6, 10, 20, 30)


def configurations():
    grid = []
    for method in ('fista', 'gd'):
        grid += [(method, b, s, 42) for b in (1, 3, 8, 16, 30) for s in (.1, 1.)]
    grid += [('lbfgs', b, 1., 42) for b in (1, 2, 3, 5, 8, 12, 20)]
    grid += [('sgd', b, s, seed) for b in (1, 3, 10) for s in (.1, 1.) for seed in (42, 123)]
    grid += [('saga', b, 1., seed) for b in (1, 3, 10) for seed in (42, 123)]
    return grid


def fit_name(config):
    method, budget, scale, seed = config
    return f'{method}_b{budget}_step{scale:g}_seed{seed}'


def setting_for(d, y, locations):
    eta, prior, phi = configuration_prior(len(y), locations[(locations >= d)&(locations <= len(y)-d)])
    return dict(delta_min=d, eta=eta, prior=prior, phi=phi, fits={})


def prepare_case(d, config, X, y, locations, root):
    name = fit_name(config); path = root/f'aux_d{d}_{name}.pkl'
    if path.exists(): return pickle.load(path.open('rb'))
    try:
        setting = setting_for(d, y, locations)
        columns = [fit_column(X, y, w, *config) for w in setting['phi']]
        B = np.column_stack([c['coef'] for c in columns]); theta = X@B
        V, b = empirical_adjustment(theta, y, setting['phi'], len(y)/X.shape[1], loss='logistic')
        if not np.isfinite(V).all(): raise FloatingPointError('nonfinite adjustment')
        fit = dict(theta=theta, V=V, b=b, coef=B, budget=config[1],
                   checks=columns, penalty='none', strength=0.)
        e, psi = pilot(theta, y, setting['eta'], 'logistic')
        S = V.T@V/len(y); rho = np.eye(2); scenarios = []
        import WERM.myocardial_approximate as impl
        old = impl.logistic_prox; impl.logistic_prox = logistic_prox_fast
        try:
            for it in range(1, max(RHO_UPDATES)+1):
                state = impl.logistic_state(rho, setting['phi'], psi, len(y)/X.shape[1], sample_power=7, max_nfev=30)
                inv = np.linalg.pinv(state['lambda'], rcond=1e-6)
                rho = project_covariance(.75*rho+.25*project_covariance(inv.T@(S-state['kappa'])@inv))
                if it in RHO_UPDATES:
                    check = impl.logistic_state(rho, setting['phi'], psi, len(y)/X.shape[1], sample_power=8, max_nfev=60)
                    scenarios.append(dict(rho=rho.copy(), iteration=it, auxiliary=name,
                        auxiliary_budget=config[1], pilot=e, loss='logistic',
                        moment_residual=float(np.linalg.norm(check['lambda'].T@rho@check['lambda']+check['kappa']-S)),
                        state_residual=check['training_residual']))
        finally: impl.logistic_prox = old
        result = dict(setting=setting, fit=fit, scenarios=scenarios, config=config)
    except Exception:
        result = dict(error=traceback.format_exc(), delta=d, config=config)
    atomic_save(path, result)
    print('Auxiliary ready:', d, name, 'error' if 'error' in result else 'ok', flush=True)
    return result


def evaluate_case(prepared, scenario, X, y, ages, reference, root, power=8, seed=42, budget=60):
    setting = prepared['setting']; name = scenario['auxiliary']
    case = f"d{setting['delta_min']}_{name}_rho{scenario['iteration']}"
    tag = f'q{power}_seed{seed}_b{budget}'
    path = root/f'{case}_{tag}_score.pkl'
    if path.exists(): return pickle.load(path.open('rb'))
    partial = root/f'{case}_{tag}_partial.pkl'
    states = pickle.load(partial.open('rb')) if partial.exists() else []
    import WERM.myocardial_approximate as impl
    old = impl.logistic_prox; impl.logistic_prox = logistic_prox_fast
    try:
        for e in setting['eta'][len(states):]:
            psi = np.zeros(len(y), int) if e < 0 else (np.arange(len(y)) >= e).astype(int)
            states.append(impl.logistic_state(scenario['rho'], setting['phi'], psi,
                len(y)/X.shape[1], sample_power=power, max_nfev=budget, seed=seed))
            if len(states)%8 == 0: atomic_save(partial, states)
        atomic_save(partial, states)
        rec = dict(label=case, scenario=scenario)
        row = score_fit(rec, states, setting, name, prepared['fit'], X, y, ages, reference, root, tag)
        row.update(solver=prepared['config'][0], step_scale=prepared['config'][2],
            fit_seed=prepared['config'][3], intercept=False, penalty='none',
            checkpoint=str(path), pilot_index=scenario['pilot'])
        # Stable log-odds/P(0) diagnostics distinguish rounding to 1 from exact certainty.
        logmass = row['log_likelihood'] + np.log(setting['prior'])
        row['log_odds_1_vs_0'] = float(logsumexp(logmass[1:])-logmass[0])
        row['P_0'] = float(expit(-row['log_odds_1_vs_0']))
    except Exception:
        row = dict(error=traceback.format_exc(), case=case, tag=tag)
    finally: impl.logistic_prox = old
    atomic_save(path, row)
    return row


def table(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k not in ('conditional','ages','log_likelihood')} for r in rows if 'error' not in r])


def select_checks(frame):
    groups = [frame] + [g for _,g in frame.groupby('solver')]
    selected = []
    for group in groups:
        selected.append(group.nsmallest(3, 'TV_to_reference'))
        middle = group[group.P_1.between(.5, .99, inclusive='left')]
        selected.append(middle.nsmallest(3, 'TV_to_reference'))
    return pd.concat(selected).drop_duplicates('case')


def run(smoke=False):
    X,y,ages,locations = load_data(); reference = reference_distribution()
    spec = dict(configurations=configurations(), deltas=SWEEP_DELTAS, rho_updates=RHO_UPDATES,
        rho_start='identity', damping=.25, eigenvalue_bounds=[1e-4,100], intercept=False,
        penalty='none', match='same solver, budget, step and random seed in both stages; same fit reused',
        count_prior=[.5,.5], conditional_prior='uniform admissible distinct-age splits',
        primary=[8,42,60], validation=[[9,42,100],[10,123,100]],
        selection='shape leaders overall and by solver; also 0.5 <= P1 < 0.99 shape leaders; post hoc',
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('tools/myocardial_no_intercept_sweep.py','WERM/myocardial_no_intercept_solvers.py',
             'WERM/myocardial_approximate.py','WERM/signal_strength.py')})
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    root = Path('results')/('myocardial_no_intercept_solvers_'+digest); root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec, indent=2))
    if smoke:
        setting = setting_for(1,y,locations)
        for method in ('fista','gd','lbfgs','sgd','saga'):
            cols = [fit_column(X,y,w,method,3) for w in setting['phi']]
            theta=X@np.column_stack([c['coef'] for c in cols])
            V,b=empirical_adjustment(theta,y,setting['phi'],len(y)/X.shape[1],loss='logistic')
            assert np.isfinite(V).all()
        tiny=dict(setting,eta=setting['eta'][:2],prior=np.array([.5,.5]))
        prepared=dict(setting=tiny,fit=dict(theta=theta,V=V,b=b,budget=3,checks=cols),config=('saga',3,1.,42))
        scenario=dict(rho=np.eye(2),iteration=1,auxiliary='smoke',auxiliary_budget=3,moment_residual=0.,pilot=-1)
        checked=evaluate_case(prepared,scenario,X,y,ages,reference,root,5,42,3)
        assert 'error' not in checked, checked
        assert np.isfinite(checked['log_odds_1_vs_0'])
        print('All five real-data fit adapters and two-location posterior smoke passed.',flush=True)
        return root
    print('Starting',len(configurations())*len(SWEEP_DELTAS)*len(RHO_UPDATES),'matched comparisons:',root,flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        prepared=Parallel(n_jobs=36)(delayed(prepare_case)(d,c,X,y,locations,root) for d in SWEEP_DELTAS for c in configurations())
        errors=[r for r in prepared if 'error' in r]
        jobs=[(p,s) for p in prepared if 'error' not in p for s in p['scenarios']]
        rows=Parallel(n_jobs=36)(delayed(evaluate_case)(p,s,X,y,ages,reference,root) for p,s in jobs)
        errors += [r for r in rows if 'error' in r]
        summary=table(rows); summary.to_csv(root/'summary.csv',index=False)
        if summary.empty: raise RuntimeError('No successful primary results; inspect checkpoints')
        selected=select_checks(summary); selected.to_csv(root/'selected_for_validation.csv',index=False)
        wanted=set(selected.case)
        follow=[(p,s) for p,s in jobs if f"d{p['setting']['delta_min']}_{s['auxiliary']}_rho{s['iteration']}" in wanted]
        checked=Parallel(n_jobs=36)(delayed(evaluate_case)(p,s,X,y,ages,reference,root,power,seed,100)
            for p,s in follow for power,seed in ((9,42),(10,123)))
    errors += [r for r in checked if 'error' in r]
    (root/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    validation=table(checked); validation.to_csv(root/'validation.csv',index=False)
    ranked=validation.groupby('case',as_index=False).agg(
        checks=('P_1','size'), solver=('solver','first'),fit_budget=('posterior_budget','first'),
        step_scale=('step_scale','first'),fit_seed=('fit_seed','first'),Delta=('Delta','first'),
        rho_updates=('rho_updates','first'),P1_min=('P_1','min'),P1_max=('P_1','max'),
        mean_TV=('TV_to_reference','mean'),max_TV=('TV_to_reference','max'),
        peak_min=('peak_age','min'),peak_max=('peak_age','max'),
        mass_above55_min=('mass_above55','min'),moment_residual=('moment_residual','max'),
        FP_residual_max=('FP_residual_max','max'))
    ranked=ranked[ranked.checks==2].sort_values('mean_TV')
    ranked.to_csv(root/'ranked.csv',index=False)
    ranked.head(10).to_csv(root/'top_ten_shape.csv',index=False)
    ranked[(ranked.P1_min>=.5)&(ranked.P1_max<.99)].head(10).to_csv(root/'top_ten_nonextreme.csv',index=False)
    print('NO-INTERCEPT SWEEP COMPLETE; errors:',len(errors),flush=True)
    return root


if __name__ == '__main__':
    import sys
    run(smoke='--smoke' in sys.argv)
