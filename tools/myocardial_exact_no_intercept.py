"""Preferred sensitivity: centered X, no intercept, identical logistic penalties.

Strengths are on the weighted mean-loss scale. These are finite-iteration
heuristics; fitting penalties are not inserted into the posterior likelihood.
All settings, including nonconverged fits and undesired outcomes, are retained.
"""
from tools.myocardial_exploration import *
from tools.myocardial_exact_penalty import matched_specifications, is_exact_match
from tools.myocardial_exploration_refinement import evaluate_refined
from tools.myocardial_independent_draw_check import independent_state


def prepare_matched(spec, setting, X, y, root):
    family, strength = spec
    name = f'{family}_{strength:g}'
    path = root/('aux_'+name+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    fit = setting['fits'].get(name)
    if fit is None:fit = fit_logistic(X, y, setting['phi'], family, strength)
    assert is_exact_match(family, strength, fit['penalty'], fit['strength'])
    scenarios = [s for s in setting['scenarios'] if s['auxiliary'] == name]
    if not scenarios:
        eta, psi = pilot(fit['theta'], y, setting['eta'], 'logistic')
        S = fit['V'].T@fit['V']/len(y)
        import WERM.myocardial_approximate as implementation
        from WERM.myocardial_numerics import logistic_prox_fast
        old = implementation.logistic_prox
        implementation.logistic_prox = logistic_prox_fast
        try:
            scenarios = [dict(r, auxiliary=name, pilot=eta, loss='logistic')
                for r in rho_path(S, setting['phi'], psi, len(y)/X.shape[1], 'logistic')]
        finally:implementation.logistic_prox = old
    result = (name, fit, scenarios)
    atomic_save(path, result)
    print('No-intercept matched auxiliary ready:', name, flush=True)
    return result


def score(rec, states, fit, setting, X, y, ages, reference, seed_label):
    ll = []
    for e, state in zip(setting['eta'], states):
        psi = np.zeros(len(y), int) if e < 0 else (np.arange(len(y)) >= e).astype(int)
        ll.append(gaussian_logistic_loglikelihood(fit['V'], y, rec['scenario']['rho'],
            state['lambda'], state['kappa'], psi, quadrature_order=160))
    ll = np.asarray(ll); joint = ll+np.log(setting['prior'])
    probs = np.exp(joint-logsumexp(joint)); curve = np.exp(ll[1:]-logsumexp(ll[1:]))
    return dict(case=rec['label'], penalty=fit['penalty'], strength=fit['strength'],
        seed_label=seed_label, P_1=probs[1:].sum(),
        mass_above55=curve[ages[setting['eta'][1:]]>55].sum(),
        peak_age=ages[setting['eta'][1:][np.argmax(curve)]], conditional=curve,
        TV_to_reference=.5*np.abs(curve-reference).sum(),
        moment_residual=rec['scenario']['moment_residual'],
        FP_residual_max=max(s['training_residual'] for s in states),
        fit_kkt_max=max(c['kkt'] for c in fit['checks']),
        fit_success=all(c['success'] for c in fit['checks']),
        iteration=rec['scenario']['iteration'], exact_match=True, fit_intercept=False)


def run(output):
    X, y, ages, locations = load_data()
    setting = pickle.load((output/'fits_delta1.pkl').open('rb'))
    root = output/'exact_matching_no_intercept'; root.mkdir(exist_ok=True)
    reference = pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl', 'rb'))
    a = np.asarray(reference['unique_ages']).ravel(); p = np.asarray(reference['posterior']).ravel()
    keep = np.isin(a, ages[locations]); a, p = a[keep], p[keep]; p = p/p.sum()
    mapping = dict(zip(a,p)); reference_curve = np.array([mapping[v] for v in ages[locations]])
    with parallel_config(backend='loky', inner_max_num_threads=1):
        prepared = Parallel(n_jobs=10)(delayed(prepare_matched)(spec, setting, X, y, root)
                                      for spec in matched_specifications())
        tasks = [(name, fit, scenario) for name, fit, scenarios in prepared for scenario in scenarios]
        records = Parallel(n_jobs=36)(delayed(evaluate_refined)(
            dict(setting, fits={name:fit}), scenario, X, y, ages, root, 9, 100)
            for name, fit, scenario in tasks)
    rows = []
    for rec, (name, fit, scenario) in zip(records, tasks):
        result = score(rec, rec['states'], fit, setting, X, y, ages, reference_curve, 'seed42_q9')
        atomic_save(root/(rec['label']+'_score.pkl'), result)
        rows.append({k:v for k,v in result.items() if k != 'conditional'})
    summary = pd.DataFrame(rows).sort_values(['penalty','strength','iteration'])
    summary.to_csv(root/'summary.csv', index=False)
    # Independent checks of the smallest strength at all iteration counts,
    # irrespective of outcome, plus closest eligible curve per family.
    smallest = summary[summary.strength == summary.strength.min()]
    eligible = summary[(summary.P_1>.5)&(summary.mass_above55>.5)]
    closest = eligible.sort_values('TV_to_reference').groupby('penalty').head(1)
    selected = pd.concat([smallest, closest]).drop_duplicates('case')
    by_case = {r['label']:(r,t) for r,t in zip(records,tasks)}
    checks = []
    for row in selected.itertuples():
        rec, (name, fit, scenario) = by_case[row.case]
        path = root/(row.case+'_independent.pkl')
        if path.exists():states = pickle.load(path.open('rb'))
        else:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                states = Parallel(n_jobs=36)(delayed(independent_state)(scenario['rho'], setting['phi'],
                    e, len(y), len(y)/X.shape[1]) for e in setting['eta'])
            atomic_save(path, states)
        result = score(rec, states, fit, setting, X, y, ages, reference_curve, 'seed123_q10')
        result['primary_P_1'] = row.P_1
        atomic_save(root/(row.case+'_independent_score.pkl'), result)
        checks.append({k:v for k,v in result.items() if k != 'conditional'})
        pd.DataFrame(checks).to_csv(root/'independent_validation.csv', index=False)
    print('No-intercept exact-match exploration complete:',len(summary),'primary settings.',flush=True)
    return summary, pd.DataFrame(checks)


if __name__ == '__main__':
    import sys, jax
    jax.config.update('jax_enable_x64', True)
    run(Path(sys.argv[1]))
