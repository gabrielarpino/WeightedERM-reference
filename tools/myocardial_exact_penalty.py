"""Exact family-and-strength matching; retain all outcomes and diagnostics.

Both stages reuse the *same* coefficient fit, hence also the same weighted
mean-loss normalization and unpenalized-intercept convention. This is a
heuristic sensitivity experiment, not calibrated model selection.
"""


def matched_specifications():
    return [(family, strength) for family in ('l1', 'l2')
            for strength in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)]


def is_exact_match(aux_family, aux_strength, post_family, post_strength):
    return aux_family == post_family and aux_strength == post_strength


def prepare_spec(spec, setting, X, y, output):
    from tools.myocardial_intercept_penalty_sweep import (
        offset_sweep_fit, offset_rho_path, pilot, atomic_save, pickle)
    family, strength = spec
    name, fit = offset_sweep_fit(spec, X, y, setting['phi'], output/'penalty_resolution')
    path = output/'exact_matching'/('aux_'+name+'.pkl')
    if path.exists():
        return pickle.load(path.open('rb'))
    assert is_exact_match(family, strength, fit['penalty'], fit['strength'])
    # Reuse existing identical auxiliary trajectories where available.
    old = [s for s in setting['scenarios'] if s['auxiliary'] == name]
    if old:
        scenarios = old
    else:
        eta, psi = pilot(fit['theta'], y, setting['eta'], 'logistic')
        S = fit['V'].T @ fit['V'] / len(y)
        scenarios = [dict(r, auxiliary=name, pilot=eta, loss='logistic')
                     for r in offset_rho_path(S, setting['phi'], psi,
                                             len(y)/X.shape[1], setting['rate'], 'logistic')]
    result = (name, fit, scenarios)
    atomic_save(path, result)
    print('Exact-match auxiliary ready:', name, flush=True)
    return result


def run(output):
    from tools.myocardial_intercept_penalty_sweep import (
        load_data, pickle, np, pd, Parallel, delayed, parallel_config,
        offset_case, offset_sweep_score, atomic_save)
    from tools.myocardial_independent_draw_check import independent_state
    X, y, ages, locations = load_data()
    setting = pickle.load((output/'offset_fits.pkl').open('rb'))
    root = output/'exact_matching'
    root.mkdir(exist_ok=True)
    reference = pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl', 'rb'))
    a = np.asarray(reference['unique_ages']).ravel()
    p = np.asarray(reference['posterior']).ravel()
    keep = np.isin(a, ages[locations]); a, p = a[keep], p[keep]; p = p/p.sum()
    mapping = dict(zip(a, p)); reference_curve = np.array([mapping[v] for v in ages[locations]])
    with parallel_config(backend='loky', inner_max_num_threads=1):
        prepared = Parallel(n_jobs=10)(delayed(prepare_spec)(s, setting, X, y, output)
                                      for s in matched_specifications())
        tasks = []
        for name, fit, scenarios in prepared:
            for scenario in scenarios:
                local = dict(setting, fits={name: fit})
                tasks.append((name, fit, scenario, local))
        # Each task evaluates only the identical-strength posterior fit.
        records = Parallel(n_jobs=36)(delayed(offset_case)(local, scenario, X, y, ages, root)
                    for name, fit, scenario, local in tasks)
    rows = []
    for rec, (name, fit, scenario, local) in zip(records, tasks):
        result = offset_sweep_score(rec, rec['states'], setting, name, fit,
                                   X, y, ages, root, 'seed42_q9')
        rows.append(dict({k: v for k, v in result.items() if k != 'conditional'},
                         TV_to_reference=.5*np.abs(result['conditional']-reference_curve).sum(),
                         iteration=scenario['iteration'], exact_match=True))
    summary = pd.DataFrame(rows).sort_values(['penalty', 'strength', 'iteration'])
    summary.to_csv(root/'summary.csv', index=False)
    # Check the smallest penalty at every saved iteration regardless of outcome.
    # Also check the closest eligible curve in each family, explicitly exploratory.
    smallest = summary[summary.strength == summary.strength.min()]
    eligible = summary[(summary.P_1 > .5) & (summary.mass_above55 > .5)]
    closest = eligible.sort_values('TV_to_reference').groupby('penalty').head(1)
    selected = pd.concat([smallest, closest]).drop_duplicates('case')
    by_case = {r['label']: (r, t) for r, t in zip(records, tasks)}
    validation = []
    for row in selected.itertuples():
        rec, (name, fit, scenario, local) = by_case[row.case]
        path = root/(row.case+'_independent_seed123_q10.pkl')
        if path.exists():
            states = pickle.load(path.open('rb'))
        else:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                states = Parallel(n_jobs=36)(delayed(independent_state)(
                    scenario['rho'], setting['phi'], e, len(y), len(y)/X.shape[1], setting['rate'])
                    for e in setting['eta'])
            atomic_save(path, states)
        check = offset_sweep_score(rec, states, setting, name, fit, X, y, ages, root, 'seed123_q10')
        validation.append(dict({k: v for k, v in check.items() if k != 'conditional'},
            primary_P_1=row.P_1, primary_TV=row.TV_to_reference,
            TV_to_reference=.5*np.abs(check['conditional']-reference_curve).sum(),
            exact_match=True))
        pd.DataFrame(validation).to_csv(root/'independent_validation.csv', index=False)
    print('Exact matching complete:', len(summary), 'primary comparisons.', flush=True)
    return summary, pd.DataFrame(validation)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    import jax
    jax.config.update('jax_enable_x64', True)
    run(Path(sys.argv[1]))
