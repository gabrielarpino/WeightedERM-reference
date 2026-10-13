"""Apply the user's LS-first / matched-penalty-family preference."""
from tools.myocardial_intercept_penalty_sweep import *
from tools.myocardial_independent_draw_check import independent_state


def preferred_family(auxiliary,penalty):
    if auxiliary=='least_squares':return 'Least-squares rho fit'
    if auxiliary.startswith('l1_') and penalty=='l1':return 'L1 / L1'
    if auxiliary.startswith('l2_') and penalty=='l2':return 'L2 / L2'
    return None


def preference_search(output,reference_vector):
    X,y,ages,locations=load_data();setting=pickle.load((output/'offset_fits.pkl').open('rb'))
    root=output/'penalty_resolution'
    fits=[(p.stem[4:],pickle.load(p.open('rb'))) for p in sorted(root.glob('fit_*.pkl'))]
    records=[pickle.load(p.open('rb')) for p in sorted(output.glob('offset_*_q9_b100.pkl'))]
    tasks=[(rec,name,fit) for rec in records for name,fit in fits
           if preferred_family(rec['scenario']['auxiliary'],fit['penalty'])]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=36)(delayed(offset_sweep_score)(rec,rec['states'],setting,name,fit,X,y,ages,root,'seed42_q9')
                                   for rec,name,fit in tasks)
    rows=[]
    case_map={r['label']:r for r in records};fit_map=dict(fits)
    for result in results:
        rec=case_map[result['case']]
        row={k:v for k,v in result.items() if k!='conditional'}
        row['family']=preferred_family(rec['scenario']['auxiliary'],result['penalty'])
        row['TV_to_reference']=.5*np.abs(result['conditional']-reference_vector).sum()
        row['meets_targets']=(row['P_1']>.5 and row['mass_above55']>.5)
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'loss_preference_grid.csv',index=False)
    # Validate the closest two eligible settings in each family. The same
    # independent seed is used; no searching for a favorable random seed.
    selected=summary[summary.meets_targets].sort_values('TV_to_reference').groupby('family',sort=False).head(2)
    validation=[]
    for case in selected.case.unique():
        rec=case_map[case];path=output/(case+'_independent_seed123_q10.pkl')
        if path.exists():states=pickle.load(path.open('rb'))
        else:
            with parallel_config(backend='loky',inner_max_num_threads=1):
                states=Parallel(n_jobs=18)(delayed(independent_state)(rec['scenario']['rho'],setting['phi'],e,len(y),len(y)/X.shape[1],setting['rate'])
                                         for e in setting['eta'])
            atomic_save(path,states)
        for row in selected[selected.case==case].itertuples():
            check=offset_sweep_score(rec,states,setting,row.posterior_fit,fit_map[row.posterior_fit],X,y,ages,root,'seed123_q10')
            original=next(r for r in results if r['case']==case and r['posterior_fit']==row.posterior_fit)
            tv=.5*np.abs(check['conditional']-reference_vector).sum()
            validation.append(dict(case=case,posterior_fit=row.posterior_fit,family=row.family,
                primary_P1=row.P_1,check_P1=check['P_1'],mass_above55_min=min(row.mass_above55,check['mass_above55']),
                primary_TV=row.TV_to_reference,check_TV=tv,mean_TV=(row.TV_to_reference+tv)/2,
                TV_between_draws=.5*np.abs(check['conditional']-original['conditional']).sum(),
                primary_peak=row.peak_age,check_peak=check['peak_age'],moment_residual=row.moment_residual,
                meets_both_in_both=bool(min(row.P_1,check['P_1'])>.5 and min(row.mass_above55,check['mass_above55'])>.5)))
    validated=pd.DataFrame(validation)
    validated.to_csv(output/'loss_preference_validation.csv',index=False)
    print('LS-first/matched-family exploration complete:',len(summary),'primary comparisons.',flush=True)
    return summary,validated


if __name__=='__main__':
    import sys,jax
    jax.config.update('jax_enable_x64',True)
    X,y,ages,locations=load_data()
    reference=pickle.load(open('results/myocardial_infarction_posterior_exact_one_chgpt.pkl','rb'))
    a=np.asarray(reference['unique_ages']).ravel();p=np.asarray(reference['posterior']).ravel()
    keep=np.isin(a,ages[locations]);a=a[keep];p=p[keep];p/=p.sum();mapping=dict(zip(a,p))
    preference_search(Path(sys.argv[1]),np.array([mapping[x] for x in ages[locations]]))
