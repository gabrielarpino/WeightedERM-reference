"""Read-only experiment aggregation: rank saved, independently checked curves."""
from pathlib import Path
import json,pickle,hashlib
import numpy as np
import pandas as pd

BASE=Path('results/myocardial_finite_iteration_443c17cf2df6')
OLD=Path('results/myocardial_exploration_4026d0e775d2')
OFFSET=Path('results/myocardial_intercept_exploration_f53f59063391')


def curve_metrics(ages,curve,reference):
    ages=np.asarray(ages,float);curve=np.asarray(curve,float)
    assert len(ages)==len(curve) and len(set(ages))==len(ages)
    assert np.isfinite(curve).all() and np.min(curve)>=0
    np.testing.assert_allclose(curve.sum(),1.,atol=1e-8)
    mapping=dict(zip(ages,curve));support=set(reference)|set(mapping)
    return dict(TV=.5*sum(abs(reference.get(a,0.)-mapping.get(a,0.)) for a in support),
        above55=float(curve[ages>55].sum()),peak=float(ages[np.argmax(curve)]),
        max_probability=float(curve.max()))


def rank_candidates(rows):
    table=pd.DataFrame(rows).sort_values('priority',ascending=False,kind='stable')
    table=table.drop_duplicates('configuration').sort_values(['mean_TV','max_TV','configuration'],kind='stable').reset_index(drop=True)
    table.insert(0,'rank',np.arange(1,len(table)+1));return table


def load_candidates():
    from tools.myocardial_finite_iteration_exploration import load_data,reference_distribution
    reference=reference_distribution();_,_,ages,locations=load_data();all_ages=ages[locations]
    rows=[];curves={}
    def add(first,second,meta,first_path,second_path):
        a=np.asarray(first.get('ages',all_ages),float)
        b=np.asarray(second.get('ages',all_ages),float)
        np.testing.assert_array_equal(a,b)
        p=np.asarray(first['conditional']);q=np.asarray(second['conditional'])
        m=curve_metrics(a,p,reference);n=curve_metrics(a,q,reference)
        probabilities=[float(first['P_1']),float(second['P_1'])]
        assert min(probabilities)>=-1e-8 and max(probabilities)<=1+1e-8
        probabilities=np.clip(probabilities,0,1)
        key=json.dumps({k:meta[k] for k in ('auxiliary','posterior','intercept','Delta','rho_updates','initialization')},sort_keys=True)
        ident=hashlib.sha256((str(first_path)+'|'+str(second_path)).encode()).hexdigest()[:12]
        row=dict(meta,id=ident,configuration=key,P1_min=min(probabilities),P1_max=max(probabilities),
            mean_TV=(m['TV']+n['TV'])/2,max_TV=max(m['TV'],n['TV']),
            numerical_curve_TV=float(.5*np.abs(p-q).sum()),
            mass_above55_min=min(m['above55'],n['above55']),
            peak_min=min(m['peak'],n['peak']),peak_max=max(m['peak'],n['peak']),
            primary_P1=probabilities[0],check_P1=probabilities[1],
            moment_residual=float(first.get('moment_residual',meta.get('moment_residual',np.nan))),
            primary_checkpoint=str(first_path),check_checkpoint=str(second_path))
        rows.append(row);curves[ident]=dict(ages=a,primary=p,check=q)
    stage_dirs={'broad':BASE,'focused':BASE/'focus_354ed8f1e782',
        'initializations':BASE/'initializations_08d005a910b5',
        'fully_solved_matched_L2':BASE/'matched_l2_full_affdef22fb6c'}
    table=pd.read_csv(BASE/'combined_with_full_l2.csv')
    for row in table.itertuples():
        folder=stage_dirs[row.stage]
        if row.stage=='fully_solved_matched_L2':
            paths=[folder/(row.case+'_'+tag+'_score.pkl') for tag in ('q9_seed42','q10_seed123')]
            auxiliary=posterior=f'Logistic L2 {row.strength:.12g} (full solve)'
        else:
            paths=[folder/(row.case+'_'+row.posterior_fit+'_'+tag+'.pkl') for tag in ('q9_seed42','q10_seed123')]
            auxiliary='LS' if row.group=='least_squares' else f'Logistic {"L-BFGS" if str(row.auxiliary).startswith("lbfgs") else "FISTA"} {int(row.auxiliary_budget)} iterations'
            posterior=f'Logistic {"L-BFGS" if row.posterior_fit.startswith("lbfgs") else "FISTA"} {int(row.posterior_budget)} iterations'
        init=row.initialization if pd.notna(row.initialization) else 'identity'
        if init=='scale1_corr0':init='identity'
        add(*[pickle.load(p.open('rb')) for p in paths],dict(
            auxiliary=auxiliary,posterior=posterior,intercept=False,Delta=int(row.Delta),
            rho_updates=int(row.rho_updates),initialization=init,stage=row.stage,priority=3,
            preference_note='No intercept; LS auxiliary' if row.group=='least_squares' else ('No intercept; matched' if row.group in ('logistic_matched','fully_solved_matched_L2') else 'No intercept; different fitting budgets')),*paths)
    # Earlier LS-auxiliary penalty sweeps, retaining every independently checked pair.
    folder=OLD/'denser_penalty_grid'
    for second_path in sorted(folder.glob('*_independent.pkl')):
        first_path=second_path.with_name(second_path.name.replace('_independent.pkl','.pkl'))
        first=pickle.load(first_path.open('rb'));second=pickle.load(second_path.open('rb'))
        case=second['case'];base=pickle.load((OLD/(case+'.pkl')).open('rb'))
        assert base['scenario']['auxiliary']=='least_squares'
        add(first,second,dict(auxiliary='LS',posterior=f'Logistic {first["penalty"].upper()} {first["strength"]:.12g} (full solve)',
            intercept=False,Delta=1,rho_updates=int(base['scenario']['iteration']),initialization='identity',
            stage='earlier_LS_penalty',priority=2,preference_note='No intercept; LS auxiliary',
            moment_residual=base['scenario']['moment_residual']),first_path,second_path)
    # Earlier intercept-aware fits: explicitly flag preference violations, never
    # silently reclassify them as no-intercept or matched-penalty candidates.
    folder=OFFSET/'penalty_resolution'
    for first_path in sorted(folder.glob('*_seed42_q9.pkl')):
        second_path=first_path.with_name(first_path.name.replace('_seed42_q9.pkl','_seed123_q10.pkl'))
        if not second_path.exists():continue
        first=pickle.load(first_path.open('rb'));second=pickle.load(second_path.open('rb'))
        rec=pickle.load((OFFSET/(first['case']+'.pkl')).open('rb'));aux=rec['scenario']['auxiliary']
        if aux=='least_squares':auxiliary='LS';note='Uses intercept'
        else:
            family,strength=aux.split('_',1);strength=float(strength)
            auxiliary=f'Logistic {family.upper()} {strength:.12g} (full solve)'
            note='Uses intercept; '+('matched penalties' if family==first['penalty'] and strength==first['strength'] else 'mismatched penalties')
        add(first,second,dict(auxiliary=auxiliary,posterior=f'Logistic {first["penalty"].upper()} {first["strength"]:.12g} (full solve)',
            intercept=True,Delta=1,rho_updates=int(rec['scenario']['iteration']),initialization='identity',
            stage='earlier_intercept',priority=2,preference_note=note),first_path,second_path)
    # Earlier exact-match checks (not all will be competitive on shape).
    folder=OLD/'exact_matching_no_intercept'
    for second_path in sorted(folder.glob('*_independent_score.pkl')):
        first_path=second_path.with_name(second_path.name.replace('_independent_score.pkl','_score.pkl'))
        first=pickle.load(first_path.open('rb'));second=pickle.load(second_path.open('rb'))
        name=f'Logistic {first["penalty"].upper()} {first["strength"]:.12g} (full solve)'
        add(first,second,dict(auxiliary=name,posterior=name,intercept=False,Delta=1,rho_updates=int(first['iteration']),
            initialization='identity',stage='earlier_exact_match',priority=1,preference_note='No intercept; matched penalties'),first_path,second_path)
    ranked=rank_candidates(rows)
    return ranked,curves,reference


def summary_table(ranked):
    result=ranked[['rank','auxiliary','posterior','intercept','Delta','rho_updates']].copy()
    result['P(1) numerical range']=[f'{lo:.3f}–{hi:.3f}' for lo,hi in zip(ranked.P1_min,ranked.P1_max)]
    result['Conditional mode (age)']=[f'{lo:g}' if lo==hi else f'{lo:g}–{hi:g}' for lo,hi in zip(ranked.peak_min,ranked.peak_max)]
    result['Mean TV']=ranked.mean_TV.round(3)
    result['Numerical curve TV']=ranked.numerical_curve_TV.round(3)
    result['Min. mass above 55']=ranked.mass_above55_min.round(3)
    result['Preference notes']=ranked.preference_note
    return result
