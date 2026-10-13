"""Denser posterior-penalty grid reusing already computed state equations."""
from tools.myocardial_exploration import *

SWEEP_FITS=[('l2',float(x)) for x in np.logspace(-9,-3,13)]+[('l1',float(x)) for x in np.logspace(-7,-3,9)]


def sweep_fit(spec,X,y,phi,root):
    kind,strength=spec;name=f'{kind}_{strength:.8g}'
    path=root/('fit_'+name+'.pkl')
    if path.exists():return name,pickle.load(path.open('rb'))
    try:fit=fit_logistic(X,y,phi,kind,strength)
    except Exception as error:fit=dict(error=repr(error))
    atomic_save(path,fit);return name,fit


def sweep_case(rec,setting,name,fit,X,y,ages,root):
    label=rec['label']+'_'+name;path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    if 'error' in fit:return dict(label=label,error=fit['error'])
    values=[]
    for e,state in zip(rec['eta'],rec['states']):
        psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
        values.append(gaussian_logistic_loglikelihood(fit['V'],y,rec['scenario']['rho'],
                       state['lambda'],state['kappa'],psi,quadrature_order=160))
    values=np.asarray(values);joint=values+np.log(setting['prior'])
    probabilities=np.exp(joint-logsumexp(joint));curve=np.exp(values[1:]-logsumexp(values[1:]))
    result=dict(label=label,base_setting=rec['label'],posterior_fit=name,
        penalty=fit['penalty'],strength=fit['strength'],P_1=float(probabilities[1:].sum()),
        conditional=curve,mass_above55=float(curve[ages[rec['eta'][1:]]>55].sum()),
        peak_age=float(ages[rec['eta'][1:][np.argmax(curve)]]),
        log_Bayes_factor=float(logsumexp(values[1:])-np.log(len(curve))-values[0]),
        max_fit_kkt=max(c['kkt'] for c in fit['checks']),FP_residual_max=max(rec['residuals']))
    atomic_save(path,result);return result


def penalty_sweep(output):
    X,y,ages,locations=load_data();setting=pickle.load((output/'fits_delta1.pkl').open('rb'))
    root=output/'denser_penalty_grid';root.mkdir(exist_ok=True)
    # All five LS covariance estimates and moderate-ridge auxiliary estimates
    # at 3/30 updates. Selection is by specification, not posterior outcomes.
    paths=[p for p in output.glob('d1*_q7_b40.pkl') if
           p.name.startswith('d1_least_squares_') or
           p.name in ['d1_l2_1e-06_it3_q7_b40.pkl','d1_l2_1e-06_it30_q7_b40.pkl']]
    fine_paths=[p.with_name(p.name.replace('_q7_b40','_q9_b100')) for p in paths]
    assert all(p.exists() for p in fine_paths), 'Run the declared numerical refinements first.'
    records=[pickle.load(p.open('rb')) for p in fine_paths]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        fits=Parallel(n_jobs=22)(delayed(sweep_fit)(spec,X,y,setting['phi'],root) for spec in SWEEP_FITS)
        results=Parallel(n_jobs=36)(delayed(sweep_case)(r,setting,name,fit,X,y,ages,root)
                    for r in records for name,fit in fits)
    pd.DataFrame([{k:v for k,v in r.items() if k!='conditional'} for r in results]).to_csv(root/'summary.csv',index=False)
    print('Denser penalty sweep complete:',len(results),'comparisons.',flush=True)
    return results


if __name__=='__main__':
    import sys
    penalty_sweep(Path(sys.argv[1]))
