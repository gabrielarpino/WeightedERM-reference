"""Bounded, outcome-disclosed penalty refinement of the two leading offsets."""
from tools.myocardial_intercept_exploration import *
from tools.myocardial_penalty_sweep import SWEEP_FITS
from WERM.myocardial_intercept import intercept_loglikelihood

OFFSET_SWEEP=SWEEP_FITS+[('l2',v) for v in (6e-7,8e-7,1.2e-6,1.5e-6,2e-6)]


def offset_sweep_fit(spec,X,y,phi,root):
    kind,strength=spec;name=f'{kind}_{strength:.8g}';path=root/('fit_'+name+'.pkl')
    if path.exists():return name,pickle.load(path.open('rb'))
    fit=fit_with_intercept(X,y,phi,kind,strength)
    atomic_save(path,fit);return name,fit


def offset_sweep_score(rec,states,setting,name,fit,X,y,ages,root,seed_label):
    label=rec['label']+'_'+name+'_'+seed_label;path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    values=[]
    for e,state in zip(setting['eta'],states):
        psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
        values.append(intercept_loglikelihood(fit['V'],y,rec['scenario']['rho'],state,psi))
    ll=np.asarray(values);joint=ll+np.log(setting['prior']);p=np.exp(joint-logsumexp(joint));curve=np.exp(ll[1:]-logsumexp(ll[1:]))
    record=dict(label=label,case=rec['label'],posterior_fit=name,seed_label=seed_label,
        penalty=fit['penalty'],strength=fit['strength'],P_1=p[1:].sum(),
        mass_above55=curve[ages[setting['eta'][1:]]>55].sum(),conditional=curve,
        peak_age=ages[setting['eta'][1:][np.argmax(curve)]],moment_residual=rec['scenario']['moment_residual'],
        max_fit_kkt=max(c['kkt'] for c in fit['checks']),FP_residual_max=max(s['training_residual'] for s in states))
    atomic_save(path,record);return record


def intercept_penalty_sweep(output):
    X,y,ages,locations=load_data();setting=pickle.load((output/'offset_fits.pkl').open('rb'))
    root=output/'penalty_resolution';root.mkdir(exist_ok=True)
    cases=[]
    for iteration in (10,30):
        name=f'offset_l1_0.0001_it{iteration}_q9_b100'
        rec=pickle.load((output/(name+'.pkl')).open('rb'))
        independent=pickle.load((output/(name+'_independent_seed123_q10.pkl')).open('rb'))
        cases.extend([(rec,rec['states'],'seed42_q9'),(rec,independent,'seed123_q10')])
    with parallel_config(backend='loky',inner_max_num_threads=1):
        fits=Parallel(n_jobs=27)(delayed(offset_sweep_fit)(spec,X,y,setting['phi'],root) for spec in OFFSET_SWEEP)
        results=Parallel(n_jobs=36)(delayed(offset_sweep_score)(rec,states,setting,name,fit,X,y,ages,root,seed_label)
                    for rec,states,seed_label in cases for name,fit in fits)
    pd.DataFrame([{k:v for k,v in r.items() if k!='conditional'} for r in results]).to_csv(root/'summary.csv',index=False)
    print('Final bounded intercept penalty refinement complete:',len(results),'comparisons.',flush=True)
    return results


if __name__=='__main__':
    import sys
    intercept_penalty_sweep(Path(sys.argv[1]))
