"""Outcome-disclosed follow-up using independent QMC draws, not a new search."""
from tools.myocardial_exploration import *


def independent_state(rho,phi,e,n,delta,offset_rate=None):
    psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
    if offset_rate is not None:
        from WERM.myocardial_intercept import logistic_intercept_state
        return logistic_intercept_state(rho,phi,psi,delta,offset_rate,sample_power=10,max_nfev=100,seed=123)
    import WERM.myocardial_approximate as m
    from WERM.myocardial_numerics import logistic_prox_fast
    original=m.logistic_prox;m.logistic_prox=logistic_prox_fast
    try:return m.logistic_state(rho,phi,psi,delta,sample_power=10,max_nfev=100,seed=123)
    finally:m.logistic_prox=original


def check_baseline(root):
    X,y,ages,locations=load_data();setting=pickle.load((root/'fits_delta1.pkl').open('rb'))
    eta=setting['eta'];phi=setting['phi'];rows=[]
    for iteration in (3,30):
        case=f'd1_least_squares_it{iteration}_q9_b100'
        old=pickle.load((root/(case+'.pkl')).open('rb'));rho=old['scenario']['rho']
        out=root/(case+'_independent_seed123_q10.pkl')
        if out.exists():states=pickle.load(out.open('rb'))
        else:
            with parallel_config(backend='loky',inner_max_num_threads=1):
                states=Parallel(n_jobs=18)(delayed(independent_state)(rho,phi,e,len(y),len(y)/X.shape[1]) for e in eta)
            atomic_save(out,states)
        for path in (root/'denser_penalty_grid').glob('fit_*.pkl'):
            name=path.stem[4:];fit=pickle.load(path.open('rb'))
            if 'error' in fit:continue
            ll=[]
            for e,state in zip(eta,states):
                psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
                ll.append(gaussian_logistic_loglikelihood(fit['V'],y,rho,state['lambda'],state['kappa'],psi,quadrature_order=160))
            ll=np.asarray(ll);joint=ll+np.log(setting['prior']);probs=np.exp(joint-logsumexp(joint));curve=np.exp(ll[1:]-logsumexp(ll[1:]))
            reference=pickle.load((root/'denser_penalty_grid'/(case+'_'+name+'.pkl')).open('rb'))
            rows.append(dict(case=case,posterior_fit=name,P_1=probs[1:].sum(),mass_above55=curve[ages[eta[1:]]>55].sum(),
                original_P_1=reference['P_1'],original_mass_above55=reference['mass_above55'],
                TV_from_original=.5*np.abs(curve-reference['conditional']).sum(),
                peak_age=ages[eta[1:][np.argmax(curve)]],FP_residual_max=max(s['training_residual'] for s in states)))
            atomic_save(root/'denser_penalty_grid'/(case+'_'+name+'_independent.pkl'),dict(conditional=curve,log_likelihood=ll,**rows[-1]))
    pd.DataFrame(rows).to_csv(root/'independent_draw_check.csv',index=False)
    print('Independent-draw follow-up complete.',flush=True)


def check_intercept(root):
    from WERM.myocardial_intercept import intercept_loglikelihood
    X,y,ages,locations=load_data();setting=pickle.load((root/'offset_fits.pkl').open('rb'))
    eta=setting['eta'];phi=setting['phi'];rows=[]
    for iteration in (10,30):
        case=f'offset_l1_0.0001_it{iteration}_q9_b100'
        old=pickle.load((root/(case+'.pkl')).open('rb'));rho=old['scenario']['rho']
        out=root/(case+'_independent_seed123_q10.pkl')
        if out.exists():states=pickle.load(out.open('rb'))
        else:
            with parallel_config(backend='loky',inner_max_num_threads=1):
                states=Parallel(n_jobs=18)(delayed(independent_state)(rho,phi,e,len(y),len(y)/X.shape[1],setting['rate']) for e in eta)
            atomic_save(out,states)
        for name,fit in setting['fits'].items():
            ll=[]
            for e,state in zip(eta,states):
                psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
                ll.append(intercept_loglikelihood(fit['V'],y,rho,state,psi))
            ll=np.asarray(ll);joint=ll+np.log(setting['prior']);probs=np.exp(joint-logsumexp(joint));curve=np.exp(ll[1:]-logsumexp(ll[1:]))
            reference=old['results'][name]
            rows.append(dict(case=case,posterior_fit=name,P_1=probs[1:].sum(),mass_above55=curve[ages[eta[1:]]>55].sum(),
                original_P_1=reference['P_1'],original_mass_above55=reference['mass_above55'],
                TV_from_original=.5*np.abs(curve-reference['conditional']).sum(),
                peak_age=ages[eta[1:][np.argmax(curve)]],FP_residual_max=max(s['training_residual'] for s in states)))
            atomic_save(root/(case+'_'+name+'_independent.pkl'),dict(conditional=curve,log_likelihood=ll,**rows[-1]))
    pd.DataFrame(rows).to_csv(root/'independent_draw_check.csv',index=False)
    print('Intercept independent-draw follow-up complete.',flush=True)


if __name__=='__main__':
    import sys
    check_baseline(Path(sys.argv[1]))
