"""Additional centering/intercept experiment with explicit model separation."""
from tools.myocardial_exploration import *
from tools.myocardial_exploration_refinement import evaluate_refined
from WERM.myocardial_intercept import (fit_with_intercept,squared_intercept_state,
    logistic_intercept_state,intercept_loglikelihood)
from scipy.optimize import least_squares


def uncentered_design():
    from scipy.linalg import eigh
    raw=pd.read_csv('results/myocardial_raw_uci579.csv')
    data=raw.loc[raw.AGE.notna()].copy();data=data.iloc[np.argsort(data.AGE.to_numpy())]
    values=data[list(raw.columns[2:112])].to_numpy(float)
    values=np.where(np.isnan(values),np.nanmean(values,axis=0),values)
    mean=values.mean(axis=0);centered=values-mean
    eig,vec=eigh(centered.T@centered/len(values)+1e-6*np.eye(values.shape[1]))
    transform=((vec/np.sqrt(eig))@vec.T)/np.sqrt(len(values))
    return values@transform,mean@transform


def offset_rho_path(S,phi,psi,delta,rate,loss):
    def state_at(rho,power=7,budget=40):
        return squared_intercept_state(rho,phi,psi,delta,rate) if loss=='squared' else logistic_intercept_state(
            rho,phi,psi,delta,rate,sample_power=power,max_nfev=budget)
    rho=np.eye(2);records=[]
    for iteration in range(1,31):
        state=state_at(rho);inv=np.linalg.pinv(state['lambda'],rcond=1e-6)
        target=project_covariance(inv.T@(S-state['kappa'])@inv)
        rho=project_covariance(.75*rho+.25*target)
        if iteration in ITERATIONS:
            check=state_at(rho,power=8,budget=60)
            records.append(dict(rho=rho.copy(),iteration=iteration,
                moment_residual=float(np.linalg.norm(check['lambda'].T@rho@check['lambda']+check['kappa']-S)),
                state_residual=check.get('training_residual',0.)))
    if loss=='squared':
        def decode(x):
            sd=np.exp(x[:2]);return np.array([[sd[0]**2,x[2]*sd.prod()],[x[2]*sd.prod(),sd[1]**2]])
        def residual(x):return (state_at(decode(x))['moment']-S).ravel()
        fits=[least_squares(residual,[v,v,0.],bounds=([-8,-8,-.9999],[np.log(10),np.log(10),.9999]),
                   max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10) for v in (-2.,-1.,0.)]
        best=min(fits,key=lambda f:np.linalg.norm(f.fun))
        records.append(dict(rho=decode(best.x),iteration=-1,moment_residual=float(np.linalg.norm(best.fun)),
                            state_residual=0.,optimizer_success=bool(best.success)))
    return records


def offset_prepare(root,X,y,ages,locations):
    path=root/'offset_fits.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    eta,prior,phi=configuration_prior(len(y),locations)
    fits={f'{k}_{v:g}':fit_with_intercept(X,y,phi,k,v) for k,v in FIT_SETTINGS}
    auxiliary={'least_squares':fit_with_intercept(X,y,phi,'none',0.,loss='squared'),
               'l2_1e-06':fits['l2_1e-06'],'l1_0.0001':fits['l1_0.0001']}
    scenarios=[]
    for name,fit in auxiliary.items():
        loss='squared' if name=='least_squares' else 'logistic'
        e,psi=pilot(fit['theta'],y,eta,loss);S=fit['V'].T@fit['V']/len(y)
        for record in offset_rho_path(S,phi,psi,len(y)/X.shape[1],y.mean(),loss):
            scenarios.append(dict(record,auxiliary=name,pilot=e,loss=loss))
    Xraw,shift=uncentered_design();affine=[]
    np.testing.assert_allclose(Xraw-shift,X,atol=2e-12)
    for name,fit in fits.items():
        transformed_intercept=fit['intercepts']-shift@fit['coef']
        prediction=Xraw@fit['coef']+transformed_intercept
        affine.append(dict(fit=name,max_prediction_difference=float(np.max(abs(prediction-fit['theta'])))))
    result=dict(eta=eta,prior=prior,phi=phi,fits=fits,scenarios=scenarios,rate=y.mean(),affine=affine)
    atomic_save(path,result);return result


def offset_case(setting,scenario,X,y,ages,root,power=9,budget=100):
    label=f"offset_{scenario['auxiliary']}_it{scenario['iteration']}_q{power}_b{budget}"
    path=root/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    values={name:[] for name in setting['fits']};states=[]
    for e in setting['eta']:
        psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
        state=logistic_intercept_state(scenario['rho'],setting['phi'],psi,len(y)/X.shape[1],setting['rate'],
                                      sample_power=power,max_nfev=budget)
        states.append(state)
        for name,fit in setting['fits'].items():
            values[name].append(intercept_loglikelihood(fit['V'],y,scenario['rho'],state,psi))
    results={}
    for name,ll in values.items():
        ll=np.array(ll);joint=ll+np.log(setting['prior']);probs=np.exp(joint-logsumexp(joint))
        curve=np.exp(ll[1:]-logsumexp(ll[1:]))
        results[name]=dict(P_1=probs[1:].sum(),conditional=curve,
            mass_above55=curve[ages[setting['eta'][1:]]>55].sum(),log_likelihood=ll,
            peak_age=ages[setting['eta'][1:][np.argmax(curve)]])
    result=dict(label=label,scenario=scenario,states=states,results=results,eta=setting['eta'])
    atomic_save(path,result);print('Completed',label,flush=True);return result


def uncentered_prepare(root,y,ages,locations):
    path=root/'uncentered_fits.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    X,shift=uncentered_design();eta,prior,phi=configuration_prior(len(y),locations)
    fits={f'{k}_{v:g}':fit_logistic(X,y,phi,k,v) for k,v in FIT_SETTINGS}
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    theta=X@B;V,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='squared')
    e,psi=pilot(theta,y,eta,'squared');S=V.T@V/len(y)
    scenarios=[dict(r,auxiliary='uncentered_least_squares',pilot=e,loss='squared') for r in rho_path(S,phi,psi,len(y)/X.shape[1],'squared')]
    direct=approximate_rho(S,phi,psi,len(y)/X.shape[1])
    scenarios.append(dict(rho=direct['rho'],iteration=-1,moment_residual=direct['residual_norm'],state_residual=0.,
                          auxiliary='uncentered_least_squares',pilot=e,loss='squared'))
    result=dict(eta=eta,prior=prior,phi=phi,fits=fits,scenarios=scenarios,delta_min=1)
    atomic_save(path,result);return result


def run_intercept_exploration():
    X,y,ages,locations=load_data()
    provenance={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['tools/myocardial_intercept_exploration.py','WERM/myocardial_intercept.py','WERM/myocardial_numerics.py']}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_intercept_exploration_'+fingerprint);root.mkdir(exist_ok=True)
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Intercept extension:',root,flush=True)
    offset=offset_prepare(root,X,y,ages,locations)
    uncentered=uncentered_prepare(root,y,ages,locations)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=26)(delayed(offset_case)(offset,r,X,y,ages,root) for r in offset['scenarios'])
        diagnostics=Parallel(n_jobs=5)(delayed(evaluate_refined)(uncentered,r,X,y,ages,root,9,100) for r in uncentered['scenarios'])
    rows=[]
    for rec in results:
        for fit,stats in rec['results'].items():
            rows.append(dict(model='prevalence-calibrated intercept',setting=rec['label'],posterior_fit=fit,
                P_1=stats['P_1'],mass_above55=stats['mass_above55'],peak_age=stats['peak_age'],
                moment_residual=rec['scenario']['moment_residual'],FP_residual_max=max(s['training_residual'] for s in rec['states'])))
    for rec in diagnostics:
        for fit,stats in rec['results'].items():
            rows.append(dict(model='uncentered fit, old equations (diagnostic)',setting=rec['label'],posterior_fit=fit,
                P_1=stats['presence'],mass_above55=stats['conditional'][ages[rec['eta'][1:]]>55].sum(),peak_age=stats['peak_age'],
                moment_residual=rec['scenario']['moment_residual'],FP_residual_max=max(rec['residuals'])))
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    print('Intercept/centering extension complete:',len(rows),'comparisons.',flush=True)
    return root


if __name__=='__main__':run_intercept_exploration()
