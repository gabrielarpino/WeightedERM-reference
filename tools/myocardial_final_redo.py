"""Fresh rank-3 reconstruction and paired eigenvalue sensitivity."""
from tools.myocardial_finite_iteration_exploration import *
from itertools import product

ARCHIVE=Path('results/myocardial_finite_iteration_443c17cf2df6/focus_354ed8f1e782')
ARCHIVE_CASE='d2_least_squares_it20_q8_b60'
LEVELS=(.8,.9,1.,1.1,1.2)
CHECKS=((9,42),(10,123))


def perturb_rho(rho, first, second):
    rho=np.asarray(rho,float)
    if first<=0 or second<=0:raise ValueError('positive eigenvalue multipliers required')
    if first==1 and second==1:return rho.copy()
    values,vectors=np.linalg.eigh(rho)
    if values.min()<=0:raise ValueError('positive definite rho required')
    return (vectors*(values*np.array([first,second])))@vectors.T


def normalize_posterior(log_likelihood, prior):
    ll=np.asarray(log_likelihood,float);prior=np.asarray(prior,float)
    if ll.shape!=prior.shape or len(ll)<2 or not np.isfinite(ll).all() or np.any(prior<=0):
        raise ValueError('finite likelihoods and positive aligned prior required')
    np.testing.assert_allclose(prior.sum(),1.,atol=1e-12)
    joint=ll+np.log(prior);q=np.exp(joint-logsumexp(joint))
    conditional=np.exp(joint[1:]-logsumexp(joint[1:]))
    return dict(probabilities=q,conditional=conditional,presence=float(q[1:].sum()),
                absence=float(q[0]),log_likelihood=ll)


def reconstruct():
    X,y,ages,locations=load_data();n,p=X.shape
    eta,prior,phi=configuration_prior(n,locations[(locations>=2)&(locations<=n-2)])
    assert eta[0]==-1
    np.testing.assert_allclose([prior[0],prior[1:].sum()],[.5,.5])
    # The exact rank-3 auxiliary procedure, freshly fitted from raw data.
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    theta=X@B;V,b=empirical_adjustment(theta,y,phi,n/p,loss='squared')
    pilot_eta,psi=pilot(theta,y,eta,'squared');S=V.T@V/n
    rho=np.eye(2);trajectory=[]
    for iteration in range(1,21):
        state=squared_logistic_state(rho,phi,psi,n/p)
        inverse=np.linalg.pinv(state['lambda'],rcond=1e-6)
        rho=project_covariance(.75*rho+.25*project_covariance(inverse.T@(S-state['kappa'])@inverse))
        check=squared_logistic_state(rho,phi,psi,n/p)
        trajectory.append(dict(iteration=iteration,rho=rho.copy(),
            moment_residual=float(np.linalg.norm(check['moment']-S))))
    fitted=finite_fit(X,y,phi,16)
    # Stored results are acceptance checks, never inputs to the new estimates.
    old_setting=pickle.load((ARCHIVE/'fits_delta2.pkl').open('rb'))
    old_rho=next(s['rho'] for s in old_setting['scenarios'] if s['auxiliary']=='least_squares' and s['iteration']==20)
    np.testing.assert_array_equal(eta,old_setting['eta'])
    np.testing.assert_allclose(phi,old_setting['phi'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(rho,old_rho,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(fitted['theta'],old_setting['fits']['fista16']['theta'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(fitted['V'],old_setting['fits']['fista16']['V'],rtol=1e-9,atol=1e-10)
    files=('tools/myocardial_final_redo.py','WERM/signal_strength.py','WERM/myocardial_approximate.py',
           'WERM/myocardial_numerics.py','WERM/myocardial_early_stopping.py','WERM/proximal.py',
           'tools/myocardial_exploration.py','results/myocardial_raw_uci579.csv')
    spec=dict(n=n,p=p,Delta=2,fit_budget=16,auxiliary='LS',posterior='unpenalized logistic FISTA',
        intercept=False,rho_updates=20,rho_start='identity',damping=.25,eigenvalue_bounds=[1e-4,100],
        checks=CHECKS,state_max_nfev=100,likelihood_quadrature=160,eigenvalue_levels=LEVELS,
        count_prior=[.5,.5],files={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    root=Path('results')/('myocardial_final_redo_'+digest);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec,indent=2))
    out=dict(X=X,y=y,ages=ages,eta=eta,prior=prior,phi=phi,rho=rho,fit=fitted,
        trajectory=trajectory,pilot=pilot_eta,root=root,specification=spec)
    atomic_save(root/'fresh_fit_and_rho.pkl',out)
    print('Fresh fits and rho reproduce rank 3. Output:',root,flush=True)
    return out


def location_task(data,levels,power,seed,index):
    root=data['root'];label=f'eig{int(levels[0]*100)}_{int(levels[1]*100)}_q{power}_seed{seed}'
    path=root/f'{label}_location{index}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    rho=perturb_rho(data['rho'],*levels);y=data['y'];n,p=data['X'].shape;e=data['eta'][index]
    psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
    import WERM.myocardial_approximate as impl
    old=impl.logistic_prox;impl.logistic_prox=logistic_prox_fast
    try:
        state=impl.logistic_state(rho,data['phi'],psi,n/p,sample_power=power,max_nfev=100,seed=seed)
        ll=gaussian_logistic_loglikelihood(data['fit']['V'],y,rho,state['lambda'],state['kappa'],psi,quadrature_order=160)
    finally:impl.logistic_prox=old
    if not np.isfinite(ll):raise FloatingPointError('nonfinite likelihood')
    record=dict(state=state,log_likelihood=float(ll),index=index,levels=levels,power=power,seed=seed)
    atomic_save(path,record);return record


def compute_scenarios(data,levels_list):
    tasks=[(levels,power,seed,i) for levels in levels_list for power,seed in CHECKS for i in range(len(data['eta']))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        outputs=Parallel(n_jobs=36)(delayed(location_task)(data,*task) for task in tasks)
    records={};k=0
    for levels in levels_list:
        for power,seed in CHECKS:
            selected=outputs[k:k+len(data['eta'])];k+=len(data['eta'])
            result=normalize_posterior([r['log_likelihood'] for r in selected],data['prior'])
            result.update(levels=levels,power=power,seed=seed,rho=perturb_rho(data['rho'],*levels),
                ages=data['ages'][data['eta'][1:]],
                residuals=np.array([r['state']['training_residual'] for r in selected]))
            if levels==(1.,1.):
                old=pickle.load((ARCHIVE/f'{ARCHIVE_CASE}_fista16_q{power}_seed{seed}.pkl').open('rb'))
                np.testing.assert_allclose(result['presence'],old['P_1'],rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose(result['conditional'],old['conditional'],rtol=1e-6,atol=1e-8)
            records[(levels,power,seed)]=result
            atomic_save(data['root']/f'scenario_{int(levels[0]*100)}_{int(levels[1]*100)}_q{power}_seed{seed}.pkl',result)
    return records
