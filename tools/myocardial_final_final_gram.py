"""Canonical analyses of two fixed, previously selected Gram candidates."""
from tools.myocardial_gram_comparison import *
from tools.myocardial_final_redo import perturb_rho

GRAM_ARCHIVE=Path('results/myocardial_gram_pair_1ce06b10b4f5')
FINAL_ROOT=Path('results/myocardial_final_final_v1')
LEVELS=(.8,.9,1.,1.1,1.2)
REPORT_CHECKS=((10,123),)


def reconstruct_gram(mode):
    # Fixed selections, not a new search for a favorable posterior probability.
    ident,denominator={'whole':('config283','p'),'segmented':('config189','n')}[mode]
    spec=next(s for s in configurations() if s['id']==ident)
    root=FINAL_ROOT/mode;root.mkdir(parents=True,exist_ok=True)
    X,y,ages,locations=load_data()
    prepared=prepare(spec,X,y,locations,root)
    if 'error' in prepared:raise RuntimeError(prepared['error'])
    B=gram_coefficients(prepared,X,y,mode,root)
    rho=B.T@B/(X.shape[1] if denominator=='p' else len(y))
    archived=pickle.load((GRAM_ARCHIVE/mode/f'{ident}_{denominator}_p10s123.pkl').open('rb'))
    np.testing.assert_allclose(rho,archived['rho'],rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose([prepared['setting']['prior'][0],prepared['setting']['prior'][1:].sum()],[.5,.5])
    data=dict(prepared=prepared,X=X,y=y,ages=ages,rho=rho,root=root,mode=mode,
              ident=ident,denominator=denominator,fit=prepared['posterior'],pilot=prepared['pilot'],
              **prepared['setting'])
    atomic_save(root/'fresh_fit_and_rho.pkl',data)
    (root/'specification.json').write_text(json.dumps(dict(spec=spec,mode=mode,denominator=denominator,
        count_prior=[.5,.5],draws=1024,seed=123,levels=LEVELS,source='selected by mean TV to rank3',
        code_sha256=hashlib.sha256(Path('tools/myocardial_final_final_gram.py').read_bytes()).hexdigest()),indent=2))
    return data


def gram_location(data,levels,index):
    label=f'eig{round(100*levels[0])}_{round(100*levels[1])}_q10_seed123_location{index}'
    path=data['root']/(label+'.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    rho=perturb_rho(data['rho'],*levels)
    e=data['eta'][index];psi=np.zeros(len(data['y']),int) if e<0 else (np.arange(len(data['y']))>=e).astype(int)
    if levels==(1.,1.):
        # Reuse exactly matched, already checked baseline integration states.
        old=GRAM_ARCHIVE/data['mode']/f"{data['ident']}_{data['denominator']}_p10s123_partial.pkl"
        state=pickle.load(old.open('rb'))[index]
    else:state=state_at(data['prepared'],rho,psi,power=10,seed=123,loss='logistic')
    ll=intercept_loglikelihood(data['fit']['V'],data['y'],rho,state,psi,order=160)
    if not np.isfinite(ll):raise FloatingPointError(label)
    result=dict(state=state,log_likelihood=float(ll))
    atomic_save(path,result);return result


def gram_scenarios(data,levels_list):
    tasks=[(levels,i) for levels in levels_list for i in range(len(data['eta']))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        outputs=Parallel(n_jobs=36)(delayed(gram_location)(data,levels,i) for levels,i in tasks)
    records={};n=len(data['eta'])
    for k,levels in enumerate(levels_list):
        selected=outputs[k*n:(k+1)*n]
        result=normalize_posterior([r['log_likelihood'] for r in selected],data['prior'])
        result.update(ages=data['ages'][data['eta'][1:]],rho=perturb_rho(data['rho'],*levels),
                      residuals=np.array([r['state']['training_residual'] for r in selected]))
        if levels==(1.,1.):
            old=pickle.load((GRAM_ARCHIVE/data['mode']/f"{data['ident']}_{data['denominator']}_p10s123.pkl").open('rb'))
            np.testing.assert_allclose(result['presence'],old['P_1'],atol=1e-7,rtol=1e-7)
            np.testing.assert_allclose(result['conditional'],old['conditional'],atol=1e-7,rtol=1e-6)
        records[(levels,10,123)]=result
        atomic_save(data['root']/f'scenario_{round(100*levels[0])}_{round(100*levels[1])}_q10_seed123.pkl',result)
    return records


def gram_diagnostics(data,baseline):
    prepared=data['prepared'];V=prepared['auxiliary']['V'];observed=V.T@V/len(V)
    state=state_at(prepared,data['rho'],prepared['psi'],power=10,seed=123,loss=prepared['loss'])
    residual=np.linalg.norm(state['moment']-observed,'fro')
    display(pd.DataFrame([{'Frobenius moment-equation residual':residual,
        'Relative moment-equation residual':residual/np.linalg.norm(observed,'fro'),
        'Correlation':data['rho'][0,1]/np.sqrt(np.prod(np.diag(data['rho'])))}]).T.rename(columns={0:'Value'}).style.format('{:.9g}'))
    values=[]
    for i,e in enumerate(data['eta']):
        state=gram_location(data,(1.,1.),i)['state']
        psi=np.zeros(len(V),int) if e<0 else (np.arange(len(V))>=e).astype(int)
        values.append(intercept_loglikelihood(data['fit']['V'],data['y'],data['rho'],state,psi,order=320))
    refined=normalize_posterior(values,data['prior'])
    check=pickle.load((GRAM_ARCHIVE/data['mode']/f"{data['ident']}_{data['denominator']}_p10s42.pkl").open('rb'))
    table=pd.DataFrame([{'QMC draws':1024,'P(1), seed 123':baseline['presence'],'P(1), seed 42':check['P_1'],
        'Conditional TV, seeds':.5*np.abs(check['conditional']-baseline['conditional']).sum(),
        'Max sampled-state equation residual':baseline['residuals'].max(),
        'Absolute P(1) difference, quadrature 160 vs 320':abs(refined['presence']-baseline['presence']),
        'Conditional TV, quadrature':.5*np.abs(refined['conditional']-baseline['conditional']).sum()}])
    table.to_csv(data['root']/'numerical_diagnostics.csv',index=False)
    display(table.T.rename(columns={0:'Value'}).style.format('{:.9g}'))
    print('Gram estimates do not solve the moment equations. These diagnostics are numerical checks, not statistical error bounds or evidence of calibration.')
