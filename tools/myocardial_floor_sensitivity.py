"""Isolate the eigenvalue floor; preserve the final-redo study unchanged."""
from tools.myocardial_final_redo import *

SOURCE=Path('results/myocardial_final_redo_e3d965cc4b9b')
FLOORS=(1e-4,1e-6,1e-8,0.)


def project(matrix,floor):
    vals,vecs=np.linalg.eigh((matrix+matrix.T)/2)
    return (vecs*np.clip(vals,floor,100.))@vecs.T


def estimate(data,floor):
    X,y,phi=data['X'],data['y'],data['phi'];n,p=X.shape
    B=np.column_stack([np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0] for w in phi])
    V,_=empirical_adjustment(X@B,y,phi,n/p,loss='squared')
    _,psi=pilot(X@B,y,data['eta'],'squared');S=V.T@V/n
    rho=np.eye(2);trace=[]
    for it in range(1,21):
        state=squared_logistic_state(rho,phi,psi,n/p)
        inverse=np.linalg.pinv(state['lambda'],rcond=1e-6)
        raw=inverse.T@(S-state['kappa'])@inverse
        target=project(raw,floor)
        rho=project(.75*rho+.25*target,floor)
        trace.append(dict(iteration=it,raw_eigenvalues=np.linalg.eigvalsh((raw+raw.T)/2),
                          raw_target=raw.copy(),projected_target=target.copy(),rho=rho.copy()))
    state=squared_logistic_state(rho,phi,psi,n/p)
    residual=np.linalg.norm(state['moment']-S)
    # Audit the anticipated clipping-only recurrence, rather than assume it.
    all_negative=all(np.max(t['raw_eigenvalues'])<0 for t in trace)
    if all_negative:
        np.testing.assert_allclose(rho,(floor+(1-floor)*.75**20)*np.eye(2),atol=1e-12)
    return dict(floor=floor,rho=rho,trace=trace,all_raw_targets_negative=all_negative,
                moment_residual=residual,relative_residual=residual/np.linalg.norm(S))


def run():
    data=pickle.load((SOURCE/'fresh_fit_and_rho.pkl').open('rb'))
    fingerprint=hashlib.sha256(Path('tools/myocardial_floor_sensitivity.py').read_bytes()+
        (SOURCE/'specification.json').read_bytes()).hexdigest()[:12]
    root=Path('results')/('myocardial_floor_sensitivity_'+fingerprint);root.mkdir(exist_ok=True)
    (root/'specification.json').write_text(json.dumps(dict(floors=FLOORS,source=str(SOURCE),
        rho_updates=20,power=10,seed=123,quadrature=160,n_jobs=36),indent=2))
    estimates=[estimate(data,f) for f in FLOORS]
    prepared=[]
    for r in estimates:
        folder=root/f"floor_{r['floor']:g}";folder.mkdir(exist_ok=True)
        atomic_save(folder/'rho_estimation.pkl',r)
        prepared.append(dict(data,rho=r['rho'],root=folder))
    np.testing.assert_allclose(estimates[0]['rho'],data['rho'],atol=1e-12)
    baseline=pickle.load((SOURCE/'scenario_100_100_q10_seed123.pkl').open('rb'))
    # Reuse baseline location checkpoints, with unchanged states and likelihood.
    for i in range(len(data['eta'])):
        path=SOURCE/f'eig100_100_q10_seed123_location{i}.pkl'
        target=prepared[0]['root']/path.name
        if not target.exists():atomic_save(target,pickle.load(path.open('rb')))
    jobs=[(j,i) for j in range(len(FLOORS)) for i in range(len(data['eta']))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=36)(delayed(location_task)(prepared[j],(1.,1.),10,123,i) for j,i in jobs)
    rows=[];curves=[]
    for j,(r,d) in enumerate(zip(estimates,prepared)):
        states=results[j*len(data['eta']):(j+1)*len(data['eta'])]
        result=normalize_posterior([s['log_likelihood'] for s in states],data['prior'])
        result.update(ages=baseline['ages'],rho=r['rho'],floor=r['floor'])
        atomic_save(d['root']/'posterior.pkl',result);curves.append(result)
        rows.append(dict(floor=r['floor'],rho_11=r['rho'][0,0],rho_12=r['rho'][0,1],rho_22=r['rho'][1,1],
            correlation=r['rho'][0,1]/np.sqrt(r['rho'][0,0]*r['rho'][1,1]),
            P_1=result['presence'],P_0=result['absence'],mode_age=result['ages'][np.argmax(result['conditional'])],
            mass_above55=result['conditional'][result['ages']>55].sum(),
            conditional_TV_vs_baseline=.5*np.abs(result['conditional']-baseline['conditional']).sum(),
            moment_residual=r['moment_residual'],relative_moment_residual=r['relative_residual'],
            all_raw_targets_negative=r['all_raw_targets_negative'],
            max_state_residual=max(s['state']['training_residual'] for s in states)))
    np.testing.assert_allclose(curves[0]['conditional'],baseline['conditional'],atol=1e-12)
    table=pd.DataFrame(rows);table.to_csv(root/'summary.csv',index=False)
    print('Floor sensitivity complete:',root,flush=True)
    return root,table,curves
