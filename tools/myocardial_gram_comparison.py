"""Paired whole-data and estimated-segment coefficient Gram experiments."""
from tools.myocardial_final_redo import *
from tools.myocardial_direct_moment_experiment import state_at
from tools.myocardial_best_ones import curve_metrics
from WERM.myocardial_finite_intercept import finite_intercept_column
from WERM.myocardial_no_intercept_solvers import fit_column
from WERM.myocardial_intercept import fit_with_intercept,intercept_loglikelihood
from tools.myocardial_matched_l2_fine import solve_logistic_l2

BASE=Path('results/myocardial_final_redo_e3d965cc4b9b')


def configurations():
    specs=[]
    for d in (1,2,5):
        for intercept in (False,True):
            for aux in ('squared','logistic'):
                for method in ('fista','lbfgs','gd'):
                    for budget in (8,16,20,30,50,100):
                        specs.append(dict(Delta=d,intercept=intercept,aux_loss=aux,
                            penalty='none',strength=0.,method=method,budget=budget))
                for penalty in ('l1','l2'):
                    for strength in (1e-8,1e-6,1e-5,1e-4,1e-3):
                        specs.append(dict(Delta=d,intercept=intercept,aux_loss=aux,
                            penalty=penalty,strength=strength,method='full',budget=0))
    for i,s in enumerate(specs):s['id']=f'config{i:03d}'
    return specs


def column(X,y,w,spec,loss):
    n,p=X.shape;intercept=spec['intercept'];family=spec['penalty'];strength=spec['strength']
    if loss=='logistic' and family=='none':
        if intercept:return finite_intercept_column(X,y,w,spec['method'],spec['budget'])
        return fit_column(X,y,w,spec['method'],spec['budget'])
    if loss=='logistic' and family=='l2' and not intercept:
        r=solve_logistic_l2(X,y,w,strength);return dict(r,intercept=0.)
    if loss=='logistic' and intercept:
        r=fit_with_intercept(X,y,np.array([w]),family,strength)
        return dict(coef=r['coef'][:,0],intercept=float(r['intercepts'][0]),**r['checks'][0])
    design=np.column_stack([X,np.ones(n)]) if intercept else X
    r=fit_weighted(design,y,sample_weight=w,loss=loss,loss_scale=2. if loss=='squared' else 1.,
        penalty=family,penalty_strength=strength,unpenalized_indices=[p] if intercept else [],
        max_iter=30000,tol=1e-8)
    return dict(coef=r.coef[:p],intercept=float(r.coef[p]) if intercept else 0.,
                kkt=r.kkt_violation,success=r.converged,iterations=r.n_iter)


def fit(X,y,phi,spec,loss):
    cols=[column(X,y,w,spec,loss) for w in phi]
    B=np.column_stack([c['coef'] for c in cols]);alpha=np.array([c.get('intercept',0.) for c in cols])
    theta=X@B+alpha;adjusted,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss=loss)
    V=adjusted-(X.mean(axis=0)@B+alpha) if spec['intercept'] else adjusted
    return dict(coef=B,theta=theta,V=V,b=b,intercepts=alpha,checks=cols)


def prepare(spec,X,y,locations,root):
    path=root/(spec['id']+'_fits.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    try:
        d=spec['Delta'];eta,prior,phi=configuration_prior(len(y),locations[(locations>=d)&(locations<=len(y)-d)])
        auxiliary=fit(X,y,phi,spec,spec['aux_loss'])
        posterior=auxiliary if spec['aux_loss']=='logistic' else fit(X,y,phi,spec,'logistic')
        e,psi=pilot(auxiliary['theta'],y,eta,spec['aux_loss'])
        result=dict(spec=spec,setting=dict(eta=eta,prior=prior,phi=phi),auxiliary=auxiliary,
            posterior=posterior,pilot=e,psi=psi,loss=spec['aux_loss'],delta=len(y)/X.shape[1],rate=float(y.mean()))
    except Exception:result=dict(error=traceback.format_exc(),spec=spec)
    atomic_save(path,result);return result


def balanced_segment_weights(y):
    y=np.asarray(y)
    if not np.all(np.isin(y,[0,1])) or len(np.unique(y))!=2:
        raise ValueError('Balanced segment requires both binary classes')
    counts=np.bincount(y.astype(int),minlength=2)
    return len(y)/(2*counts[y.astype(int)])


def gram_coefficients(prepared,X,y,mode,root):
    if mode=='whole':return prepared['auxiliary']['coef']
    path=root/(prepared['spec']['id']+'_segment_coefficients.pkl')
    if path.exists():return pickle.load(path.open('rb'))
    e=prepared['pilot']
    if not 0<e<len(y):raise ValueError('No pilot change point: segment Gram undefined')
    spec=prepared['spec'];columns=[]
    for segment in (slice(None,e),slice(e,None)):
        Xi,yi=X[segment],y[segment]
        w=balanced_segment_weights(yi) if mode=='segmented_balanced' else np.ones(len(yi))
        c=column(Xi,yi,w,spec,spec['aux_loss']);columns.append(c)
    B=np.column_stack([c['coef'] for c in columns])
    atomic_save(path,B);atomic_save(root/(spec['id']+'_segment_checks.pkl'),columns)
    return B


def evaluate(prepared,X,y,ages,mode,denominator,root,power=9,seed=123):
    ident=prepared['spec']['id']+'_'+denominator
    path=root/f'{ident}_p{power}s{seed}.pkl'
    if path.exists():return pickle.load(path.open('rb'))
    partial=root/f'{ident}_p{power}s{seed}_partial.pkl'
    try:
        spec=prepared['spec'];B=gram_coefficients(prepared,X,y,mode,root)
        divisor=len(y) if denominator=='n' else X.shape[1]
        rho=B.T@B/divisor;eigenvalues=np.linalg.eigvalsh(rho)
        if eigenvalues.min()<=0 or eigenvalues.min()<1e-12*eigenvalues.max():
            raise ValueError('Singular/numerically rank-deficient Gram: no jitter, clipping or repair applied')
        states=pickle.load(partial.open('rb')) if partial.exists() else []
        eta=prepared['setting']['eta'];n=len(y)
        for e in eta[len(states):]:
            psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            states.append(state_at(prepared,rho,psi,power=power,seed=seed,loss='logistic'))
            if len(states)%8==0:atomic_save(partial,states)
        atomic_save(partial,states);values=[]
        for e,state in zip(eta,states):
            psi=np.zeros(n,int) if e<0 else (np.arange(n)>=e).astype(int)
            if spec['intercept']:ll=intercept_loglikelihood(prepared['posterior']['V'],y,rho,state,psi,order=160)
            else:ll=gaussian_logistic_loglikelihood(prepared['posterior']['V'],y,rho,state['lambda'],state['kappa'],psi,quadrature_order=160)
            values.append(ll)
        result=normalize_posterior(values,prepared['setting']['prior']);a=ages[eta[1:]]
        target=pickle.load((BASE/'scenario_100_100_q10_seed123.pkl').open('rb'))
        metrics=curve_metrics(a,result['conditional'],dict(zip(target['ages'],target['conditional'])))
        result.update(spec,id=ident,mode=mode,denominator=denominator,rho=rho,ages=a,
            pilot=prepared['pilot'],pilot_age=ages[prepared['pilot']] if prepared['pilot']>=0 else np.nan,
            P_1=result['presence'],TV_to_rank3=metrics['TV'],peak_age=metrics['peak'],mass_above55=metrics['above55'],
            correlation=rho[0,1]/np.sqrt(rho[0,0]*rho[1,1]),rho_11=rho[0,0],rho_12=rho[0,1],rho_22=rho[1,1],
            max_state_residual=max(s['training_residual'] for s in states),power=power,seed=seed,checkpoint=str(path),
            max_aux_kkt=max(c.get('kkt',np.nan) for c in prepared['auxiliary']['checks']),
            max_post_kkt=max(c.get('kkt',np.nan) for c in prepared['posterior']['checks']))
    except Exception:result=dict(error=traceback.format_exc(),id=ident,spec=prepared['spec'],mode=mode,power=power,seed=seed)
    atomic_save(path,result);return result


def table(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k not in ('rho','ages','conditional','probabilities','log_likelihood')}
        for r in rows if 'error' not in r])


def run(mode='whole',smoke=False):
    if mode not in ('whole','segmented','segmented_balanced'):raise ValueError(mode)
    X,y,ages,locations=load_data();specs=configurations()
    fingerprint=hashlib.sha256(Path('tools/myocardial_gram_comparison.py').read_bytes()).hexdigest()[:12]
    shared=Path('results')/('myocardial_gram_pair_'+fingerprint);shared.mkdir(exist_ok=True)
    root=shared/mode;root.mkdir(exist_ok=True)
    (shared/'specification.json').write_text(json.dumps(dict(configurations=specs,normalizations=['n','p'],
        primary_draws=512,final_draws=1024,final_seeds=[123,42],coefficient_gram='slopes only, intercept excluded',
        segmented='same pilot auxiliary loss, solver, penalty; full-data n or p divisor, no segment rewhitening',
        penalty_matching='same family and strength in auxiliary and posterior; squared loss scale=2'),indent=2))
    if smoke:
        chosen=[dict(specs[0],id='smoke_ls'),dict(next(s for s in specs if s['aux_loss']=='logistic' and s['intercept']),id='smoke_logistic')]
        for spec in chosen:
            p=prepare(spec,X,y,locations,shared)
            assert 'error' not in p,p
            B=gram_coefficients(p,X,y,mode,root)
            assert B.shape==(X.shape[1],2)
            for denom in ('n','p'):assert np.linalg.eigvalsh(B.T@B/(len(y) if denom=='n' else X.shape[1])).min()>0
        print('Gram fit and',mode,'segment checks passed; configurations:',len(specs)*2,flush=True);return root
    with parallel_config(backend='loky',inner_max_num_threads=1):
        prepared=Parallel(n_jobs=36)(delayed(prepare)(s,X,y,locations,shared) for s in specs)
        # Segment coefficients are populated once per setting before the two
        # normalization tasks, preventing competing writes to their checkpoint.
    errors=[p for p in prepared if 'error' in p]
    valid=[]
    for p in prepared:
        if 'error' in p:continue
        if mode!='whole':
            try:gram_coefficients(p,X,y,mode,root)
            except Exception:
                errors.append(dict(id=p['spec']['id'],error=traceback.format_exc()));continue
        valid.append(p)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=36)(delayed(evaluate)(p,X,y,ages,mode,d,root) for p in valid for d in ('n','p'))
    errors += [r for r in results if 'error' in r]
    summary=table(results);summary.to_csv(root/'summary.csv',index=False)
    selected=pd.concat([summary.sort_values('TV_to_rank3').head(20),
        summary.sort_values('TV_to_rank3').groupby(['aux_loss','penalty','denominator']).head(2)]).drop_duplicates('id')
    selected.to_csv(root/'selected_for_validation.csv',index=False)
    mapping={p['spec']['id']:p for p in valid}
    with parallel_config(backend='loky',inner_max_num_threads=1):
        checked=Parallel(n_jobs=36)(delayed(evaluate)(mapping[row.id.rsplit('_',1)[0]],X,y,ages,mode,row.denominator,root,10,seed)
            for row in selected.itertuples() for seed in (123,42))
    errors += [r for r in checked if 'error' in r];table(checked).to_csv(root/'validation.csv',index=False)
    ranked=[]
    for ident in selected.id:
        pair=[r for r in checked if r['id']==ident and 'error' not in r]
        if len(pair)!=2:continue
        a,b=pair;row=table([a]).iloc[0].to_dict()
        row.update(mean_TV_to_rank3=(a['TV_to_rank3']+b['TV_to_rank3'])/2,P1_min=min(a['P_1'],b['P_1']),
            P1_max=max(a['P_1'],b['P_1']),numerical_TV=.5*np.abs(a['conditional']-b['conditional']).sum(),check_checkpoint=b['checkpoint'])
        ranked.append(row)
    ranked=pd.DataFrame(ranked).sort_values('mean_TV_to_rank3');ranked.insert(0,'rank',np.arange(1,len(ranked)+1))
    ranked.to_csv(root/'ranked.csv',index=False);ranked.head(20).to_csv(root/'top_twenty.csv',index=False)
    (root/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    if mode!='whole' and (shared/'whole'/'summary.csv').exists():
        whole=pd.read_csv(shared/'whole'/'summary.csv')
        summary.merge(whole,on='id',suffixes=('_segmented','_whole')).to_csv(root/'paired_whole_vs_segmented.csv',index=False)
    print('GRAM STUDY COMPLETE',mode,root,flush=True);return root


if __name__=='__main__':
    import sys
    run(mode='segmented' if '--segmented' in sys.argv else 'whole',smoke='--smoke' in sys.argv)
