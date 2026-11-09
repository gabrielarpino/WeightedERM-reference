"""Paired Gaussian-signal WERM/AMP experiment; resumable, without whitening.

Run from the repository root with ``python -m tools.werm_amp_experiment``.
Every dataset is saved independently, including unsuccessful AMP fits.  The
notebook renders these records; plotting never invokes an estimator.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import gammaln
from WERM.proximal import fit_weighted
from tools.werm_amp_partition import prior_marginals, amp_prior_marginals, partition_minimize

ROOT = Path(__file__).resolve().parents[1]
WERM_P_VERSION = 'upper_bound_half_squared_v1'
DEFAULT_SPEC = dict(
    schema=1, seed=20261109, p=200, trials=100,
    deltas=[1.5, 3.5, 7.5, 11.5], covariances=['identity', 'ar1', 'm3'],
    studies=['unknown', 'known'], max_segments=4,
    unknown_truth=[.2, .5], known_truth=[.25, .45, .70],
    noise_std=float(np.sqrt(.1)), coefficient_variance='delta',
    ar1_correlation=.4, m3_df=10, m3_seed=42, folds=10,
    separation_fraction=1/40, minimum_gap_fraction=1/50,
    endpoint_minimum=1, gamma=6., amp_iterations=10, amp_mc_samples=16384,
    coefficient_loss='squared', coefficient_penalty='none',
    cv_validation_loss='absolute',
    werm_p_version=WERM_P_VERSION,
    penalty='0.5 * RSS_K + gamma * log binomial(n-1,K-1); shared upper-bound fit',
    penalty_search_loss='half squared',
    penalty_prior='uniform K=1..L, uniform admissible locations conditional on K, prefix labels',
    marginal_weights='exact finite-n conditional-K weights for CV; upper-bound mixture for WERM+P',
    amp_configuration_prior=('uniform K=1..L, uniform admissible boundaries, '
        'uniform source label subsets: any constant label for K=1; '
        'label 0 then increasing subsets for K>=2'),
    whitening=False, response_standardization=False,
)
METHOD_STYLES = {
    'WERM+CV': ('black', 'D', '-'), 'WERM+P': ('#4477AA', 's', '--'),
    'WERM': ('black', 'D', '-'), 'AMP': ('#EE7733', 'o', '-.'),
}
COVARIANCE_TITLES = {'identity': r'$\Sigma=I$',
    'ar1': r'AR(1), $r=0.4$', 'm3': 'M3 random correlation'}


def atomic_pickle(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)


def minimum_gap(n, spec):
    return max(1, int(np.ceil(n*spec['minimum_gap_fraction'])),
               int(n*spec['separation_fraction']))


def covariance_matrices(spec):
    p = spec['p']
    rng = np.random.default_rng(spec['m3_seed'])
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eigenvalues = rng.chisquare(spec['m3_df'], size=p)
    S = Q.T @ np.diag(eigenvalues) @ Q
    corr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    indices = np.arange(p)
    return {'identity': np.eye(p),
            'ar1': spec['ar1_correlation']**np.abs(indices[:,None]-indices),
            'm3': corr}


def prepare(spec=None):
    """Freeze a complete specification and priors, without running any fits."""
    cfg = dict(DEFAULT_SPEC)
    if spec is not None:
        cfg.update(spec)
    for key in ('werm_p_version', 'penalty', 'penalty_search_loss',
                'penalty_prior', 'marginal_weights'):
        if cfg[key] != DEFAULT_SPEC[key]:
            raise ValueError(f'Unsupported WERM+P convention in {key}; '
                             'prepare a new specification for the corrected upper-bound fit.')
    assert cfg['p'] >= 1 and cfg['trials'] >= 1
    assert cfg['max_segments'] == 4 and cfg['folds'] == 10
    assert set(cfg['studies']) <= {'unknown', 'known'}
    assert set(cfg['covariances']) <= set(COVARIANCE_TITLES)
    assert not cfg['whitening'] and not cfg['response_standardization']
    for study in cfg['studies']:
        assert np.all(np.diff([0., *cfg[study+'_truth'], 1.]) > 0)
    # Sources are captured so changed algorithms never silently reuse results.
    source_paths = ['tools/werm_amp_experiment.py', 'tools/werm_amp_adapter.py',
                    'tools/werm_amp_partition.py', 'WERM/proximal.py']
    amp_root = ROOT.parent/'AMP_chgpt_reg'
    cfg['sources_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in source_paths}
    cfg['amp_sources_sha256'] = {str(p.relative_to(amp_root)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((amp_root/'amp').glob('*.py'))}
    import importlib.metadata
    cfg['versions'] = {name:importlib.metadata.version(name)
        for name in ('numpy','scipy','jax','jaxlib','pandas')}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    root = ROOT/'results/WERM_AMP_comparison'/digest
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root/'specification.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == cfg
    else:
        spec_path.write_text(json.dumps(cfg, indent=2)+'\n')
    if not (root/'weights.pkl').exists():
        weights = {}
        for delta in cfg['deltas']:
            n = round(cfg['p']*delta)
            sizes = {n, *[n-len(range(fold,n,cfg['folds'])) for fold in range(cfg['folds'])]}
            for m in sizes:
                gap = minimum_gap(m, cfg)
                for k in range(1,cfg['max_segments']+1):
                    weights[m,k] = prior_marginals(m,k,gap,fixed_count=True)
            weights[n,'mixture'] = amp_prior_marginals(n,cfg['max_segments'],
                                                  minimum_gap(n,cfg),fixed_count=False)
            # WERM uses prefix labels; AMP's label-subset mixture is different.
            weights[n,'werm_upper'] = prior_marginals(n,cfg['max_segments'],
                                                    minimum_gap(n,cfg),fixed_count=False)
        atomic_pickle(root/'weights.pkl', weights)
        matrices = covariance_matrices(cfg)
        atomic_pickle(root/'covariances.pkl', matrices)
        stats = {name:dict(min_eigenvalue=float(np.linalg.eigvalsh(S)[0]),
                          max_eigenvalue=float(np.linalg.eigvalsh(S)[-1]),
                          condition=float(np.linalg.cond(S)),trace=float(np.trace(S)))
                 for name,S in matrices.items()}
        (root/'covariance_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
    return root


def generate_data(spec, study, covariance, delta, trial, matrices=None):
    """Same latent Z, beta, and noise across covariances within each trial."""
    p, n = spec['p'], round(delta*spec['p'])
    code = 0 if study == 'unknown' else 1
    seed = np.random.SeedSequence([spec['seed'],code,n,p,trial])
    rng = np.random.default_rng(seed)
    truth = np.rint(n*np.asarray(spec[study+'_truth'])).astype(int)
    beta = rng.normal(size=(p,len(truth)+1))*np.sqrt(n/p)
    Z = rng.normal(size=(n,p))
    noise = rng.normal(0,spec['noise_std'],n)
    matrices = covariance_matrices(spec) if matrices is None else matrices
    X = Z @ np.linalg.cholesky(matrices[covariance]).T / np.sqrt(n)
    labels = np.searchsorted(truth,np.arange(n),side='right')
    y = np.einsum('ij,ij->i',X,beta[:,labels].T)+noise
    latent_hash = hashlib.sha256(beta.tobytes()+Z.tobytes()+noise.tobytes()).hexdigest()
    return X,y,truth,latent_hash


def least_squares(X,y,w,*,loss_scale=2.):
    fit = fit_weighted(X,y,w,loss='least_squares',loss_scale=loss_scale,
                       penalty='none',penalty_strength=0.)
    if not fit.converged or not np.isfinite(fit.coef).all():
        raise RuntimeError(f'Least-squares KKT failure: {fit.kkt_violation}')
    return fit.coef


def fixed_werm(X,y,phi,gap):
    B = np.column_stack([least_squares(X,y,w) for w in phi])
    losses = (X@B-y[:,None])**2
    cp = partition_minimize(losses,gap)
    labels = np.searchsorted(cp,np.arange(len(y)),side='right')
    return dict(cp=np.asarray(cp,dtype=int), count=len(cp),segments=len(cp)+1,
                raw_loss=float(losses[np.arange(len(y)),labels].sum()))


def penalized_werm(X,y,phi,gap,gamma):
    """Fit once with upper-bound weights, then search .5*RSS + log penalty.

    Each candidate K uses the first K columns of the same L-column estimate,
    as in E_L_upper_bound. The weights are WERM's count-mixture marginals,
    not AMP's different label-subset mixture or CV's conditional-K weights.
    """
    n = len(y)
    phi = np.asarray(phi,dtype=float)
    if (phi.ndim != 2 or phi.shape[1] != n or not np.isfinite(phi).all()
            or np.any(phi < 0) or not np.allclose(phi.sum(axis=0),1.)
            or np.any(phi.sum(axis=1) <= 0)):
        raise ValueError('Expected upper-bound occupancy probabilities with positive row masses')
    L = len(phi)
    B = np.column_stack([least_squares(X,y,w/w.sum(),loss_scale=1.) for w in phi])
    losses = .5*(X@B-y[:,None])**2
    fits = []
    for k in range(1,L+1):
        cp = partition_minimize(losses[:,:k],gap)
        labels = np.searchsorted(cp,np.arange(n),side='right')
        fits.append(dict(cp=cp,count=k-1,segments=k,
                         raw_loss=float(losses[np.arange(n),labels].sum())))
    penalties = np.asarray([gamma*(gammaln(n)-gammaln(k)-gammaln(n-k+1))
                            for k in range(1,L+1)])
    raw = np.asarray([fit['raw_loss'] for fit in fits])
    total = raw+penalties
    return dict(fits[int(total.argmin())],status='ok',candidate_raw=raw,
                candidate_penalties=penalties,candidate_total=total,
                candidate_cp=[fit['cp'] for fit in fits],
                coefficient_fit_scheme=WERM_P_VERSION,search_loss='half squared')


def fit_werm_methods(X,y,weights,spec,known=False):
    """Canonical M2 CV and single-fit upper-bound WERM+P with half-squared loss."""
    n,p = X.shape
    L,folds = spec['max_segments'],spec['folds']
    gap = minimum_gap(n,spec)
    if known:
        return {'WERM':dict(fixed_werm(X,y,weights[n,L],gap),status='ok')}
    scores = np.zeros((L,folds))
    min_refit_rows = n
    for fold in range(folds):
        test = np.arange(fold,n,folds)
        train = np.flatnonzero(np.arange(n)%folds != fold)
        m = len(train)
        for k in range(1,L+1):
            fit = fixed_werm(X[train],y[train],weights[m,k],minimum_gap(m,spec))
            cp = train[fit['cp']]
            boundaries = np.r_[0,cp,n]
            B = []
            for start,stop in zip(boundaries[:-1],boundaries[1:]):
                rows = train[(train>=start)&(train<stop)]
                min_refit_rows = min(min_refit_rows,len(rows))
                B.append(least_squares(X[rows],y[rows],np.ones(len(rows))))
            test_labels = np.searchsorted(cp,test,side='right')
            prediction = np.einsum('ij,ij->i',X[test],np.asarray(B)[test_labels])
            scores[k-1,fold] = np.abs(y[test]-prediction).sum()
    full = [fixed_werm(X,y,weights[n,k],gap) for k in range(1,L+1)]
    totals = scores.sum(axis=1)
    if not np.isfinite(totals).all():
        raise RuntimeError('Nonfinite CV scores')
    a = int(totals.argmin())
    return {'WERM+CV':dict(full[a],status='ok',cv_scores=totals/n,
                         fold_scores=scores,min_refit_rows=min_refit_rows),
            'WERM+P':penalized_werm(X,y,weights[n,'werm_upper'],gap,spec['gamma'])}


def add_metrics(fit,truth,n):
    fit = dict(fit)
    if fit.get('status') != 'ok':
        fit.update(count=np.nan,haus=np.nan,empty=False,correct_count=False)
        return fit
    cp = np.asarray(fit['cp'],dtype=int)
    assert np.all((cp>0)&(cp<n)) and np.all(np.diff(cp)>0)
    fit.update(cp=cp,count=len(cp),empty=len(cp)==0,correct_count=len(cp)==len(truth))
    if len(cp):
        distances = np.abs(cp[:,None]-truth[None,:])/n
        fit['haus'] = float(max(distances.min(axis=0).max(),distances.min(axis=1).max()))
    else:
        fit['haus'] = np.nan
    # Secondary fully-defined diagnostic: unsuccessful or empty outputs cost 1.
    fit['haus_with_empty_one'] = fit['haus'] if len(cp) else 1.
    return fit


def _record_path(root,study,covariance,delta,trial):
    return Path(root)/f'{study}__{covariance}__d{delta:g}__t{trial:03d}.pkl'


def run_trial(root,study,covariance,delta,trial):
    root = Path(root)
    path = _record_path(root,study,covariance,delta,trial)
    if path.exists():
        return str(path)
    spec = json.loads((root/'specification.json').read_text())
    weights = pickle.loads((root/'weights.pkl').read_bytes())
    matrices = pickle.loads((root/'covariances.pkl').read_bytes())
    X,y,truth,latent_hash = generate_data(spec,study,covariance,delta,trial,matrices)
    data_hash = hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
    n,p = X.shape
    record = dict(study=study,covariance=covariance,delta=delta,trial=trial,
                  n=n,p=p,truth=truth,latent_hash=latent_hash,data_hash=data_hash,
                  spec_digest=root.name,methods={})
    started = time.perf_counter()
    try:
        werm = fit_werm_methods(X,y,weights,spec,known=study=='known')
    except Exception:
        methods = ['WERM'] if study=='known' else ['WERM+CV','WERM+P']
        werm = {key:dict(status='failed',error=traceback.format_exc()) for key in methods}
    record['werm_seconds'] = time.perf_counter()-started
    record['methods'].update(werm)
    amp_started = time.perf_counter()
    try:
        from tools.werm_amp_adapter import fit_amp
        seed = int(np.random.SeedSequence([spec['seed'],n,p,trial,
                         1801 if study=='unknown' else 1802]).generate_state(1)[0])
        phi = weights[n,spec['max_segments']] if study=='known' else weights[n,'mixture']
        with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            amp = fit_amp(X,y,delta=n/p,noise_std=spec['noise_std'],
                prior_cov=(n/p)*np.eye(spec['max_segments']),phi=phi,
                gap=minimum_gap(n,spec),seed=seed,iterations=spec['amp_iterations'],
                mc_samples=spec['amp_mc_samples'],fixed_count=study=='known')
    except Exception:
        amp = dict(status='failed',error=traceback.format_exc())
    record['methods']['AMP'] = amp
    record['amp_seconds'] = time.perf_counter()-amp_started
    record['methods'] = {method:add_metrics(fit,truth,n) for method,fit in record['methods'].items()}
    assert hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest() == data_hash
    record['seconds'] = time.perf_counter()-started
    atomic_pickle(path,record)
    return str(path)


def load_results(root):
    root = Path(root)
    spec = json.loads((root/'specification.json').read_text())
    records = [pickle.loads(p.read_bytes()) for p in sorted(root.glob('*__t*.pkl'))]
    return dict(specification=spec,records=records,root=str(root))


def status(root):
    payload = load_results(root)
    cfg = payload['specification']
    expected = cfg['trials']*len(cfg['studies'])*len(cfg['covariances'])*len(cfg['deltas'])
    return dict(completed=len(payload['records']),expected=expected,
                complete=len(payload['records'])==expected,
                failed_fits=sum(f['status']!='ok' for r in payload['records'] for f in r['methods'].values()))


def verify_sources(spec):
    """Never append trials from a changed implementation to an existing run."""
    for base,key in [(ROOT,'sources_sha256'),(ROOT.parent/'AMP_chgpt_reg','amp_sources_sha256')]:
        for name,digest in spec[key].items():
            path = base/name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Source changed since this run was prepared: {path}. '
                                   'Call prepare() to create a new result directory.')


def summarize(payload):
    rows = []
    for record in payload['records']:
        for name,fit in record['methods'].items():
            rows.append(dict(study=record['study'],covariance=record['covariance'],
                delta=record['delta'],method=name,success=fit['status']=='ok',
                count=fit['count'],haus=fit['haus'],empty=fit['empty'],
                correct_count=fit['correct_count'],
                haus_all=fit.get('haus_with_empty_one',1.)))
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    return data.groupby(['study','covariance','delta','method'],sort=False).agg(
        trials=('success','size'),successes=('success','sum'),
        failure_rate=('success',lambda a:1-a.mean()),empty_rate=('empty','mean'),
        mean_haus=('haus','mean'),median_haus=('haus','median'),
        mean_haus_empty_or_failure_one=('haus_all','mean'),
        mean_count=('count','mean'),correct_count_rate=('correct_count','mean')).reset_index()


def _worker_init():
    # Restrict JAX's CPU thread pools before the first JAX import. BLAS also uses 1.
    if hasattr(os,'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        worker = multiprocessing.current_process()._identity
        index = (worker[0]-1 if worker else 0)%len(cpus)
        os.sched_setaffinity(0,{cpus[index]})


def run(root,jobs=4):
    root = Path(root)
    cfg = json.loads((root/'specification.json').read_text())
    verify_sources(cfg)
    for record in load_results(root)['records']:
        if record['spec_digest']!=root.name:
            raise RuntimeError('A checkpoint belongs to a different specification')
    tasks = [(str(root),s,c,d,t) for t in range(cfg['trials']) for d in cfg['deltas']
             for s in cfg['studies'] for c in cfg['covariances']
             if not _record_path(root,s,c,d,t).exists()]
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    print(f'Results: {root}; remaining datasets: {len(tasks)}; workers: {jobs}',flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init) as pool:
        futures = [pool.submit(run_trial,*task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            filename = future.result()
            record = pickle.loads(Path(filename).read_bytes())
            failures = {k:v.get('error',v.get('status')) for k,v in record['methods'].items()
                        if v['status']!='ok'}
            print(f'{i}/{len(tasks)} {Path(filename).stem}: {record["seconds"]:.1f}s'
                  +(f' failures={failures}' if failures else ''),flush=True)
            if i%24==0 or i==len(tasks):
                payload = load_results(root)
                summarize(payload).to_csv(root/'summary.csv',index=False)
                (root/'status.json').write_text(json.dumps(status(root),indent=2)+'\n')
    payload = load_results(root)
    atomic_pickle(root/'study.pkl',payload)
    summarize(payload).to_csv(root/'summary.csv',index=False)
    (root/'status.json').write_text(json.dumps(status(root),indent=2)+'\n')
    print(f'Completed in {time.perf_counter()-started:.1f}s',flush=True)
    return payload


def plot_study(payload,study,output_dir):
    """Canonical colored means/quartiles and pooled-location histogram grids."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    cfg = payload['specification']
    methods = ['WERM+CV','WERM+P','AMP'] if study=='unknown' else ['WERM','AMP']
    deltas = cfg['deltas']
    records = [r for r in payload['records'] if r['study']==study]
    if not records:
        return []
    style = {'font.family':'serif','font.size':11,'axes.labelsize':13,
             'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,
             'axes.linewidth':1.1,'text.usetex':True,'savefig.dpi':180}
    paths = []
    def save(fig,name):
        for ext in ('pdf','png'):
            fig.savefig(output_dir/f'{name}.{ext}',bbox_inches='tight')
        paths.append(output_dir/f'{name}.png')
        plt.close(fig)
    with plt.rc_context(style):
        for covariance in cfg['covariances']:
            selected = [r for r in records if r['covariance']==covariance]
            if not selected:
                continue
            groups = {d:[r for r in selected if r['delta']==d] for d in deltas}
            fig,axes = plt.subplots(1,2,figsize=(8.0,3.0))
            for ax,key in zip(axes,['haus','count']):
                for method in methods:
                    color,marker,line = METHOD_STYLES[method]
                    xs,means,lower,upper = [],[],[],[]
                    for delta in deltas:
                        vals = np.array([r['methods'][method][key] for r in groups[delta]
                                         if r['methods'][method]['status']=='ok'],dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if len(vals):
                            xs.append(delta);means.append(vals.mean())
                            q1,q3 = np.quantile(vals,[.25,.75])
                            lower.append(q1);upper.append(q3)
                    ax.plot(xs,means,label=method,color=color,marker=marker,ls=line,
                            lw=2,ms=6,markeredgecolor='white',markeredgewidth=.8)
                    # Quartiles are drawn at their actual positions, even if the
                    # mean lies outside [q25,q75] for a skewed/discrete sample.
                    ax.vlines(xs,lower,upper,color=color,alpha=.6,lw=1.3)
                    for x,lo,hi in zip(xs,lower,upper):
                        ax.hlines([lo,hi],x-.07,x+.07,color=color,lw=1.3)
                ax.set_xlabel(r'$\delta=n/p$')
                ax.set_xticks(deltas)
                ax.yaxis.set_major_locator(MaxNLocator(5,integer=key=='count'))
            axes[0].set_ylabel('Hausdorff error')
            axes[1].set_ylabel('No. change points')
            axes[1].axhline(len(cfg[study+'_truth']),color='gray',lw=1,ls=':')
            axes[1].set_ylim(-.15,cfg['max_segments']-.65)
            axes[1].legend(loc='best',frameon=True,framealpha=.9)
            title = COVARIANCE_TITLES[covariance]
            total = len(selected)
            bad = sum(f['status']!='ok' for r in selected for f in r['methods'].values())
            empty = sum(f.get('empty',False) for r in selected for f in r['methods'].values())
            fig.suptitle(title)
            fig.text(.5,-.035,f'{total} paired datasets; {bad} failed fits; {empty} empty estimates. '
                     'Hausdorff error conditional on a nonempty, successful fit.',
                     ha='center',fontsize=8)
            fig.tight_layout()
            save(fig,f'{study}_{covariance}_metrics')
            fig,axes = plt.subplots(len(deltas),len(methods),
                figsize=(2.4*len(methods),1.65*len(deltas)),sharex=True,squeeze=False)
            for row,delta in enumerate(deltas):
                for col,method in enumerate(methods):
                    ax = axes[row,col]
                    fits = [r['methods'][method] for r in groups[delta]]
                    vals = np.concatenate([np.asarray(f['cp'])/round(cfg['p']*delta)
                         for f in fits if f['status']=='ok' and len(f['cp'])]) if any(
                         f['status']=='ok' and len(f['cp']) for f in fits) else np.array([])
                    if len(vals):
                        ax.hist(vals,bins=12,density=True,color=METHOD_STYLES[method][0],
                                alpha=.75,edgecolor='white')
                        if len(vals)>=10:
                            kde = GridSearchCV(KernelDensity(),
                                {'bandwidth':np.logspace(-2,0,50)},cv=5,n_jobs=1)
                            kde.fit(vals[:,None])
                            xx = np.linspace(0,1,200)
                            yy = np.exp(kde.best_estimator_.score_samples(xx[:,None]))
                            ax.plot(xx,yy,color='#555555',lw=1.)
                            ax.fill_between(xx,yy,color='#999999',alpha=.15)
                    else:
                        ax.text(.5,.5,'No locations',ha='center',transform=ax.transAxes,fontsize=9)
                    for cp in cfg[study+'_truth']:
                        ax.axvline(cp,color='black',ls='--',lw=1.3)
                    if row==0:ax.set_title(method)
                    if col==0:ax.set_ylabel(rf'$\delta={delta:g}$'+'\nDensity')
                    if row==len(deltas)-1:ax.set_xlabel('Fractional location')
                    ax.set_xlim(0,1)
                    ax.set_xticks(cfg[study+'_truth'])
                    ax.tick_params(labelsize=9)
                    nf = sum(f['status']!='ok' for f in fits)
                    ne = sum(f.get('empty',False) for f in fits)
                    ax.text(.98,.95,f'{len(fits)} trials; {ne} empty; {nf} failed',
                            ha='right',va='top',transform=ax.transAxes,fontsize=6.5)
            fig.suptitle(title,y=1.005)
            fig.tight_layout(h_pad=.6,w_pad=.5)
            save(fig,f'{study}_{covariance}_histograms')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=DEFAULT_SPEC['trials'])
    parser.add_argument('--p',type=int,default=DEFAULT_SPEC['p'])
    parser.add_argument('--deltas',type=float,nargs='+',default=DEFAULT_SPEC['deltas'])
    parser.add_argument('--studies',nargs='+',default=DEFAULT_SPEC['studies'])
    parser.add_argument('--covariances',nargs='+',default=DEFAULT_SPEC['covariances'])
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--refresh-notebook',action='store_true')
    args = parser.parse_args()
    spec = {k:getattr(args,k) for k in ['trials','p','deltas','studies','covariances']}
    root = args.root or prepare(spec)
    payload = load_results(root) if args.plot_only else run(root,args.jobs)
    for study in payload['specification']['studies']:
        plot_study(payload,study,root/'figures')
    print(summarize(payload).to_string(index=False),flush=True)
    if args.refresh_notebook:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpec
        path = ROOT/'WERM_AMP_comparison.ipynb'
        notebook = nbformat.read(path,as_version=4)
        os.environ['WERM_AMP_RESULTS_DIR'] = str(root)
        # Use this exact interpreter, independently of the machine's default
        # python3 kernel (which may not contain scipy/JAX). No kernel install.
        manager = KernelManager(kernel_name='python3')
        manager._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher',
            '-f','{connection_file}'],display_name='Python (weighted-erm)',language='python')
        NotebookClient(notebook,timeout=1800,km=manager,
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        tmp = path.with_suffix('.ipynb.tmp')
        nbformat.write(notebook,tmp)
        tmp.replace(path)


if __name__=='__main__':
    main()
