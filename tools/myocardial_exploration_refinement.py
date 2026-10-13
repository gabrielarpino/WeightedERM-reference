"""Declared numerical refinements and clearly separated scoring diagnostics."""
from tools.myocardial_exploration import *
from scipy.stats import multivariate_normal
from scipy.special import ndtr, log_ndtr


def evaluate_refined(*args):
    import WERM.myocardial_approximate as implementation
    from WERM.myocardial_numerics import logistic_prox_fast
    original=implementation.logistic_prox
    implementation.logistic_prox=logistic_prox_fast
    try:return evaluate_scenario(*args)
    finally:implementation.logistic_prox=original


def compare_scores(root):
    """Compare joint vs conditional-observation scores without changing the fit.

    Conditional-only normalization is a diagnostic pseudo-posterior, not the
    audited joint posterior. It does not repair the original covariance error.
    """
    X,y,ages,locations=load_data();rows=[]
    for path in sorted(root.glob('d*_q7_b40.pkl')):
        rec=pickle.load(path.open('rb'))
        setting=pickle.load((root/f"fits_delta{rec['delta_min']}.pkl").open('rb'))
        for name,fit in setting['fits'].items():
            joint=rec['results'][name]['log_likelihood'];normal=[]
            for state in rec['states']:
                cov=state['lambda'].T@rec['scenario']['rho']@state['lambda']+state['kappa']
                normal.append(multivariate_normal.logpdf(fit['V'],mean=np.zeros(2),cov=cov).sum())
            conditional=joint-np.array(normal)
            logweights=conditional+np.log(setting['prior'])
            probabilities=np.exp(logweights-logsumexp(logweights))
            curve=np.exp(conditional[1:]-logsumexp(conditional[1:]))
            rows.append(dict(setting=rec['label'],posterior_fit=name,
                joint_P_1=rec['results'][name]['presence'],conditional_only_P_1=probabilities[1:].sum(),
                conditional_only_peak_age=ages[rec['eta'][1:][np.argmax(curve)]]))
    pd.DataFrame(rows).to_csv(root/'conditional_only_diagnostic.csv',index=False)


def refine(root):
    X,y,ages,locations=load_data()
    settings=[pickle.load((root/f'fits_delta{d}.pkl').open('rb')) for d in DELTAS]
    # Declared independently of the desired posterior: all auxiliary fits at
    # 3 and 30 rho updates for Delta=1, plus the joint LS fit at every Delta.
    selected=[(s,r) for s in settings for r in s['scenarios']
              if r['iteration']==-1 or (s['delta_min']==1 and
                  (r['iteration'] in (3,30) or r['auxiliary']=='least_squares'))]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=26)(delayed(evaluate_refined)(s,r,X,y,ages,root,power,budget)
                  for s,r in selected for power,budget in [(7,10),(9,100)])
    rows=[]
    for rec in results:
        base=pickle.load((root/(rec['label'].split('_q')[0]+'_q7_b40.pkl')).open('rb'))
        for name,stats in rec['results'].items():
            old=base['results'][name]
            rows.append(dict(setting=rec['label'],posterior_fit=name,P_1=stats['presence'],
                coarse_P_1=old['presence'],delta_P_1=stats['presence']-old['presence'],
                mass_above55=stats['conditional'][ages[rec['eta'][1:]]>55].sum(),
                coarse_mass_above55=old['conditional'][ages[rec['eta'][1:]]>55].sum(),
                TV_from_coarse=.5*np.abs(stats['conditional']-old['conditional']).sum(),
                peak_age=stats['peak_age'],FP_residual_max=max(rec['residuals'])))
    pd.DataFrame(rows).to_csv(root/'numerical_refinement.csv',index=False)
    quadrature=[]
    for rec in results:
        if not rec['label'].endswith('_q9_b100'):continue
        setting=next(s for s in settings if s['delta_min']==rec['delta_min'])
        for name,fit in setting['fits'].items():
            values=[]
            for e,state in zip(rec['eta'],rec['states']):
                psi=np.zeros(len(y),int) if e<0 else (np.arange(len(y))>=e).astype(int)
                values.append(gaussian_logistic_loglikelihood(fit['V'],y,rec['scenario']['rho'],
                    state['lambda'],state['kappa'],psi,quadrature_order=160))
            values=np.asarray(values);joint=values+np.log(setting['prior'])
            probability=np.exp(joint-logsumexp(joint))
            curve=np.exp(values[1:]-logsumexp(values[1:]))
            old=rec['results'][name]
            quadrature.append(dict(setting=rec['label'],posterior_fit=name,
                P_1_80=old['presence'],P_1_160=probability[1:].sum(),
                TV_80_to_160=.5*np.abs(curve-old['conditional']).sum(),
                max_loglikelihood_difference=np.max(np.abs(values-old['log_likelihood']))))
    pd.DataFrame(quadrature).to_csv(root/'quadrature_check.csv',index=False)
    compare_scores(root)
    print('Numerical refinements and likelihood decomposition complete.',flush=True)


if __name__=='__main__':
    import sys
    refine(Path(sys.argv[1]))
