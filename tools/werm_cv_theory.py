"""First-order SE plug-in prediction for squared-loss WERM-CV.

This is a composition extension requiring stable limiting partitions/refits and
separated CV optima; it is not a finite-sample joint law for overlapping folds.
No empirical fits, validation responses, or selected counts enter this module.
"""
import numpy as np
from tools.el_state_evolution import squared_se
from tools.m2_adjusted_experiment import squared_partition
from tools.werm_cv_empirical import minimum_gap, get_weights
from tools.werm_cv_partition import additive_partition


def expected_partition(phi, gamma, labels, delta, noise_variance, gap, end_min=1):
    state = squared_se(phi,gamma,labels,delta,noise_variance)
    d = state['lam'][None,:,:] - np.eye(len(gamma))[labels,:,None]
    bias = np.einsum('nrl,rs,nsl->nl',d,gamma,d)
    profile = (1-state['omega'])**2*(noise_variance+bias+np.diag(state['kappa']))
    if np.any(profile < -1e-12):
        raise ValueError('Negative expected residual variance')
    cp = additive_partition(profile,gap,end_min)
    return dict(cp=cp,state=state,profile=profile)


def uniform_refit_risk(gamma, labels, train, test, training_cp, p, noise_variance):
    labels,train,test = np.asarray(labels),np.asarray(train),np.asarray(test)
    mapped = train[np.asarray(training_cp,dtype=int)]
    boundaries = np.r_[0,mapped,len(labels)]
    risks=[];sizes=[];score=0.
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        rows=train[(train>=start)&(train<stop)]
        validation=test[(test>=start)&(test<stop)]
        ratio=len(rows)/p
        if ratio<=1:
            raise ValueError(f'Uniform-refit SE requires m_s/p>1, got {ratio:g}')
        q=np.bincount(labels[rows],minlength=len(gamma))/len(rows)
        mixture_variance=float(q@np.diag(gamma)-q@gamma@q)
        kappa=(noise_variance+mixture_variance)/(ratio-1)
        d=q[None,:]-np.eye(len(gamma))
        variance=noise_variance+np.einsum('ir,rs,is->i',d,gamma,d)+kappa
        risk=np.sqrt(2/np.pi)*np.sqrt(variance)
        score+=float(risk[labels[validation]].sum())
        sizes.append(len(rows))
        risks.append(dict(q=q,kappa=kappa,risk_by_regime=risk,local_delta=ratio))
    return dict(score=score,mean_score=score/len(test),segment_sizes=np.asarray(sizes),
                segment_ratios=np.asarray(sizes)/p,segment_risks=risks,mapped_cp=mapped)


def predict_cv(gamma, labels, p, noise_variance, max_segments, weights,
               delta_fraction, folds=10, minimum_segment_fraction=None):
    n=len(labels);indices=np.arange(n);scores=np.zeros((max_segments,folds))
    cache={};diagnostics={}
    for count in range(1,max_segments+1):
        for fold in range(folds):
            train=indices[indices%folds!=fold];test=indices[fold::folds]
            train_labels=labels[train]
            key=(count,len(train),train_labels.tobytes())
            if key not in cache:
                gap=minimum_gap(len(train),delta_fraction)
                end_min=1 if minimum_segment_fraction is None else int(np.ceil(minimum_segment_fraction*len(train)))
                gap=max(gap,end_min)
                phi=get_weights(weights,len(train),count,gap)
                cache[key]=expected_partition(phi,gamma,train_labels,len(train)/p,noise_variance,gap,end_min)
            fitted=cache[key]
            risk=uniform_refit_risk(gamma,labels,train,test,fitted['cp'],p,noise_variance)
            scores[count-1,fold]=risk['score']
            risk['train_cp']=fitted['cp']
            risk['se_residual']=max(fitted['state'][k] for k in
                ('b_residual','bias_residual','covariance_residual'))
            diagnostics[count,fold]=risk
    totals=scores.sum(axis=1)
    selected={L:dict(selected_segments=int(np.argmin(totals[:L]))+1)
              for L in range(1,max_segments+1)}
    return dict(fold_scores=scores,cv_scores=totals,normalized_cv_scores=totals/n,
                selected_by_bound=selected,fold_results=diagnostics)


def final_surrogates(gamma,labels,p,noise_variance,count,weights,delta_fraction,draws,rng,
                     minimum_segment_fraction=None):
    n=len(labels);gap=minimum_gap(n,delta_fraction)
    end_min=1 if minimum_segment_fraction is None else int(np.ceil(minimum_segment_fraction*n))
    gap=max(gap,end_min)
    phi=get_weights(weights,n,count,gap)
    expected=expected_partition(phi,gamma,labels,n/p,noise_variance,gap,end_min)
    state=expected['state'];fits=[]
    for _ in range(draws):
        Z=rng.normal(size=(n,len(gamma)))@np.linalg.cholesky(gamma).T
        y=Z[np.arange(n),labels]+rng.normal(0,np.sqrt(noise_variance),n)
        V=Z@state['lam']+rng.normal(size=(n,count))@state['covariance_root'].T
        theta=(1-state['omega'])*V+state['omega']*y[:,None]
        fits.append(dict(cp=additive_partition((theta-y[:,None])**2,gap,end_min)))
    return dict(final_fits=fits,limiting_cp=expected['cp'],
                se_residual=max(state[k] for k in ('b_residual','bias_residual','covariance_residual')))
