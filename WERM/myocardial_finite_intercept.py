"""Finite-budget, unpenalized logistic fits with an intercept.

The constant column has Euclidean norm one, matching the whitened slope
columns. This is a declared preconditioning choice, important for early stopping.
"""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from .proximal import fit_weighted
from .signal_strength import empirical_adjustment


def finite_intercept_column(X,y,w,method,budget):
    if method not in ('fista','gd','lbfgs'):raise ValueError('unknown optimizer')
    if isinstance(budget,bool) or int(budget)!=budget or budget<1:raise ValueError('positive integer budget required')
    X=np.asarray(X,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if not np.isfinite(X).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('invalid data/weights')
    if y.shape!=(len(X),) or w.shape!=(len(X),) or not np.isin(y,[0,1]).all():raise ValueError('invalid binary response dimensions')
    n,p=X.shape;wn=w/w.sum();scale=np.sqrt(n)
    D=np.column_stack([X,np.ones(n)/scale])
    initial=np.r_[np.zeros(p),scale*logit(np.clip(wn@y,1e-8,1-1e-8))]
    def objective(beta):
        z=D@beta
        return float(wn@(np.logaddexp(0,z)-y*z)),D.T@(wn*(expit(z)-y))
    if method=='lbfgs':
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
            options=dict(maxiter=int(budget),ftol=0.,gtol=0.,maxcor=10,maxls=20))
        beta=result.x;iterations=result.nit;success=bool(result.success)
    else:
        L=float(np.linalg.eigvalsh(D.T@(wn[:,None]*D))[-1])/4
        step=1/L
        if method=='fista':
            result=fit_weighted(D,y,loss='logistic',sample_weight=w,penalty='none',
                penalty_strength=0.,initial=initial,initial_step=step,max_iter=int(budget),tol=1e-14)
            beta=result.coef;iterations=result.n_iter;success=result.converged
        else:
            beta=initial.copy()
            for _ in range(int(budget)):beta-=step*objective(beta)[1]
            iterations=int(budget);success=np.max(abs(objective(beta)[1]))<=1e-14
    value,gradient=objective(beta);z=D@beta
    return dict(coef=beta[:-1],intercept=float(beta[-1]/scale),objective=value,
        kkt=float(np.max(abs(gradient))),intercept_score=float(wn@(expit(z)-y)),
        iterations=iterations,success=bool(success),method=method,budget=int(budget))


def finite_intercept_fit(X,y,phi,method,budget):
    fits=[finite_intercept_column(X,y,w,method,budget) for w in phi]
    B=np.column_stack([f['coef'] for f in fits]);alpha=np.array([f['intercept'] for f in fits])
    theta=X@B+alpha
    adjusted,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss='logistic')
    offsets=X.mean(axis=0)@B+alpha
    V=adjusted-offsets
    if not np.isfinite(V).all():raise FloatingPointError('Nonfinite adjusted scores')
    return dict(coef=B,intercepts=alpha,theta=theta,V=V,b=b,total_offsets=offsets,
        penalty='none',strength=0.,method=method,budget=budget,fit_intercept=True,
        checks=[{k:v for k,v in f.items() if k!='coef'} for f in fits])
