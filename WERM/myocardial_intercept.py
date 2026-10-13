"""Intercept-aware heuristic state model, distinct from the zero-mean audit.

Each latent logistic intercept is calibrated to the observed global prevalence.
Fitted intercepts are unpenalized; adjusted scores are centered by those fits.
This is an empirical-Bayes model extension, not a proof for the original theorem.
"""
import numpy as np
from scipy.optimize import brentq, minimize, least_squares
from scipy.special import expit, logit, ndtri, roots_hermitenorm, logsumexp, log_expit
from scipy.stats import qmc, multivariate_normal
from .signal_strength import solve_b, empirical_adjustment
from .myocardial_numerics import logistic_prox_fast
from .proximal import fit_weighted

NODES,WEIGHTS=roots_hermitenorm(160)
WEIGHTS=WEIGHTS/np.sqrt(2*np.pi)


def response_offset(variance,rate):
    sd=np.sqrt(variance)
    f=lambda offset:WEIGHTS@expit(offset+sd*NODES)-rate
    bound=50+10*sd
    return brentq(f,-bound,bound,xtol=1e-12)


def response_parameters(rho,rate):
    offsets=np.array([response_offset(v,rate) for v in np.diag(rho)])
    p=expit(offsets[:,None]+np.sqrt(np.diag(rho))[:,None]*NODES)
    return offsets,(p*(1-p))@WEIGHTS


def fit_with_intercept(X,y,phi,penalty,strength,loss='logistic'):
    design=np.column_stack([X,np.ones(len(y))]);columns=[];checks=[]
    for w in phi:
        if loss=='squared':
            beta=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            err=np.max(abs(design.T@(w*(design@beta-y))/w.sum()))
            checks.append(dict(kkt=float(err),success=True))
        elif penalty=='l2':
            wn=w/w.sum();mask=np.r_[np.ones(X.shape[1]),0.]
            def objective(beta):
                z=design@beta
                return float(wn@(np.logaddexp(0,z)-y*z)+strength*np.sum(mask*beta**2)),design.T@(wn*(expit(z)-y))+2*strength*mask*beta
            fit=minimize(objective,np.r_[np.zeros(X.shape[1]),logit(y.mean())],jac=True,method='L-BFGS-B',
                         options=dict(maxiter=5000,ftol=1e-14,gtol=1e-10,maxcor=40))
            beta=fit.x;checks.append(dict(kkt=float(np.max(abs(objective(beta)[1]))),success=bool(fit.success)))
        else:
            fit=fit_weighted(design,y,loss='logistic',sample_weight=w,penalty='l1',penalty_strength=strength,
                             unpenalized_indices=[X.shape[1]],max_iter=30000,tol=1e-8)
            beta=fit.coef;checks.append(dict(kkt=fit.kkt_violation,success=fit.converged))
        columns.append(beta)
    B=np.column_stack(columns);theta=design@B
    adjusted,b=empirical_adjustment(theta,y,phi,len(y)/X.shape[1],loss=loss)
    # For a centered design, the empirical mean equals the fitted intercept.
    means=X.mean(axis=0)@B[:-1]+B[-1]
    return dict(theta=theta,V=adjusted-means,uncentered_adjusted=adjusted,b=b,intercepts=B[-1],
                total_offsets=means,coef=B[:-1],checks=checks,penalty=penalty,strength=strength)


def squared_intercept_state(rho,phi,psi,delta,rate):
    offsets,h=response_parameters(rho,rate)
    b=np.array([solve_b(2*w,delta) for w in phi]);a=(2*b[:,None]*phi/(1+2*b[:,None]*phi)).T
    transfer=np.array([np.mean(a*(psi==s)[:,None],axis=0)/a.mean(axis=0) for s in range(2)])
    lam=h[:,None]*transfer;A=lam.T@rho@lam
    vy=(rho[:,psi].T*h[psi,None])@lam
    R=A[None,:,:]-vy[:,:,None]-vy[:,None,:]+rate*(1-rate)
    aa=a[:,:,None]*a[:,None,:]
    K=delta*np.mean(aa*R,axis=0)/(1-delta*aa.mean(axis=0))
    return dict(b=b,**{'lambda':lam},kappa=K,moment=A+K,response_offsets=offsets,fit_offsets=np.full(2,rate))


def logistic_intercept_state(rho,phi,psi,delta,rate,sample_power=7,max_nfev=60,seed=42):
    groups,counts=np.unique(np.column_stack([phi.T,psi]),axis=0,return_counts=True)
    weights=counts/counts.sum();labels=groups[:,2].astype(int)
    normals=ndtri(qmc.Sobol(4,scramble=True,seed=seed).random_base2(sample_power))
    Z=normals[:,:2]@np.linalg.cholesky(rho).T
    offsets,_=response_parameters(rho,rate)
    prob=expit(Z[:,labels]+offsets[labels]);dp=prob*(1-prob)
    def average(x):return np.einsum('sg...,g->...',x,weights)/len(Z)
    def unpack(x):
        b=np.exp(x[:2]);lam=x[2:6].reshape(2,2)
        chol=np.array([[np.exp(x[6]),0],[x[7],np.exp(x[8])]])
        return b,lam,chol,x[9:11]
    def residual(x):
        b,lam,chol,alpha=unpack(x);K=chol@chol.T
        V=(Z@lam+normals[:,2:]@chol.T)[:,None,:]
        a=(groups[:,:2]*b)[None,:,:]
        t0=logistic_prox_fast(V+alpha,a,0);t1=logistic_prox_fast(V+alpha,a,1)
        p0=expit(t0);p1=expit(t1);w0=a*p0;w1=a*(p1-1)
        d0=a*p0*(1-p0);d1=a*p1*(1-p1);q=prob[:,:,None]
        derivative=average((1-q)*d0/(1+d0)+q*d1/(1+d1))
        cross=np.array([average(dp[:,:,None]*(w1-w0)*(labels==s)[None,:,None]) for s in range(2)])
        cov=delta*average((1-prob)[:,:,None,None]*w0[:,:,:,None]*w0[:,:,None,:]+prob[:,:,None,None]*w1[:,:,:,None]*w1[:,:,None,:])
        return np.r_[derivative-1/delta,(lam*derivative+cross).ravel(),(K-cov)[np.tril_indices(2)],average((1-q)*w0+q*w1)]
    b=[solve_b(.25*w,delta) for w in phi]
    initial=np.r_[np.log(b),np.eye(2).ravel(),np.log(.2),0,np.log(.2),np.full(2,logit(rate))]
    fit=least_squares(residual,initial,max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    b,lam,chol,alpha=unpack(fit.x)
    return dict(b=b,**{'lambda':lam},kappa=chol@chol.T,fit_offsets=alpha,response_offsets=offsets,
                training_residual=float(np.linalg.norm(fit.fun)),parameters=fit.x,optimizer_success=bool(fit.success))


def intercept_loglikelihood(V,y,rho,state,psi,order=160):
    lam=state['lambda'];K=state['kappa'];S=lam.T@rho@lam+K;cross=rho@lam
    conditional=rho-cross@np.linalg.solve(S,cross.T)
    means=V@np.linalg.solve(S,cross.T)
    mu=means[np.arange(len(y)),psi]+state['response_offsets'][psi]
    sd=np.sqrt(np.maximum(np.diag(conditional)[psi],0))
    nodes,w=roots_hermitenorm(order)
    ll=logsumexp(log_expit((2*y[:,None]-1)*(mu[:,None]+sd[:,None]*nodes))+np.log(w/np.sqrt(2*np.pi)),axis=1).sum()
    return float(multivariate_normal.logpdf(V,mean=np.zeros(2),cov=S).sum()+ll)
