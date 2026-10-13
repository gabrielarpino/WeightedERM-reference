"""Exploratory finite-accuracy solvers; residuals are diagnostics, not gates."""
import numpy as np
from scipy.optimize import least_squares, minimize, brentq
from scipy.special import expit, ndtri
from scipy.stats import qmc
from .signal_strength import signal_strength_certificate, logistic_slope, solve_b


def approximate_rho(moment, phi, psi, delta, correlation_margin=1e-4):
    cert = signal_strength_certificate(moment, phi, psi, delta)
    tri = np.tril_indices(2)
    scale = np.array([1., np.sqrt(2), 1.])
    A = cert['operator'] * scale[:, None]
    target = (moment-cert['offset'])[tri]*scale
    cap = 1/(2*np.pi)-1e-6
    def fun(c): return np.sum((A@c-target)**2)
    def jac(c): return 2*A.T@(A@c-target)
    factor = (1-correlation_margin)**2
    constraint = dict(type='ineq', fun=lambda c: factor*c[0]*c[2]-c[1]**2,
                      jac=lambda c: np.array([factor*c[2],-2*c[1],factor*c[0]]))
    fit = minimize(fun, [.02,0,.02], jac=jac, method='SLSQP',
                   bounds=[(1e-8,cap),(-cap,cap),(1e-8,cap)],
                   constraints=[constraint], options=dict(ftol=1e-15,maxiter=1000))
    c=fit.x
    # Only correct roundoff in the constrained solve, not the fitted moment.
    c[1]=np.clip(c[1],-(1-correlation_margin)*np.sqrt(c[0]*c[2]),
                (1-correlation_margin)*np.sqrt(c[0]*c[2]))
    C=np.array([[c[0],c[1]],[c[1],c[2]]])
    variances=[]
    for value in np.diag(C):
        f=lambda v:v*logistic_slope(v)**2-value
        hi=1.
        while f(hi)<0: hi*=2
        variances.append(brentq(f,0,hi,xtol=1e-12))
    h=np.array([logistic_slope(v) for v in variances])
    residual=float(np.sqrt(fun(c)))
    return dict(rho=C/h[:,None]/h[None,:],residual_norm=residual,
                optimizer_success=bool(fit.success),exact_moment_match=residual<1e-6,
                correlation_margin=correlation_margin,certificate=cert)


def logistic_prox(v,a,y):
    v,a=np.broadcast_arrays(v,a)
    lo=v-a*(1-y);hi=v+a*y
    t=np.clip(v,lo,hi)
    for _ in range(60):
        p=expit(t);f=t+a*(p-y)-v
        if np.max(np.abs(f))<5e-14: break
        lo=np.where(f<0,t,lo);hi=np.where(f>0,t,hi)
        new=t-f/(1+a*p*(1-p))
        safe=(new>=lo+.05*(hi-lo))&(new<=hi-.05*(hi-lo))
        t=np.where(np.abs(f)<5e-14,t,np.where(safe,new,(lo+hi)/2))
    return t


def logistic_state(rho,phi,psi,delta,sample_power=8,seed=42,max_nfev=150,
                   initial=None):
    """QMC fixed point with Bernoulli noise integrated, Stein cross equation.

    Deliberately returns finite-iteration estimates even when not converged.
    The residual is for the sampled equations, not a population accuracy claim.
    """
    groups,counts=np.unique(np.column_stack([phi.T,psi]),axis=0,return_counts=True)
    weights=counts/counts.sum(); labels=groups[:,2].astype(int)
    normals=ndtri(qmc.Sobol(4,scramble=True,seed=seed).random_base2(sample_power))
    Z=normals[:,:2]@np.linalg.cholesky(rho).T
    prob=expit(Z[:,labels]);dp=prob*(1-prob)
    def unpack(x):
        b=np.exp(x[:2]);lam=x[2:6].reshape(2,2)
        chol=np.array([[np.exp(x[6]),0],[x[7],np.exp(x[8])]])
        return b,lam,chol
    def average(x): return np.einsum('sg...,g->...',x,weights)/len(Z)
    def residual(x):
        b,lam,chol=unpack(x);K=chol@chol.T
        V=(Z@lam+normals[:,2:]@chol.T)[:,None,:]
        a=(groups[:,:2]*b)[None,:,:]
        t0=logistic_prox(V,a,0);t1=logistic_prox(V,a,1)
        p0=expit(t0);p1=expit(t1)
        w0=a*p0;w1=a*(p1-1)
        d0=a*p0*(1-p0);d1=a*p1*(1-p1)
        q=prob[:,:,None]
        derivative=average((1-q)*d0/(1+d0)+q*d1/(1+d1))
        cross=np.array([average(dp[:,:,None]*(w1-w0)*(labels==s)[None,:,None]) for s in range(2)])
        cov=delta*average((1-prob)[:,:,None,None]*w0[:,:,:,None]*w0[:,:,None,:]
                          +prob[:,:,None,None]*w1[:,:,:,None]*w1[:,:,None,:])
        return np.r_[derivative-1/delta,(lam*derivative+cross).ravel(),
                     (K-cov)[np.tril_indices(2)]]
    if initial is None:
        b=[solve_b(.25*w,delta) for w in phi]
        initial=np.r_[np.log(b),np.eye(2).ravel(),np.log(.2),0,np.log(.2)]
    fit=least_squares(residual,initial,max_nfev=max_nfev,
                      ftol=1e-9,xtol=1e-9,gtol=1e-9)
    b,lam,chol=unpack(fit.x)
    return dict(b=b,**{'lambda':lam},kappa=chol@chol.T,
                training_residual=float(np.linalg.norm(fit.fun)),
                optimizer_success=bool(fit.success),nfev=fit.nfev,parameters=fit.x,
                sample_power=sample_power,seed=seed)
