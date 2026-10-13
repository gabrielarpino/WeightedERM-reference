"""Audited JRSSB signal-strength equations; additive, not a legacy API replacement.

Notation: rho = Gamma; lam = Lambda (NOT Cov(Z,V)); V = Z @ lam + G.
Squared fitting loss and Bernoulli(sigmoid(Z_s)) observation model are distinct.
"""
import numpy as np
from scipy.optimize import brentq, least_squares, linprog
from scipy.special import expit, roots_hermitenorm, logsumexp, log_expit
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from functools import lru_cache


@lru_cache(maxsize=8)
def _quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2*np.pi)


def logistic_slope(variance, order=80):
    """E sigmoid'(N(0,variance)); stable also for broad Gaussian signals."""
    if variance < 0 or not np.isfinite(variance): raise ValueError('Invalid variance')
    if variance <= 4:
        nodes,w=_quadrature(order)
        t=expit(np.sqrt(variance)*nodes)
        return float((t*(1-t))@w)
    # Integrate on the logistic, not the very broad Gaussian, length scale.
    return quad(lambda x:np.exp(-x*x/(2*variance))*expit(x)*expit(-x),
                -45,45,epsabs=1e-12,epsrel=1e-12)[0]/np.sqrt(2*np.pi*variance)


def configuration_prior(n, locations):
    locations = np.asarray(locations, dtype=int)
    if locations.ndim != 1 or not len(locations) or np.any((locations <= 0) | (locations >= n)) or len(np.unique(locations)) != len(locations):
        raise ValueError('Locations must be unique strict split indices')
    eta = np.r_[-1, np.sort(locations)]
    prior = np.r_[.5, np.full(len(locations), .5/len(locations))]
    second = np.mean(np.arange(n)[None,:] >= eta[1:,None], axis=0)/2
    return eta, prior, np.array([1-second, second])


def _validate(phi, delta):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or not np.isfinite(phi).all() or np.any(phi < 0) or not np.isfinite(delta) or delta <= 1:
        raise ValueError('Nonnegative finite L-by-n weights and delta > 1 required')
    return phi


def solve_b(curvature, delta):
    h = np.asarray(curvature, dtype=float)
    if not np.isfinite(h).all() or np.any(h < 0) or np.mean(h == 0) >= 1-1/delta:
        raise ValueError('No finite positive empirical b root for these curvatures')
    fun = lambda b: np.mean(1/(1+b*h)) - (1-1/delta)
    upper = 1.
    while fun(upper) > 0:
        upper *= 2
        if upper > 1e16: raise ValueError('Could not bracket b')
    return brentq(fun, 0, upper, xtol=1e-13, rtol=1e-13)


def empirical_adjustment(theta, y, phi, delta, loss='squared'):
    phi = _validate(phi, delta)
    theta,y = np.asarray(theta),np.asarray(y)
    if theta.shape != phi.T.shape or y.shape != (phi.shape[1],):
        raise ValueError('Incompatible shapes')
    if loss == 'squared':
        gradient=2*(theta-y[:,None]);curvature=2*np.ones_like(theta)
    elif loss == 'logistic':
        mu=expit(theta);gradient=mu-y[:,None];curvature=mu*(1-mu)
    else: raise ValueError('Unknown loss')
    b=np.array([solve_b(phi[l]*curvature[:,l],delta) for l in range(len(phi))])
    return theta+phi.T*gradient*b, b


def squared_logistic_state(rho, phi, psi, delta, quadrature_order=80):
    """Eliminate b, Lambda, K analytically except 1D Gaussian quadrature.

    a_il=2*b_l*phi_li/(1+2*b_l*phi_li); mean_i a_il=1/delta.
    h_s=E sigmoid'(N(0,rho_ss)), Lambda_sl=h_s mean(a_l 1_{psi=s})/mean(a_l).
    K_lm=delta mean(a_l a_m R_i,lm)/(1-delta mean(a_l a_m)).
    R=E[(Z Lambda_l-Y)(Z Lambda_m-Y)] and E[Y^2]=1/2.
    """
    phi=_validate(phi,delta);rho=np.asarray(rho,dtype=float);psi=np.asarray(psi,dtype=int)
    L,n=phi.shape
    if rho.shape!=(L,L) or not np.allclose(rho,rho.T) or np.linalg.eigvalsh(rho).min() <= 0:
        raise ValueError('rho must be symmetric positive definite')
    if psi.shape!=(n,) or np.any((psi<0)|(psi>=L)):raise ValueError('Invalid signal labels')
    b=np.array([solve_b(2*w,delta) for w in phi])
    a=(2*b[:,None]*phi/(1+2*b[:,None]*phi)).T
    h=np.array([logistic_slope(v,quadrature_order) for v in np.diag(rho)])
    transfer=np.array([np.mean(a*(psi==s)[:,None],axis=0)/np.mean(a,axis=0) for s in range(L)])
    lam=h[:,None]*transfer
    signal_moment=lam.T@rho@lam
    zy=rho[:,psi].T*h[psi,None]
    vy=zy@lam
    R=signal_moment[None,:,:]-vy[:,:,None]-vy[:,None,:]+.5
    aa=a[:,:,None]*a[:,None,:]
    denominator=1-delta*aa.mean(axis=0)
    if denominator.min() <= 0:raise ValueError('Nonpositive covariance denominator')
    kappa=delta*np.mean(aa*R,axis=0)/denominator
    return dict(b=b, **{'lambda':lam}, kappa=kappa, moment=signal_moment+kappa, h=h)


def _rho_from_params(x):
    sd=np.exp(x[:2]);corr=x[2]
    return np.array([[sd[0]**2, corr*sd.prod()],[corr*sd.prod(),sd[1]**2]])


def signal_strength_certificate(moment, phi, psi, delta):
    """Linear identification of C=diag(h) rho diag(h), which must be PSD.

    For full-rank operator, a negative eigenvalue of the unique C is a
    certificate that no positive-semidefinite rho solves the moment equations.
    For rank-deficient operators we report nonidentification, not infeasibility.
    """
    phi=_validate(phi,delta);psi=np.asarray(psi,dtype=int);S=np.asarray(moment,dtype=float)
    if phi.shape[0]!=2 or S.shape!=(2,2):raise ValueError('L=2 required')
    if psi.shape!=(phi.shape[1],) or np.any((psi<0)|(psi>1)):raise ValueError('Invalid signal labels')
    b=np.array([solve_b(2*w,delta) for w in phi])
    a=(2*b[:,None]*phi/(1+2*b[:,None]*phi)).T
    T=np.array([np.mean(a*(psi==s)[:,None],axis=0)/a.mean(axis=0) for s in range(2)])
    aa=a[:,:,None]*a[:,None,:];D=delta*aa.mean(axis=0)
    if np.any(D>=1):raise ValueError('No finite covariance denominator')
    def forward(C):
        v=C[:,psi].T@T
        return (T.T@C@T-delta*np.mean(aa*(v[:,:,None]+v[:,None,:]),axis=0)+.5*D)/(1-D)
    base=forward(np.zeros((2,2)));tri=np.tril_indices(2)
    bases=np.array([[[1,0],[0,0]],[[0,1],[1,0]],[[0,0],[0,1]]])
    operator=np.column_stack([(forward(C)-base)[tri] for C in bases])
    coefs,_,rank,singular=np.linalg.lstsq(operator,(S-base)[tri],rcond=None)
    C=np.einsum('i,ijk->jk',coefs,bases);eig=np.linalg.eigvalsh(C)
    feasible=rank==3 and eig.min()>0 and np.max(np.diag(C))<1/(2*np.pi)
    return dict(effective_covariance=C,effective_eigenvalues=eig,operator=operator,offset=base,
                operator_rank=int(rank),operator_singular_values=singular,
                operator_condition=float(np.linalg.cond(operator)),
                linear_residual=float(np.linalg.norm(forward(C)-S)),feasible=bool(feasible),
                infeasibility_certified=bool(rank==3 and (eig.min() < -1e-10 or np.max(np.diag(C))>=1/(2*np.pi))))


def estimate_signal_strength(moment, phi, psi, delta, tolerance=1e-7, starts=None):
    """Fit all three entries of a 2x2 Gamma jointly; never freeze pilot diagonals.

    Acceptance requires moment residual, interior solution and full local rank,
    not merely scipy's optimization-success flag. Bounds diagnose saturation.
    """
    phi=_validate(phi,delta);S=np.asarray(moment,dtype=float)
    if phi.shape[0]!=2 or S.shape!=(2,2):raise ValueError('This inverse solver supports L=2')
    if len(np.unique(psi))!=2:raise ValueError('Both signal labels are required to identify the full matrix')
    if not np.allclose(S,S.T) or np.linalg.eigvalsh(S).min()<=0:raise ValueError('Moment must be SPD')
    tri=np.tril_indices(2);scale=np.array([1,np.sqrt(2),1])
    certificate=signal_strength_certificate(S,phi,psi,delta)
    def residual(x):
        rho=_rho_from_params(x)
        h=np.array([logistic_slope(v) for v in np.diag(rho)])
        C=h[:,None]*rho*h[None,:]
        return (certificate['operator']@C[tri]+certificate['offset'][tri]-S[tri])*scale
    if certificate['feasible']:
        C=certificate['effective_covariance']
        variances=[]
        for value in np.diag(C):
            fun=lambda v:v*logistic_slope(v)**2-value
            hi=1.
            while fun(hi)<0:
                hi*=2
                if hi>1e12:raise ValueError('Signal strength too saturated to identify numerically')
            variances.append(brentq(fun,0,hi,xtol=1e-12))
        h=np.array([logistic_slope(v) for v in variances]);rho=C/h[:,None]/h[None,:]
        # The analytic inverse supplies an exact initialization, verified below.
        starts=[np.r_[np.log(np.sqrt(variances)),rho[0,1]/np.sqrt(rho[0,0]*rho[1,1])]]
    if starts is None:starts=[np.array([a,b,c]) for a,b in [(-1,-1),(0,0),(1,1),(-1,1),(1,-1)] for c in [-.7,.7]]
    solutions=[least_squares(residual,x,bounds=([-14,-14,-1+1e-12],[14,14,1-1e-12]),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=2000,x_scale='jac') for x in starts]
    best=min(solutions,key=lambda r:np.linalg.norm(r.fun))
    rho=_rho_from_params(best.x);state=squared_logistic_state(rho,phi,psi,delta)
    singular=np.linalg.svd(best.jac,compute_uv=False);rank=int(np.sum(singular>max(singular[0]*1e-6,1e-10)))
    residual_norm=float(np.linalg.norm(best.fun))
    boundary=bool(np.any(np.abs(best.active_mask)>0) or abs(best.x[2])>.99999)
    return dict(rho=rho,state=state,accepted=bool(certificate['feasible'] and best.success and residual_norm<tolerance and rank==3 and not boundary),
                residual_norm=residual_norm,jacobian_rank=rank,jacobian_singular_values=singular,
                certificate=certificate,
                boundary=boundary,optimizer_success=bool(best.success),parameters=best.x,
                starts=[dict(residual=float(np.linalg.norm(r.fun)),rho=_rho_from_params(r.x),success=bool(r.success)) for r in solutions])


def gaussian_logistic_loglikelihood(V,y,rho,lam,kappa,psi,quadrature_order=80):
    """Joint log p(V,Y|psi) using V=Z Lambda+G and Gaussian quadrature.

    Includes the candidate-dependent density of V, unlike the legacy routine.
    """
    V=np.asarray(V);y=np.asarray(y);psi=np.asarray(psi,dtype=int)
    S=lam.T@rho@lam+kappa
    cross=rho@lam
    conditional=rho-cross@np.linalg.solve(S,cross.T)
    variances=np.diag(conditional)
    if variances.min() < -1e-9: raise ValueError('Invalid joint covariance')
    means=V@np.linalg.solve(S,cross.T)
    mu=means[np.arange(len(y)),psi]
    sd=np.sqrt(np.maximum(variances[psi],0))
    nodes,w=roots_hermitenorm(quadrature_order)
    log_bernoulli=logsumexp(log_expit((2*y[:,None]-1)*(mu[:,None]+sd[:,None]*nodes))+np.log(w/np.sqrt(2*np.pi)),axis=1)
    return float(multivariate_normal.logpdf(V,mean=np.zeros(V.shape[1]),cov=S).sum()+log_bernoulli.sum())


def logistic_separation(X, y, sample_weight=None):
    """Certify a logistic recession direction on positive-weight observations.

    A_i=(2*y_i-1)*X_i. Seek A d >= 0 and sum(A d) >= 1. A validated
    witness excludes a finite unpenalized MLE; LP tolerances are reported.
    """
    X=np.asarray(X,dtype=float);y=np.asarray(y)
    if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Finite design and binary targets required')
    w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,dtype=float)
    if w.shape!=y.shape or not np.isfinite(w).all() or np.any(w<0) or not np.any(w>0):raise ValueError('Invalid weights')
    active=w>0;A=(2*y[active]-1)[:,None]*X[active]
    result=linprog(np.zeros(X.shape[1]),A_ub=np.vstack([-A,-A.sum(axis=0)]),
                   b_ub=np.r_[np.zeros(len(A)),-1],bounds=[(None,None)]*X.shape[1],method='highs')
    if result.status not in (0,2):raise RuntimeError('Separation LP is inconclusive: '+result.message)
    if result.status==2:return dict(separated=False,direction=None,status=result.message)
    margins=A@result.x
    valid=bool(margins.min()>=-1e-9 and margins.sum()>=1-1e-7)
    if not valid:raise RuntimeError('LP solution did not pass the numerical witness check')
    return dict(separated=True,direction=result.x,min_margin=float(margins.min()),
                sum_margins=float(margins.sum()),max_margin=float(margins.max()),status=result.message)
