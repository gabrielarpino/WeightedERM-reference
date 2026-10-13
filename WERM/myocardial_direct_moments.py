"""Direct bounded covariance moment matching, without eigenvalue clipping."""
import numpy as np
from scipy.optimize import least_squares

LOWER=np.array([np.log(1e-10),np.log(1e-10),-.9999])
UPPER=np.array([np.log(100.),np.log(100.),.9999])
STARTS=tuple(np.array([np.log(v),np.log(v),r]) for v,r in ((.003,0.),(.03,.5),(.3,-.5)))


def decode(x):
    variances=np.exp(np.asarray(x)[:2]);cross=float(x[2])*np.sqrt(variances.prod())
    return np.array([[variances[0],cross],[cross,variances[1]]])


def solve_direct(state_at,observed,start,max_nfev=60):
    observed=np.asarray(observed,float);scale=max(np.linalg.norm(observed,'fro'),1e-12)
    cache={};evaluations=[]
    def state(x):
        key=tuple(x)
        if key not in cache:
            rho=decode(x);s=state_at(rho)
            moment=s.get('moment')
            if moment is None:moment=s['lambda'].T@rho@s['lambda']+s['kappa']
            if not np.isfinite(moment).all():raise FloatingPointError('nonfinite model moment')
            cache[key]=(np.asarray(moment),float(s.get('training_residual',0.)))
        return cache[key]
    def fun(x):
        moment,inner=state(x);D=moment-observed
        r=np.array([D[0,0],np.sqrt(2)*D[0,1],D[1,1]])/scale
        evaluations.append(dict(parameters=x.copy(),relative_residual=float(np.linalg.norm(r)),inner_residual=inner))
        return r
    def jac(x):
        # Deliberate finite-difference scale larger than nested solver noise.
        out=[]
        for j in range(3):
            left=x.copy();right=x.copy()
            left[j]=max(LOWER[j],x[j]-1e-3);right[j]=min(UPPER[j],x[j]+1e-3)
            out.append((fun(right)-fun(left))/(right[j]-left[j]))
        return np.column_stack(out)
    fit=least_squares(fun,np.asarray(start),jac=jac,bounds=(LOWER,UPPER),
        max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-6)
    moment,inner=state(fit.x);residual=np.linalg.norm(moment-observed,'fro')
    near=np.minimum(fit.x-LOWER,UPPER-fit.x)<.02
    return dict(rho=decode(fit.x),parameters=fit.x,start=np.asarray(start),
        moment_residual=float(residual),relative_residual=float(residual/scale),
        inner_residual=inner,optimizer_success=bool(fit.success),message=fit.message,
        optimality=float(fit.optimality),nfev=fit.nfev,state_evaluations=len(cache),
        boundary=bool(near.any()),boundary_parameters=near.tolist(),trace=evaluations)
