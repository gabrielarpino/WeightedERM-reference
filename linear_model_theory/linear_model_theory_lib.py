import numpy as np
from scipy.optimize import brentq
from scipy.special import comb
def weights(n,L):
    i=np.arange(1,n+1)
    return np.array([comb(i-1,l)*comb(n-i,L-1-l)/comb(n-1,L-1) for l in range(L)])
def solve(pi,Psi,delta,Gam,sigma):
    L,n=pi.shape
    b=np.array([brentq(lambda bb: np.mean(1/(1+2*bb*pi[l]))-(1-1/delta),1e-12,1e12) for l in range(L)])
    E=np.eye(L)[Psi]; om=2*b[:,None]*pi/(1+2*b[:,None]*pi)
    Lam=np.stack([delta*(om[l][:,None]*E).mean(0) for l in range(L)],1)
    K=np.zeros((L,L))
    for l in range(L):
        for m in range(L):
            c=2*pi[l]/(1+2*b[l]*pi[l])*2*pi[m]/(1+2*b[m]*pi[m])
            a=np.einsum('ij,jk,ik->i',Lam[:,l][None,:]-E,Gam,Lam[:,m][None,:]-E)
            s=delta*b[l]*b[m]*np.mean(c)
            K[l,m]=delta*b[l]*b[m]*np.mean(c*(a+sigma**2))/(1-s)
    Ev=np.stack([(np.einsum('ij,jk,ik->i',Lam[:,l][None,:]-E,Gam,Lam[:,l][None,:]-E)+K[l,l]+sigma**2)/(1+2*b[l]*pi[l])**2 for l in range(L)],1)
    return b,Lam,K,Ev
def split(R):
    n=R.shape[0]
    A=np.concatenate([[0],np.cumsum(R[:,0]-R[:,1])]); Bc=np.concatenate([[0],np.cumsum(R[:,1]-R[:,2])])
    # objective A[k1-1]+B[k2-1], 2<=k1<k2<=n-1  (eta = first index of new segment)
    pre=np.minimum.accumulate(A[1:n-2]); arg=np.array([np.argmin(A[1:j+2]) for j in []]) if False else None
    best=np.inf; out=None; amv=np.inf; am=None
    for k2 in range(3,n):
        if A[k2-2]<amv: amv=A[k2-2]; am=k2-1
        v=amv+Bc[k2-1]
        if v<best: best=v; out=(am,k2)
    return np.array(out)
def empirical(n,p,Gam,sigma,alpha,reps,rng):
    L=3; pi=weights(n,L); i=np.arange(1,n+1)
    eta=[int(alpha[0]*n),int(alpha[1]*n)]
    Psi=np.where(i<eta[0],0,np.where(i<eta[1],1,2)); C=np.linalg.cholesky(Gam); out=[]
    for r in range(reps):
        B=rng.standard_normal((p,L))@C.T*np.sqrt(n/p)
        X=rng.standard_normal((n,p))/np.sqrt(n)
        y=np.einsum('ij,ji->i',X,B[:,Psi])+sigma*rng.standard_normal(n)
        Bh=np.stack([np.linalg.lstsq(X*np.sqrt(pi[l])[:,None],y*np.sqrt(pi[l]),rcond=None)[0] for l in range(L)],1)
        out.append(split((y[:,None]-X@Bh)**2))
    return np.array(out)/n, pi, Psi

