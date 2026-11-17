import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from linear_model_theory_lib import solve, split
N=1500; t=(np.arange(N)+0.5)/N; L=3
Phi=np.array([(1-t)**2,2*t*(1-t),t**2]); delta=4; rho=0.5
Gam=np.array([[1,rho,rho**2],[rho,1,rho],[rho**2,rho,1]])
def lim(a1,a2,sigma):
    Psi=np.where(t<a1,0,np.where(t<a2,1,2))
    b,Lam,K,v=solve(Phi,Psi,delta,Gam,sigma)
    return v,(split(v)-1)/N
def dH(A,B): return max(max(min(abs(x-y) for y in B) for x in A),max(min(abs(x-y) for x in A) for y in B))
plt.rcParams.update({'font.size':11})
fig,ax=plt.subplots(1,4,figsize=(17,3.7)); plt.subplots_adjust(wspace=0.38)
for k,(a1,a2) in enumerate([(0.4,0.7),(0.1,0.3)]):
    v,tau=lim(a1,a2,1.0); print((a1,a2),tau)
    for l in range(L): ax[k].plot(t,v[:,l],label=fr'$v_{l+1}(t)$')
    for a in (a1,a2): ax[k].axvline(a,color='k',ls='--',lw=0.9)
    for s in tau: ax[k].axvline(s,color='r',ls=':',lw=1.4)
    ax[k].set_xlabel('$t$'); ax[k].set_title(fr'$(\alpha_1,\alpha_2)=({a1},{a2})$, $\sigma=1$'); ax[k].legend(fontsize=9)
g=np.linspace(0.05,0.95,37)
for k,sigma in [(2,1.0),(3,0.3)]:
    H=np.full((len(g),len(g)),np.nan)
    for i,a1 in enumerate(g):
        for j,a2 in enumerate(g):
            if a2-a1>=0.049: H[j,i]=dH((a1,a2),lim(a1,a2,sigma)[1])
    im=ax[k].imshow(H,origin='lower',extent=[g[0],g[-1],g[0],g[-1]],cmap='viridis',aspect='auto',vmin=0,vmax=0.6)
    ax[k].set_xlabel(r'$\alpha_1$'); ax[k].set_ylabel(r'$\alpha_2$'); ax[k].set_title(fr'$d_H(\tau^*,\alpha)$, $\sigma={sigma}$')
fig.colorbar(im,ax=ax[2:].tolist(),fraction=0.03)
plt.savefig('linear_model_limit.pdf',bbox_inches='tight'); plt.savefig('linear_model_limit.png',dpi=100,bbox_inches='tight')

