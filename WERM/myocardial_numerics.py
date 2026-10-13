"""Equivalent, residual-safeguarded proximal roots for the refinement stage."""
import numpy as np
from scipy.special import expit


def logistic_prox_fast(v,a,y):
    v,a=np.broadcast_arrays(v,a)
    lo=v-a*(1-y);hi=v+a*y;t=v.copy()
    for _ in range(60):
        p=expit(t);f=t+a*(p-y)-v
        tolerance=8*np.finfo(float).eps*(1+abs(v)+abs(t)+a)
        done=abs(f)<=tolerance
        if done.all():return t
        lo=np.where(f<0,t,lo);hi=np.where(f>0,t,hi)
        new=t-f/(1+a*p*(1-p))
        new_f=new+a*(expit(new)-y)-v
        accept=(new>=lo)&(new<=hi)&((abs(new_f)<=.5*abs(f))|(abs(new_f)<=tolerance))
        t=np.where(done,t,np.where(accept,new,(lo+hi)/2))
    return t
