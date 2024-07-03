import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from WERM.model import ζ
import matplotlib.pyplot as plt

def η_to_ψ_jax(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.

    Involves many tricks in order to avoid jax errors. 
    """
    # Compatible with the -1 entries:
    # num_chgpts = jnp.where(η != -1, 1, 0).sum() # Checking for -1's causes jax errors
    # assert jnp.all(η >= -1), "η should be non-negative" # Causes jax errors. To do this check, would have to first iterate through η and check for -1's.
    num_chgpts = η.flatten().shape[0]
    ψ = jnp.ones(n, dtype=int) * num_chgpts
    # ψ = ψ.at[:η[0]].set(0)
    # ψ[:η[0]] = 0
    ψ = jnp.where(jnp.arange(n) < η[0], 0, ψ)
    cond_0 = jnp.where(jnp.arange(n) < η[0], 0, ψ)
    for l in range(1, num_chgpts):
        cond_1 = jnp.where(jnp.arange(n) >= η[l-1], l, cond_0) # Tricks to avoid jax errors
        cond_2 = jnp.where(jnp.arange(n) < η[l], cond_1, l + 1)
        ψ = cond_2
        cond_0 = cond_2
    # assert jnp.diff(ψ).min() >= 0, "ψ should be non-decreasing" # Creates jax errors
    return ψ
η_to_ψ_jax_mapped = jax.vmap(η_to_ψ_jax, (0, None), 0)

def hausdorff(x1, x2):
  """ Calculate the Hausdorff distance between two sets of points using scipy."""
  x1 = np.array(x1).reshape(-1, 1)
  x2 = np.array(x2).reshape(-1, 1)
  return max(sp.spatial.distance.directed_hausdorff(x1, x2)[0], sp.spatial.distance.directed_hausdorff(x2, x1)[0])

def hausdorff_jax(x1, x2):
    """ Calculate the Hausdorff distance between two sets of points using jax. NOT TESTED. """
    x1 = jnp.array(x1).reshape(-1, 1)
    x2 = jnp.array(x2).reshape(-1, 1)
    return jnp.maximum(jnp.max(jnp.abs(x1 - x2)), jnp.max(jnp.abs(x2 - x1)))

### --- Logistic regression with change points --- ###

def LR_gof_j(Θ, y, c_j, L):
    if Θ.ndim == 1:
        Θ = Θ.reshape((1, L))
    num_samples = Θ.shape[0]
    Ψ = jnp.zeros((num_samples, L))
    Ψ = Ψ.at[:, c_j].set(1)

    Z_ψ = jnp.multiply(Θ, (Ψ == 1).astype(int)).sum(axis=1).reshape(
        (num_samples, 1)) # Trick that allows jax to differentiate through, turns indexing into multiplication
    gof_j = ζ(Z_ψ) - jnp.multiply(y, Z_ψ)
    return gof_j

def LR_gof_i(Θ, y, C):
    """ Logistic Regression with a Change Point. This takes in Θ ∈ R^{nxL} and C ∈ {0,L-1}^{nxL} and returns Y ∈ R^{nx1}.  """

    L = Θ.shape[1]
    n = Θ.shape[0]
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    gof_vec = jax.vmap(LR_gof_j, in_axes=(0, 0, 0, None), \
                    out_axes=0)(Θ, y, C, L)
    return gof_vec.sum() / y.shape[0]

def LR_chgpt_gof_jax(Θ, y, η_arr):
    """ Has to be generalized to allow -1's in η_arr"""

    C_s = η_to_ψ_jax_mapped(η_arr, Θ.shape[0])  

    gof = jax.vmap(LR_gof_i, in_axes=(None, None, 0), out_axes=0)(Θ, y, C_s)
    idx_opt = jnp.argmin(gof)
    η_opt = η_arr[idx_opt]
    return η_opt[η_opt != -1], idx_opt

def LR_plot_gof_jax(Θ, y, η_arr):
    C_s = η_to_ψ_jax_mapped(η_arr, Θ.shape[0])  

    gof = jax.vmap(LR_gof_i, in_axes=(None, None, 0), out_axes=0)(Θ, y, C_s)
    fig, ax = plt.subplots()
    ax.plot(gof)
    return fig