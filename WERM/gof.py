import jax
import jax.numpy as jnp
import numpy as np
from WERM.model import ζ
import matplotlib.pyplot as plt
from WERM.config import η_to_ψ_jax_mapped, η_to_ψ_jax_combinations_mapped

### --- Linear model with change points --- ###

def lin_reg_opt_chgpt_gof(Z, y, η_arr, p_η_arr, σ = 0.5, ϵ = 0.15):
    """ Z is n × L, where L is the number of signals. 
    Don't need to vmap this whole thing, but can jit the inside of for loop to make it faster. 
    Assumes we are using the linear regression loss. """

    L = Z.shape[1]
    C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Z.shape[0], L)

    # Implement with jax.lax.scan. Uses less memory than vmap, and nearly as fast.
    gof_list = []
    p_η_arr_ = jnp.array(p_η_arr)

    def scan_fn(carry, x):
        y, Z = carry
        ψ, idx = x[:-1], x[-1]
        gof = jnp.linalg.norm(y - Z[jnp.arange(Z.shape[0]), ψ], ord=2) / (2 * jnp.maximum(σ**2, ϵ**2)) - jnp.log(p_η_arr_[idx]) 
        return (y, Z), gof
    xs = jnp.block([C_s, η_idx.reshape(-1, 1)])
    _, gof_list = jax.lax.scan(scan_fn, (y, Z), xs)

    gof_list = np.array(gof_list)
    i_opt = gof_list.argmin()
    η_opt = η_arr[η_idx[i_opt]]

    return η_opt[η_opt != -1]

### --- Logistic model with change points --- ###

def logistic_reg_opt_chgpt_gof(Z, y, η_arr, p_η_arr):

    L = Z.shape[1]
    C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Z.shape[0], L)

    # Implement with jax.lax.scan. Uses less memory than vmap, and nearly as fast.
    gof_list = []
    p_η_arr_ = jnp.array(p_η_arr)

    def scan_fn(carry, x):
        y, Z = carry
        ψ, idx = x[:-1], x[-1]
        Z_ψ = Z[jnp.arange(Z.shape[0]), ψ]
        # gof = jnp.linalg.norm(y - Z[jnp.arange(Z.shape[0]), ψ], ord=2) / (2 * jnp.maximum(σ**2, ϵ**2)) - jnp.log(p_η_arr_[idx]) 
        gof = (ζ(Z_ψ) - jnp.multiply(y, Z_ψ)).sum() - jnp.log(p_η_arr_[idx]) 
        return (y, Z), gof
    xs = jnp.block([C_s, η_idx.reshape(-1, 1)])
    _, gof_list = jax.lax.scan(scan_fn, (y, Z), xs)

    gof_list = np.array(gof_list)
    i_opt = gof_list.argmin()
    η_opt = η_arr[η_idx[i_opt]]

    return η_opt[η_opt != -1]

### --- OLD --- ###

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

    # C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Θ.shape[0], Θ.shape[1])

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