import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from WERM.model import ζ
import matplotlib.pyplot as plt
import itertools
from functools import partial

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
        
        # A -1 in the change point configuration indicates end of change points
        if η[l] == -1:
            break

        cond_1 = jnp.where(jnp.arange(n) >= η[l-1], l, cond_0) # Tricks to avoid jax errors
        cond_2 = jnp.where(jnp.arange(n) < η[l], cond_1, l + 1)
        ψ = cond_2
        cond_0 = cond_2
    # assert jnp.diff(ψ).min() >= 0, "ψ should be non-decreasing" # Creates jax errors
    return ψ
η_to_ψ_jax_mapped = jax.vmap(η_to_ψ_jax, (0, None), 0)

def η_to_ψ_jax_combinations(η, n, comb): 
    """ Can only deal with >0 change points. Does not work if η has all -1's."""

    if len(comb) == 1:
        return jnp.where(jnp.arange(n) < η[0], 0, comb[0])

    ψ = jnp.ones(n, dtype=int)
    cond_0 = jnp.where(jnp.arange(n) < η[0], 0, ψ)

    for i in range(0, len(comb)-1):
        l = comb[i]
        l_next = comb[i+1]
        
        cond_1 = jnp.where(jnp.arange(n) >= η[i], l, cond_0) # Tricks to avoid jax errors
        cond_2 = jnp.where(jnp.arange(n) < η[i+1], cond_1, l_next)
        ψ = cond_2
        cond_0 = cond_2
    return ψ

def η_to_ψ_jax_combinations_mapped(η_arr, n, L):
    """ Returns an array of signal configurations ψ_arr, along with an array η_idx indicating the index of the η configuration that generated each ψ configuration (useful for gof computations and point estimation). 
    
    To be tested.  
    """

    combinations = []
    max_num_chgpts = η_arr.shape[1]
    for num_chgpts in range(0, max_num_chgpts):
        combinations.append(list(itertools.combinations(range(1, L), num_chgpts + 1)))
    print("combinations: ", combinations)

    num_chgpts_mask = jnp.array([jnp.where(η_arr != -1, 1, 0).sum(axis=1)]).T # To be tested
    print("num_chgpts_mask: ", num_chgpts_mask)

    η_to_ψ_jax_combinations_vmap = (jax.vmap(η_to_ψ_jax_combinations, (0, None, None), 0)) # The key step that speeds everything up
    for num_chgpts in range(0, max_num_chgpts+1): 
        print("num_chgpts: ", num_chgpts)

        # Deal with the special case of zero change points
        if num_chgpts == 0:
            ψ_arr = jnp.zeros(n, dtype=int)
            η_idx = jnp.array([0])
            continue

        # Identify the configurations with the correct number of change points
        mask = jnp.where(num_chgpts_mask == num_chgpts, 1, 0).flatten()
        print("mask: ", mask)

        # Apply the mask to η_arr
        η_arr_masked = η_arr[mask.astype(bool)]

        # Apply the function to the configurations with the correct number of change points, and stack the results
        combs = combinations[num_chgpts-1]
        for comb in combs:
            ψ = η_to_ψ_jax_combinations_vmap(η_arr_masked, n, comb)
            if 'ψ_arr' not in locals():
                ψ_arr = ψ
            else:
                ψ_arr = jnp.vstack((ψ_arr, ψ))

            if 'η_idx' not in locals():
                η_idx = jnp.where(mask)[0]
            else:
                η_idx = jnp.concatenate((η_idx, jnp.where(mask)[0]))

    return ψ_arr, η_idx

def η_to_ψ(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.

    Does not involve jax, but can take η with -1s.
    """
    # Compatible with the -1 entries:
    num_chgpts = np.sum(η != -1)
    ψ = np.ones(n, dtype=int) * num_chgpts
    # ψ = ψ.at[:η[0]].set(0)
    ψ[:η[0]] = 0
    for l in range(1, num_chgpts):
        # ψ = ψ.at[η[l-1]:η[l]].set(l)
        ψ[η[l-1]:η[l]] = l
    assert np.diff(ψ).min() >= 0, "ψ should be non-decreasing"
    return ψ

def η_arr_to_C_s(η_arr, n):
    """
    Convert η_arr (num_configs x (L-1) matrix) into C_s 
    (num_configs x n matrix). Does not involve jax, but can take η with -1s.
    """
    num_configs = η_arr.shape[0]
    C_s = np.zeros((num_configs, n), dtype=int)
    for i_config in range(num_configs):
        η = η_arr[i_config, :]
        C_s[i_config, :] = η_to_ψ(η, n)
    return C_s

def ψ_to_η(ψ, L):
    """
    Input C: a length-n vector storing a possible chgpt config i.e. 
    the signal index underlying each of the n observations    

    Returns: indices of changepoints, where the first observation of a 
    new signal is counted as the location of the changepoint.
    """
    if jnp.all(ψ == ψ[0]):
        return jnp.array([-1 for ℓ in range(L - 1)])

    idx_before_chgpt = -2 * jnp.ones(L - 1)

    # @jax.jit
    def jit_subfn(ψ):
        diff_vec = jnp.abs(jnp.diff(ψ.reshape(1, -1), axis = 1))
        diff_vec = jnp.minimum(diff_vec, 1).squeeze() # if jumped from signal 0 to 2, still count as a chgpt
        return diff_vec
    
    jit_res = (jit_subfn(ψ) == 1)
    idx_before_chgpt_size = (jit_res).sum()
    for j in range(0, idx_before_chgpt_size): 
        idx_before_chgpt = idx_before_chgpt.at[j].set(jnp.where(jit_res)[0][j])
    # idx_before_chgpt = jnp.where(jit_subfn(ψ) == 1)[0] # idx of last observation from previous signal
    # print("idx_before_chgpt: ", idx_before_chgpt)
    # print("idx_before_chgpt.shape: ", idx_before_chgpt.shape)
    # idx_before_chgpt = jnp.array(jnp.nonzero(jit_subfn(ψ, L))[0])
    # print("idx_before_chgpt: ", idx_before_chgpt)
    # print("idx_before_chgpt.shape: ", idx_before_chgpt.shape)
    # idx_before_chgpt = 
    # Replace the above line with jax.lax.select to avoid jax errors
    # idx_before_chgpt = jnp.where(jnp.abs(diff_vec) == 1, jnp.arange(diff_vec.shape[0]), -1)
    # idx_before_chgpt = jax.lax.select(diff_vec == 1, jnp.arange(diff_vec.shape[0]), -1)
    

    assert len(idx_before_chgpt) > 0
    res = idx_before_chgpt + 1 

    # Pad the results with -1s so that it has length L - 1
    # res = jnp.concatenate((res, -1 * jnp.ones(L - 1 - len(res))), axis = 0)
    return res.astype(int)

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