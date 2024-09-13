import itertools
import jax.numpy as jnp
import jax
import numpy as np
from scipy.special import comb
from tqdm.auto import tqdm


def unif_prior_to_η_ϕ_combinations(Lmin, Lmax, Δ, n, p_l: np.array = None):
    """
    Returns: Lxn matrix ϕ, where ϕ[l, t] is the marginal probability of observing 
    signal l at time t.

    Allow at most L signals i.e. 0, 1, 2, ... L signals.
    When L_tmp < L signals, assume signals present are sequential: 0, ..., L_tmp-1.
    
    p_l: length-L vector storing the prior probability over the number of signals.
    When left blank, assume uniform prior over all η configs, across l ∈ [Lmin, Lmax].
    Can be uniform or arbitrary as long as it sums to 1. [tested in test_chgpt_configs.py]

    η: tot_nconfigs x (L-1) matrix, where each row stores a config via 
       the starting index i of each signal l=1, ..., (L-1). 
       Observation index i and signal index l are 0-indexed.

    Considers all possible combinations of signals with number of signals l < Lmax, and computes the marginal probability of observing signal l at time t.
    """

    assert Lmin > 0 and Lmax >= Lmin
    num_L = Lmax - Lmin + 1
    if p_l is not None:
        assert len(p_l) == num_L, "p_l must store the prior probability over [Lmin, Lmax]"
        assert np.isclose(np.sum(p_l), 1), "p_l must be normalised to 1"

    assert Δ >= 1 and Δ * Lmax <= n, \
        "If Δ*Lmax > n, then some L∈[Lmin, Lmax] will " + \
        "have no valid configs satisfying minimum Δ intersignal distance" + \
        "making p_l meaningless as well."
    # Each config stores starting index i of each signal:
    η_arr = -np.ones((0, Lmax-1), dtype=int)
    # Stores number of configs when there exist l signals, for each l ∈ [Lmin, Lmax]:
    nconfigs_arr = np.zeros(num_L, dtype=int) # exactly l signals
    marg_nconfigs_arr = np.zeros((n, Lmax), dtype=int)
    if p_l is not None:
        # Marginal probability that each yi in i∈[n] is from signal l:
        ϕ = np.zeros((n, Lmax))
    for l in tqdm(range(Lmin, Lmax+1)): 

        nchgpts_l = l - 1
        η_l = configs_arr_for_exact_L(n, l, Δ) # nconfigs_l x nchgpts_l. Most computationally expensive, stays the same regardless of combinations. 
        nconfigs_l = η_l.shape[0]
        nconfigs_arr[l-Lmin] = nconfigs_l
        η_l_padded = -np.ones((nconfigs_l, Lmax-1), dtype=int) # -1s represent non-existent chgpts
        η_l_padded[:, :nchgpts_l] = η_l
        η_arr = np.concatenate((η_arr, η_l_padded), axis=0)

        marg_nconfigs_arr_l = nconfigs_arr_for_exact_L(η_l, n) # nxl, this is going to hold for each possible combination of signals
        if l <= 2: # No combinations need to be considered, back to the original code.
            marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
            if p_l is not None:
                ϕ[:, :l] += marg_nconfigs_arr_l / nconfigs_l * p_l[l-Lmin] # This is where the key update happens, and where the combo will be used.
                assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
                # assuming given l, chgpts are uniform:
                # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
            continue
        
        # Combinations code begins
        combs = list(itertools.combinations(range(1, Lmax), l-1))

        for j in range(len(combs)): 
            combs_idx = np.array([0, *combs[j]]) # Prepend the 0 because we always assume β_0 always takes the first spot
            marg_nconfigs_arr[:, combs_idx] += marg_nconfigs_arr_l
            # marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
        
        if p_l is not None:
            for j in range(len(combs)): 
                combs_idx = np.array([0, *combs[j]]) # Prepend the 0 because we always assume β_0 always takes the first spot
                ϕ[:, combs_idx] += marg_nconfigs_arr_l / nconfigs_l * (p_l[l-Lmin] / len(combs)) # This is where the key update happens, and where the combo will be used.
            assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
            # assuming given l, chgpts are uniform:
            # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
    
    tot_nconfigs = np.sum(nconfigs_arr)
    assert η_arr.shape[0] == tot_nconfigs

    # Marginal probability that each yi in i∈[n] is from signal l:
    if p_l is None: # assume uniform prior over all η configs.
        ϕ = marg_nconfigs_arr / tot_nconfigs # nxLmax
    assert np.allclose(np.sum(ϕ, axis=1), 1)

    # This section stays the same as it only updates p_η_arr, which is independent of combinations (only change-point location dependent). 
    if p_l is None:
        p_η_arr = np.ones(tot_nconfigs) / tot_nconfigs
    else:
        p_η_arr = np.zeros(tot_nconfigs)
        nconfigs_so_far = 0
        for l in range(Lmin, Lmax+1):
            nconfigs_l = nconfigs_arr[l - Lmin]
            idx_start = nconfigs_so_far
            idx_end = nconfigs_so_far + nconfigs_l
            p_η_arr[idx_start:idx_end] = p_l[l - Lmin] / nconfigs_l 
            nconfigs_so_far += nconfigs_l
    assert np.allclose(np.sum(p_η_arr), 1)
    # η: tot_nconfigs x (Lmax-1)
    # p_η: length-tot_nconfigs vector
    # ϕ.T: Lmax x n matrix
    return tot_nconfigs, η_arr, p_η_arr, ϕ.T 

def unif_prior_to_η_ϕ(Lmin, Lmax, Δ, n, p_l: np.array = None):
    """
    Returns: Lxn matrix ϕ, where ϕ[l, t] is the marginal probability of observing 
    signal l at time t.

    Allow at most L signals i.e. 0, 1, 2, ... L signals.
    When L_tmp < L signals, assume signals present are sequential: 0, ..., L_tmp-1.
    
    p_l: length-L vector storing the prior probability over the number of signals.
    When left blank, assume uniform prior over all η configs, across l ∈ [Lmin, Lmax].
    Can be uniform or arbitrary as long as it sums to 1. [tested in test_chgpt_configs.py]

    η: tot_nconfigs x (L-1) matrix, where each row stores a config via 
       the starting index i of each signal l=1, ..., (L-1). 
       Observation index i and signal index l are 0-indexed.
    """
    assert Lmin > 0 and Lmax >= Lmin
    num_L = Lmax - Lmin + 1
    if p_l is not None:
        assert len(p_l) == num_L, "p_l must store the prior probability over [Lmin, Lmax]"
        assert np.isclose(np.sum(p_l), 1), "p_l must be normalised to 1"

    assert Δ >= 1 and Δ * Lmax <= n, \
        "If Δ*Lmax > n, then some L∈[Lmin, Lmax] will " + \
        "have no valid configs satisfying minimum Δ intersignal distance" + \
        "making p_l meaningless as well."
    # Each config stores starting index i of each signal:
    η_arr = -np.ones((0, Lmax-1), dtype=int)
    # Stores number of configs when there exist l signals, for each l ∈ [Lmin, Lmax]:
    nconfigs_arr = np.zeros(num_L, dtype=int) # exactly l signals
    marg_nconfigs_arr = np.zeros((n, Lmax), dtype=int)
    if p_l is not None:
        # Marginal probability that each yi in i∈[n] is from signal l:
        ϕ = np.zeros((n, Lmax))
    for l in tqdm(range(Lmin, Lmax+1)): 

        nchgpts_l = l - 1
        η_l = configs_arr_for_exact_L(n, l, Δ) # nconfigs_l x nchgpts_l. The slowest part of the code.
        nconfigs_l = η_l.shape[0]
        nconfigs_arr[l-Lmin] = nconfigs_l
        η_l_padded = -np.ones((nconfigs_l, Lmax-1), dtype=int) # -1s represent non-existent chgpts
        η_l_padded[:, :nchgpts_l] = η_l
        η_arr = np.concatenate((η_arr, η_l_padded), axis=0)

        marg_nconfigs_arr_l = nconfigs_arr_for_exact_L(η_l, n) # nxl, this is going to hold for each possible combination of signals
        marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
        if p_l is not None:
            ϕ[:, :l] += marg_nconfigs_arr_l / nconfigs_l * p_l[l-Lmin] # This is where the key update happens, and where the combo will be used.
            assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
            # assuming given l, chgpts are uniform:
            # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
    
    tot_nconfigs = np.sum(nconfigs_arr)
    assert η_arr.shape[0] == tot_nconfigs

    # Marginal probability that each yi in i∈[n] is from signal l:
    if p_l is None: # assume uniform prior over all η configs.
        ϕ = marg_nconfigs_arr / tot_nconfigs # nxLmax
    assert np.allclose(np.sum(ϕ, axis=1), 1)

    if p_l is None:
        p_η_arr = np.ones(tot_nconfigs) / tot_nconfigs
    else:
        p_η_arr = np.zeros(tot_nconfigs)
        nconfigs_so_far = 0
        for l in range(Lmin, Lmax+1):
            nconfigs_l = nconfigs_arr[l - Lmin]
            idx_start = nconfigs_so_far
            idx_end = nconfigs_so_far + nconfigs_l
            p_η_arr[idx_start:idx_end] = p_l[l - Lmin] / nconfigs_l 
            nconfigs_so_far += nconfigs_l
    assert np.allclose(np.sum(p_η_arr), 1)
    # η: tot_nconfigs x (Lmax-1)
    # p_η: length-tot_nconfigs vector
    # ϕ.T: Lmax x n matrix
    return tot_nconfigs, η_arr, p_η_arr, ϕ.T 

def configs_arr_for_exact_L(n:int, L:int, Δ:int=1):
    """
    Returns a nconfigs x (L-1) matrix configs_arr, where each row 
    stores a config via the starting index of each signal l.
    
    configs_arr[j, l] = starting index of signal l in config j.

    This function may cause memory issues for larger L (L>5).

    Tested on Jupyter notebook with (L = 4, L^* = 3), (L = 4, L^* = 2). 
    """
    assert L >= 1
    nchgpts = L - 1
    # tot_nconfigs = int(comb(n-1, nchgpts))

    # η_arr = np.zeros((tot_nconfigs, nchgpts), dtype=int)
    # for i, combo in tqdm(enumerate(itertools.combinations(range(1, n), nchgpts)), total=tot_nconfigs):
    #     η_arr[i] = combo
    range_start = np.maximum(1, Δ).astype(int)
    desired_range = range(np.maximum(1, Δ).astype(int),n - Δ)
    tot_nconfigs = int(comb(len(desired_range), nchgpts))
    η_arr = np.array(list(itertools.combinations(desired_range, nchgpts))
                           ).reshape(tot_nconfigs, nchgpts)

    # η_arr is nconfigs x nchgpts. Signal 0 always starts from index 0.
    if Δ > 1:
        # Discard violating configs:
        # if Δ > 1:
        #     # Discard violating configs:
        #     def is_ok_config(η):
        #         if np.all(η == -1):
        #             return True
        #         η_extended = np.concatenate(([0], η[η!=-1], [n-1]))
        #         # adjacent chgpts must be at least Δ apart, first/last chgpt 
        #         # must be at least Δ away from the endpoints:
        #         return np.all(np.diff(η_extended) > Δ)
        #     η_arr = η_arr[np.apply_along_axis(is_ok_config, axis=1, arr=η_arr)]

        # Vectorize the above for-loop. This is correct because η_arr will have no -1's, and so we can take the diff. 
        η_extended = jnp.block([jnp.zeros((η_arr.shape[0], 1)), η_arr , (n-1) * jnp.ones((η_arr.shape[0], 1))])
        η_ext_diff = jnp.diff(η_extended, axis=1)
        # mask = jnp.all(jnp.logical_or(η_ext_diff > Δ, (η_ext_diff == 0)), axis = 1)
        mask = jnp.all(η_ext_diff > Δ, axis = 1)
        # Discard violating configs according to mask: 
        η_arr = η_arr[mask]

    assert np.all(np.diff(η_arr, axis=1) > 0), "Each row is non-decreasing"
    return η_arr.astype(int)
        
def nconfigs_arr_for_exact_L(η_arr, n):
    """
    Returns nxL matrix, where each row stores the number of configs
    that have yi underlain by signal l, for each i ∈ [n] and l ∈ [L].

    η_arr is nconfigs x (L-1)
    """
    nconfigs = η_arr.shape[0]
    L = η_arr.shape[1] + 1
    nconfigs_arr = np.zeros((n, L), dtype=int)
    for i_config in tqdm(range(nconfigs)):
        η = η_arr[i_config, :]
        num_chgpts = np.sum(η != -1)
        if num_chgpts == 0:
            nconfigs_arr[:, 0] += 1
        else:
            η_extended = np.concatenate(([0], η[η!=-1], [n]))
            for l in range(num_chgpts+1):
                nconfigs_arr[η_extended[l]:η_extended[l+1], l] += 1
    assert np.all(np.sum(nconfigs_arr, axis=1) == nconfigs)
    return nconfigs_arr

def η_to_ψ(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.
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

    
    Tested on Jupyter notebook with (L = 4, L^* = 3), (L = 4, L^* = 2), (L = 4, L^* = 1).
    """

    num_chgpts_mask = jnp.array([jnp.where(η_arr != -1, 1, 0).sum(axis=1)]).T # To be tested
    min_num_chgpts = num_chgpts_mask.min()
    max_num_chgpts = num_chgpts_mask.max()
    assert max_num_chgpts == L - 1, "The maximum number of change points should be L - 1"
    assert max_num_chgpts == η_arr.shape[1], "The maximum number of change points should be equal to the number of columns in η_arr"

    L_star_min = min_num_chgpts + 1
    assert L_star_min >= 1

    η_to_ψ_jax_combinations_vmap = (jax.vmap(η_to_ψ_jax_combinations, (0, None, None), 0)) # The key step that speeds everything up
    if L_star_min == L: # Account for the special case of L_star = L
        combinations = list(itertools.combinations(range(1, L), L-1))
        comb = combinations[0]
        ψ_arr = η_to_ψ_jax_combinations_vmap(η_arr, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
        η_idx = jnp.arange(η_arr.shape[0])
        return ψ_arr, η_idx
    assert L_star_min == 1, "L_star_min should either be equal to 1 or equal to L in order for the code to work"


    combinations = []
    max_num_chgpts = η_arr.shape[1]
    for num_chgpts in range(0, max_num_chgpts):
        combinations.append(list(itertools.combinations(range(1, L), num_chgpts + 1)))

    for num_chgpts in range(0, max_num_chgpts+1): 

        # Deal with the special case of zero change points
        # if num_chgpts == 0:
        #     ψ_arr = jnp.zeros(n, dtype=int)
        #     η_idx = jnp.array([0])
        #     continue

        # Identify the configurations with the correct number of change points
        mask = jnp.where(num_chgpts_mask == num_chgpts, 1, 0).flatten()

        # Apply the mask to η_arr
        η_arr_masked = η_arr[mask.astype(bool)]

        # Apply the function to the configurations with the correct number of change points, and stack the results
        combs = combinations[num_chgpts-1]
        if num_chgpts == 0: # Special case of zero change points
            # combs = [(0, )]
            ψ_arr = jnp.multiply(jnp.ones((L, n), dtype=int), jnp.arange(0, L).reshape(L, 1))
            η_idx = jnp.repeat(jnp.where(mask)[0], L)
            continue

        for comb in combs:
            ψ = η_to_ψ_jax_combinations_vmap(η_arr_masked, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
            if 'ψ_arr' not in locals():
                ψ_arr = ψ
            else:
                ψ_arr = jnp.vstack((ψ_arr, ψ))

            if 'η_idx' not in locals():
                η_idx = jnp.where(mask)[0]
            else:
                η_idx = jnp.concatenate((η_idx, jnp.where(mask)[0]))

    return ψ_arr, η_idx


    # combinations = []
    # L_star_list = list(range(L_star_min, L))
    # for L_star in L_star_list:
    #     combinations.append(list(itertools.combinations(range(1, L), L_star)))

    # num_chgpts_mask = jnp.array([jnp.where(η_arr != -1, 1, 0).sum(axis=1)]).T # To be tested


    # for i in range(0, len(combinations)):
    #     L_star = L_star_list[i]
    #     num_chgpts = L_star - 1

    #     # Identify the configurations with the correct number of change points
    #     mask = jnp.where(num_chgpts_mask == num_chgpts, 1, 0).flatten()

    #     # Apply the mask to η_arr
    #     η_arr_masked = η_arr[mask.astype(bool)]

    #     # Apply the function to the configurations with the correct number of change points, and stack the results
    #     combs = combinations[i]
    #     if num_chgpts == 0: # Special case of zero change points
    #         ψ_arr = jnp.multiply(jnp.ones((L, n), dtype=int), jnp.arange(0, L).reshape(L, 1))
    #         η_idx = jnp.repeat(jnp.where(mask)[0], L)
    #         continue

    #     for comb in combs:
    #         ψ = η_to_ψ_jax_combinations_vmap(η_arr_masked, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
    #         if 'ψ_arr' not in locals():
    #             ψ_arr = ψ
    #         else:
    #             ψ_arr = jnp.vstack((ψ_arr, ψ))

    #         if 'η_idx' not in locals():
    #             η_idx = jnp.where(mask)[0]
    #         else:
    #             η_idx = jnp.concatenate((η_idx, jnp.where(mask)[0])) 

    # return ψ_arr, η_idx

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
    # idx_before_chgpt = jnp.array(jnp.nonzero(jit_subfn(ψ, L))[0])
    # idx_before_chgpt = 
    # Replace the above line with jax.lax.select to avoid jax errors
    # idx_before_chgpt = jnp.where(jnp.abs(diff_vec) == 1, jnp.arange(diff_vec.shape[0]), -1)
    # idx_before_chgpt = jax.lax.select(diff_vec == 1, jnp.arange(diff_vec.shape[0]), -1)
    

    assert len(idx_before_chgpt) > 0
    res = idx_before_chgpt + 1 

    # Pad the results with -1s so that it has length L - 1
    # res = jnp.concatenate((res, -1 * jnp.ones(L - 1 - len(res))), axis = 0)
    return res.astype(int)
