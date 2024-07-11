import itertools
import jax.numpy as jnp
import jax
import numpy as np
from scipy.special import comb
from tqdm.auto import tqdm

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
        η_l = configs_arr_for_exact_L(n, l, Δ) # nconfigs_l x nchgpts_l
        nconfigs_l = η_l.shape[0]
        nconfigs_arr[l-Lmin] = nconfigs_l
        η_l_padded = -np.ones((nconfigs_l, Lmax-1), dtype=int) # -1s represent non-existent chgpts
        η_l_padded[:, :nchgpts_l] = η_l
        η_arr = np.concatenate((η_arr, η_l_padded), axis=0)

        marg_nconfigs_arr_l = nconfigs_arr_for_exact_L(η_l, n) # nxl
        marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
        if p_l is not None:
            ϕ[:, :l] += marg_nconfigs_arr_l / nconfigs_l * p_l[l-Lmin]
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
