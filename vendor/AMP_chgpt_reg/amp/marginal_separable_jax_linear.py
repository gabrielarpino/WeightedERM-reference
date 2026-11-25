import datetime
from typing import Tuple
import jax
from jax import config
config.update("jax_enable_x64", True) # float64
from matplotlib import pyplot as plt
from amp import PAL

from amp.performance_measures import norm_sq_corr
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
import numpy.random as nprandom
import numpy as np
from tqdm.auto import tqdm
import amp.signal_configuration
from amp.signal_priors import SignalPrior, SparseDiffSignal,\
    SparseGaussianSignal, GaussianSignal
import scipy

# Debugging nans
from jax import config; config.update("jax_debug_nans", True)

import amp.changepoint_jax

class q_old():
    """ The heterogeneous output function.
    Generalized q which takes in the C_s matrix, each row consisting of a vector indicating which signal influences data point i. 
    Non-Separable setting with L>=2 signals. 
    Samples a dataset with signal heterogeneity uniform among those dictated by the rows of C_s. 
    We have to implement this in Jax because the code will differentiate through it. 
    """

    def __init__(self, Θ, ϕ, σ):
        self.key = jax.random.PRNGKey(0) # jax.random.PRNGKey(random.randint(0, 100000))
        self.Θ = Θ # XB
        self.n = Θ.shape[0]
        self.ϕ = ϕ # prior on L signals
        assert len(ϕ) == self.Θ.shape[1], "Prior signal probabilities ϕ must have length L"
        assert jnp.isclose(jnp.sum(ϕ), 1), "Prior signal probabilities ϕ must sum to 1"
        self.σ = σ

    def sample(self):
        η = nprandom.normal(loc=0, scale=self.σ, size=(self.n, 1)) # noise
        self.η = jnp.array(η)
        # Draw signal index for each row of X:
        self.Ψ = nprandom.multinomial(1, self.ϕ, size=self.n) # returns nxL binary array
        self.Ψ = jnp.array(self.Ψ)

        Y = self.Θ[self.Ψ == 1].reshape(self.n, 1) + self.η
        return Y

def q_j_sample(Θ, c_j, η, L):
    """Sample output yj according to signal index cj. θ=XB.
    We need to provide the noise instead of sampling it inside the function because jax vamp is unable to handle it."""
    if Θ.ndim == 1:
        Θ = Θ.reshape((1, L))
    num_samples = Θ.shape[0]
    #η = jnp.array(nprandom.normal(loc=0, scale=σ, size=(num_samples, 1))) # noise
    Ψ = jnp.zeros((num_samples, L))
    Ψ = Ψ.at[:, c_j].set(1)

    Y = (jnp.multiply(Θ, (Ψ == 1).astype(int)).sum(axis=1)+ η).reshape((num_samples, 1)) 
    return Y

def generate_noise(σ, n):
    """ Sample noise to provide as input to function q"""
    η = jnp.array(nprandom.normal(loc=0, scale=σ, size=(n, 1)))
    return η

def q(Θ, C, η):
    """ This takes in Θ ∈ R^{nxL} and C ∈ {0,L-1}^{nxL} and returns Y ∈ R^{nx1}.
    Needs to be differentiable to go into SE_fixed_C.  
    One of the inputs is the noise vector η which contains the level of random noise at each observation."""

    L = Θ.shape[1]
    return jax.vmap(q_j_sample, in_axes=(0, 0, 0, None), \
                    out_axes=0)(Θ, C, η, L).reshape((Θ.shape[0], 1))

def log_N(x, Σ):
        """
        Log of the pdf of N_{L+1}(0, Σ) evaluated at x in R^{L+1}.
        """
        assert x.shape[1] == 1
        dim = x.shape[0]
        # This regularization matters a lot, even for just applying g:
        # jax.debug.print(f"Σ in log_N = {Σ}")
        # if jnp.any(jnp.linalg.eigh(Σ)[0] < -1e-5):
        # jax.debug.print(f"eigenvals of Σ in log_N = {jnp.linalg.eigh(Σ)[0]}")
        res = multivariate_normal.logpdf(x.flatten(), mean=jnp.zeros((dim,)), cov=Σ + 1e-10 * jnp.eye(dim)) 
        # \ + (x.shape[0])/2 * jnp.log(2 * jnp.pi)
        return res

log_N_all = jax.jit(jax.vmap(log_N, (None, 0), 0))

def f_j(s, δ, ρ, ν̂, κ_B):
    """Optimal denoiser f_j*: R^L -> R^L, assuming E[B_j]=0."""
    L = ρ.shape[0]
    # ν̂ is symmetric when we use optimal g^t*
    χ = (ρ @ ν̂) @ jnp.linalg.pinv(ν̂ @ ρ @ ν̂ + 1/δ * κ_B)
    # tmp = ν̂ @ ρ @ ν̂ + 1/δ * κ_B
    # ρ_ν̂ = ρ @ ν̂
    # χ_new = jax.scipy.linalg.solve(tmp.T, ρ_ν̂.T, assume_a='her').T
    # The condition number tends to be very large, making solve less stable than pinv.
    # print(f'condition number of tmp: {jnp.linalg.cond(tmp)}')
    assert χ.shape == (L, L)
    return s.reshape((1, L)) @ χ.T
f = jax.jit(jax.vmap(f_j, (0, None, None, None, None), 0))

def soft_thres(s: jnp.array, λ: float) -> jnp.array:
    """The soft-thresholding function for LASSO."""
    # assert λ >= 0, "the threshold λ must be nonnegative"
    return jnp.sign(s) * jnp.maximum(jnp.abs(s) - λ, 0)

def f_j_st(s, ν̂, κ_B, st_ζ: float):
    """Soft thresholding denoiser f_j: R^L -> R^L."""
    L = ν̂.shape[0]
    Σ_ϵ = jnp.linalg.inv(ν̂) @ κ_B @ jnp.linalg.inv(ν̂).T
    B_ν = s.reshape((1, L)) @ jnp.linalg.inv(ν̂)
    def entrywise_denoise(l):
        """Denoise the l-th entry of B_j. R^1-> R^1."""
        return soft_thres(B_ν[0, l], st_ζ * jnp.sqrt(Σ_ϵ[l, l])) # scalar
    res = jax.vmap(entrywise_denoise, 0, 0)(jnp.arange(L)).reshape((1, L))
    return res
f_st = jax.jit(jax.vmap(f_j_st, (0, None, None, None), 0)) # returns pxL

def d_soft_thres(s: jnp.array, λ: float) -> jnp.array:
    """The derivative of the soft-thresholding function for LASSO."""
    # assert λ >= 0, "the threshold λ must be nonnegative"
    s = s.reshape((1, -1)) # row vector
    L = s.shape[1]
    dfji_dsi = (jnp.abs(s) > λ).astype(int)
    dfds = jnp.diag(dfji_dsi.flatten()) # zero off-diagonal
    assert dfds.shape == (L, L)
    return dfds


def dfdB_j_st(s, ν̂, κ_B, st_ζ: float):
    """s is a vector of length L"""
    L = ν̂.shape[0]
    return jax.jacfwd(f_j_st, 0)(s, ν̂, κ_B, st_ζ).reshape((L, L))
# apply dfdB_j row-wise to s, stack results row-wise
dfdB_st = jax.jit(jax.vmap(dfdB_j_st, (0, None, None, None), 0)) # returns pxLxL


def all_binary_sequences(len: int) -> jnp.ndarray:
    """
    Returns an 2^len x len matrix where each row stores a distinct 
    binary sequence (stored as boolean) of length len.

    Needed for sparse prior (binary sequence represents which signals are nonzero.)
    """
    assert len > 0, "len must be a positive integer"
    # 2^len x 1 matrix storing numbers from 0 to (2^len)-1:
    a = jnp.arange(2**len, dtype=jnp.int8)[:, jnp.newaxis]
    # 1 x len matrix for the powers of 2 with exponents from 0 to (len-1):
    b = jnp.arange(start=len - 1, stop=-1, step=-1, dtype=jnp.int8)[jnp.newaxis, :]
    powers_of2 = 2**b
    # u & v = bitwise-and of binary u and binary v:
    # binary_sequences = jnp.array((a & powers_of2) > 0, dtype=jnp.int8)
    binary_sequences = (a & powers_of2) > 0 # bool
    # The first row is always all-False.
    return binary_sequences

def f_j_sparse(s, δ, signal_prior: SparseGaussianSignal, ν̂, κ_B):
    """
    Optimal denoiser f_j*: R^L -> R^L, assuming E[B_j]=0
    for sparse prior. Works for any L.
    Tested in test_f_j_sparse.py.
    """
    α, ρ = signal_prior.α, signal_prior.ρ_B
    L = ρ.shape[0]
    binary_seq_arr = all_binary_sequences(L) # 2^L x L
    num_nz_signals_arr = jnp.sum(binary_seq_arr, axis=1, keepdims=True)
    mixture_prob_s = α ** num_nz_signals_arr * (1-α) ** (L - num_nz_signals_arr)
    assert mixture_prob_s.shape == (2**L, 1)

    # For k in [2^L] (i.e. for the k-th type of signal mixture) construct params:
    ################################# Denominator #################################
    def Σ_k(β_l_is_nonzero_arr: jnp.array) -> jnp.ndarray:
        """
        Calculates the covariance Σ of the given mixture of zero and nonzero signals.
        β_l_is_nonzero_arr is a boolean vector specifying which signals are nonzero.
        """
        ρ̃ = jnp.diag(β_l_is_nonzero_arr * jnp.diag(ρ)) # zero off-diagonal
        assert ρ̃.shape == ρ.shape # ensures shape is static
        # When β_l_is_nonzero_arr is all-zero, ρ̃ is an all-zero matrix.
        Σ = δ/α * ν̂.T @ ρ̃ @ ν̂ + κ_B
        return Σ
    Σ_k_all = jax.vmap(Σ_k, 0, 0) # 2^L Σ matrices stacked along axis 0
    
    Σ_s = Σ_k_all(binary_seq_arr)
    log_N_s = log_N_all(s.reshape((L, 1)), Σ_s) # 2^L x 1
    # Subtract away max term:
    max_expo_denom = jnp.max(log_N_s, axis=0)
    log_denom_arr, denom_sign_arr = logsumexp(
        a=log_N_s.reshape(2**L, 1) - max_expo_denom, axis=0, \
            b=mixture_prob_s.reshape(2**L, 1), return_sign=True)

    ################################# Numerator #################################
    Δ = ν̂ @ jnp.linalg.inv(κ_B) @ ν̂.T
    def params_k(β_l_is_nonzero_arr: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Construct key params for the k-th type of signal mixture."""
        # assert jnp.any(β_l_is_nonzero_arr), \
        #     "β^{(1)}_j, ...β^{(L)}_j must have at least one nonzero"
        ################### Basic params incl Σ̃_B, Σ̃_c, μ̃_c ###################
        ρ̃ = jnp.diag(β_l_is_nonzero_arr * jnp.diag(ρ)) 
        Σ̃_B = δ/α * ρ̃
        inv_ρ̃ = jnp.diag(β_l_is_nonzero_arr * (1 / jnp.diag(ρ)))
        inv_Σ̃_B = α/δ * inv_ρ̃
        # at.set below is incompatible with jit
        # Δ̃ = Δ.at[~β_l_is_nonzero_arr, :].set(0)
        # Δ̃ = Δ.at[:, ~β_l_is_nonzero_arr].set(0)
        Δ̃ = Δ * β_l_is_nonzero_arr.reshape((L, 1)) * β_l_is_nonzero_arr.reshape((1, L))
        inv_Σ̃_c = inv_Σ̃_B + Δ̃
        Σ̃_c = jnp.linalg.pinv(inv_Σ̃_c) # inv doesnt work because inv_Σ̃_c may be singular
        # Ensure near-zero entries are absolute zeros:
        Σ̃_c = Σ̃_c * β_l_is_nonzero_arr.reshape((1, L)) * β_l_is_nonzero_arr.reshape((L, 1))

        μ̃_c = s.reshape((1, L)) @ jnp.linalg.inv(κ_B) @ ν̂.T @ Σ̃_c # 1xL

        ############################# Pre-multiplier ###########################
        # Create surrogates of Σ̃_B and Σ̃_c to help calculate det(nonzero submatrix of Σ̃_B or Σ̃_c).
        # Fill zero diagonal entries of Σ̃_B and Σ̃_c with ones:
        semi_identity_mat = jnp.diag(~β_l_is_nonzero_arr).astype(int)
        det_Σ̃_B = jnp.linalg.det(Σ̃_B + semi_identity_mat)
        
        det_Σ̃_c = jnp.linalg.det(Σ̃_c + semi_identity_mat)
        pre_multiplier = (2 * jnp.pi) ** (-L/2) * \
                det_Σ̃_B ** (-1/2) * \
                jnp.linalg.det(κ_B) ** (-1/2) * \
                det_Σ̃_c ** (1/2) # scalar

        ############################# Exponent #################################
        exponent = -1/2 * (- μ̃_c @ inv_Σ̃_c @ μ̃_c.T + \
                       s.reshape((1, L)) @ jnp.linalg.inv(κ_B) @ s.reshape((L, 1))).squeeze() # scalar
        return μ̃_c, pre_multiplier, exponent
    params_k_all = jax.vmap(params_k, 0, 0) # (2^L-1) triplets stacked along axis 0
    # the minus 1 is because we exclude the all-zero case.
    μ̃_c_s, pre_multipliers, exponents = params_k_all(binary_seq_arr[1:, :]) 
    # μ̃_c_s is (2^L-1) x (1xL) 

    # Subtract away max term in exponents:
    max_expo_num = jnp.max(exponents, axis=0)
    log_num_arr, num_sign_arr = logsumexp(
        a=exponents.reshape((2**L-1, 1)) - max_expo_num, axis=0, 
        b=mixture_prob_s[1:].reshape(2**L-1, 1) * pre_multipliers.reshape(2**L-1, 1) * μ̃_c_s.reshape((2**L-1, L)), 
        return_sign=True) # sum over (2^L-1) terms
    
    res = (num_sign_arr * denom_sign_arr * \
           jnp.exp(max_expo_num - max_expo_denom + log_num_arr - log_denom_arr)).reshape((1, L))
    return res
f_sparse = jax.jit(jax.vmap(f_j_sparse, \
                (0, None, None, None, None), 0), static_argnames='signal_prior') # apply f_j row-wise to s, stack results row-wise

def dfdB_j_sparse(s, δ, signal_prior: SparseGaussianSignal, ν̂, κ_B):
    """s is a vector of length L"""
    L = signal_prior.ρ_B.shape[0]
    return jax.jacfwd(f_j_sparse, 0)(s, δ, signal_prior, ν̂, κ_B).reshape((L, L))
# apply dfdB_j row-wise to s, stack results row-wise
dfdB_sparse = jax.jit(jax.vmap(dfdB_j_sparse, \
                    (0, None, None, None, None), 0), static_argnames='signal_prior')

def f_j_sparse_diff_old(s, δ, signal_prior: SparseDiffSignal, ν̂, κ_B):
    """
    Optimal denoiser f_j*: R^L -> R^L, assuming E[B_j]=0
    for L=2 signals with sparse difference prior.
    Tested in test_f_j_sparse.py.
    """
    L = ν̂.shape[0]
    α, ρ_B_same, ρ_B_diff = signal_prior.α, signal_prior.ρ_B_same, signal_prior.ρ_B_diff
    assert ρ_B_same.shape == ρ_B_diff.shape
    assert L == 2, "This function currently only works for L=2 signals"
    # ν̂ is symmetric when we use optimal g^t*
    Σ_V_same = ν̂.T @ ρ_B_same @ ν̂ + 1/δ * κ_B
    Σ_V_diff = ν̂.T @ ρ_B_diff @ ν̂ + 1/δ * κ_B
    # if jnp.any(jnp.linalg.eigh(Σ_V_same)[0] < 0):
    #     jax.debug.print(f"eigenvals of Σ_V_same = {jnp.linalg.eigh(Σ_V_same)[0]}")
    # if jnp.any(jnp.linalg.eigh(Σ_V_diff)[0] < 0):
    #     jax.debug.print(f"eigenvals of Σ_V_diff = {jnp.linalg.eigh(Σ_V_diff)[0]}")
    χ_same = (ρ_B_same @ ν̂) @ jnp.linalg.pinv(Σ_V_same)
    χ_diff = (ρ_B_diff @ ν̂) @ jnp.linalg.pinv(Σ_V_diff)
    assert χ_same.shape == (L, L) and χ_diff.shape == (L, L)
    log_N_same = log_N(s.reshape((L, 1)), δ * Σ_V_same)
    log_N_diff = log_N(s.reshape((L, 1)), δ * Σ_V_diff)
    log_N_arr = jnp.array([log_N_same, log_N_diff]).reshape((2, 1))
    # Subtract max term to avoid overflow:
    log_N_arr = log_N_arr - jnp.max(log_N_arr, axis=0)

    mixture_prob_arr = jnp.array((1-α, α)).reshape((2, 1))
    s_χ_arr = jnp.array([s.reshape((1, L)) @ χ_same.T, \
                          s.reshape((1, L)) @ χ_diff.T]).reshape((2, L))
    log_num_arr, num_sign_arr = logsumexp(
        a=log_N_arr, axis=0, b=mixture_prob_arr * s_χ_arr, return_sign=True)
    log_denom_arr, sign_denom_arr = logsumexp(
        a=log_N_arr, axis=0, b=mixture_prob_arr, return_sign=True)
    # print(f'log_num_arr: {log_num_arr}')
    # print(f'log_denom_arr: {log_denom_arr}')
    res = (num_sign_arr * sign_denom_arr * jnp.exp(log_num_arr - log_denom_arr)
           ).reshape((1, L))
    # assert not np.isnan(np.array(log_num_arr)).any(), "log_num_arr contains nan"
    # checkify.check(not np.isnan(np.array(log_denom_arr)).any(), "log_denom_arr contains nan")
    return res
f_sparse_diff_old = jax.jit(jax.vmap(f_j_sparse_diff_old, \
                (0, None, None, None, None), 0), static_argnames='signal_prior') # apply f_j row-wise to s, stack results row-wise

def dfdB_j_sparse_diff_old(s, δ, signal_prior: SparseDiffSignal, ν̂, κ_B):
    """s is a vector of length L"""
    L = ν̂.shape[0]
    assert L == 2, "This function currently only works for L=2 signals"
    return jax.jacfwd(f_j_sparse_diff_old, 0)(s, δ, signal_prior, ν̂, κ_B).reshape((L, L))
# apply dfdB_j row-wise to s, stack results row-wise
dfdB_sparse_diff_old = jax.jit(jax.vmap(dfdB_j_sparse_diff_old, 
                            (0, None, None, None, None), 0), static_argnames='signal_prior')


def f_j_sparse_diff(s, signal_prior: SparseDiffSignal, ν̂, κ_B):
    """General version of f_j_sparse_diff_old. Allows any L>=2."""
    L = ν̂.shape[0]
    num_configs = 2**(L-1)
    cov_seq = signal_prior.cov_seq
    is_change_seq = signal_prior.is_change_seq
    prob_seq = signal_prior.prob_seq
    # print(f"cov_seq.shape = {cov_seq.shape}; \
    #       is_change_seq.shape = {is_change_seq.shape}; \
    #         prob_seq.shape = {prob_seq.shape}")
    assert cov_seq.shape[0] == is_change_seq.shape[0] == len(prob_seq) == num_configs
    def one_case(i_case):
        Σ_V = ν̂.T @ cov_seq[i_case] @ ν̂ + κ_B
        χ = (cov_seq[i_case] @ ν̂) @ jnp.linalg.pinv(Σ_V)
        assert χ.shape == (L, L)
        s_χ = s.reshape((1, L)) @ χ.T
        assert s_χ.shape == (1, L)
        return log_N(s.reshape((L, 1)), Σ_V), s_χ # scalar, then 1xL vector
    log_N_arr, s_χ_arr = jax.vmap(one_case, 0, 0)(jnp.arange(num_configs))
    log_N_arr = log_N_arr.reshape((num_configs, 1))
    log_N_arr = log_N_arr - jnp.max(log_N_arr, axis=0) # subtract max term to avoid overflow
    # print(f"s_χ_arr.shape = {s_χ_arr.shape}")
    s_χ_arr = s_χ_arr.reshape(num_configs, L)

    log_num_arr, num_sign_arr = logsumexp(
        a=log_N_arr, axis=0, b=prob_seq.reshape((num_configs, 1)) * s_χ_arr, 
        return_sign=True)
    log_denom_arr, sign_denom_arr = logsumexp(
        a=log_N_arr, axis=0, b=prob_seq.reshape((num_configs, 1)), return_sign=True)
    res = (num_sign_arr * sign_denom_arr * jnp.exp(log_num_arr - log_denom_arr)
           ).reshape((1, L))
    return res
f_sparse_diff = jax.jit(jax.vmap(f_j_sparse_diff, \
                (0, None, None, None), 0), static_argnames='signal_prior') # apply f_j row-wise to s, stack results row-wise

def dfdB_j_sparse_diff(s, signal_prior: SparseDiffSignal, ν̂, κ_B):
    L = ν̂.shape[0]
    return jax.jacfwd(f_j_sparse_diff, 0)(s, signal_prior, ν̂, κ_B).reshape((L, L))    
dfdB_sparse_diff = jax.jit(jax.vmap(dfdB_j_sparse_diff,
                            (0, None, None, None), 0), static_argnames='signal_prior')

def form_Σ(idx, σ, ρ, ν, κ_T):
    """
    Construct the (L+1)-by-(L+1) covariance matrix of
    [(V_{Θ,i}^t)^T, bar{Y_i}] in R^{L+1} conditioned on 
    c_i = idx, where idx in [0,L-1]. Checked. 
    """
    # assert idx in range(L) # this line causes jit to break
    L = ρ.shape[0]
    Σ_V_Y = jnp.block(
        [
            [ν.T @ jax.scipy.linalg.inv(ρ) @ ν + κ_T, ν[idx, :].reshape(L, 1)],
            [ν[idx, :].reshape(1, L), ρ[idx, idx] + σ**2]
        ]
    )
    # Σ_V_Y = jnp.block(
    #     [
    #         [ν, ν[idx, :].reshape(L, 1)],
    #         [ν[idx, :].reshape(1, L), ρ[idx, idx] + σ**2]
    #     ]
    # )
    # jax.debug.print(f"eigenvals of Σ_V_Y = {jnp.linalg.eigh(Σ_V_Y)[0]}")
    # print(is_pos_semi_def_scipy(Σᵢ))
    # assert np.all(np.linalg.eigvals(Σᵢ) >= 0) # semi-positive definite, 
    # also np.linalg.eigvals isn't numerically accurate so avoid using it
    return Σ_V_Y  
# Apply form_Σ to each idx in [0,L-1], stack the L Σᵢ's along axis 0 i.e. returns Lx(L+1)x(L+1) tensor:
# form_Σ_all = jax.jit(jax.vmap(form_Σ, (0, None, None, None, None), 0)) 
form_Σ_all = jax.vmap(form_Σ, (0, None, None, None, None), 0)

def form_Σ_λ_all(V, u, σ, ρ, ν, κ_T):
    """
    Calculate Σ = Cov([(V_{Θ,i}^t)^T, bar{Y_i}] | c_i=idx).
    Calculate λ_idx = E[Z_{B,i} | V_{Θ,i}^t=V, bar{Y}_i=u, c_i = idx] 
    for idx in [0,L-1].

    Inputs:
    V is 1xL (L, ), u is scalar.
    σ is scalar noise standard deviation, ρ, ν, κ_T in R^{LxL}.
    Output λ is Lx1. Checked. 
    """
    L = ρ.shape[0]
    idx_arr = jnp.arange(L)
    Σ_s = form_Σ_all(idx_arr, σ, ρ, ν, κ_T) # L signal indices along axis 0
    # Σ_s_is_all_psd = jnp.all(jnp.array([is_pos_semi_def_scipy(Σ_s[i]) for i in range(L)]))
    assert Σ_s.shape == (L, L+1, L+1)
   
    def λᵢ(idx):
        term1 = jnp.concatenate((ν, ρ[:, idx].reshape(L, 1)), axis=1) # Lx(L+1)
        term2 = jnp.linalg.pinv(Σ_s[idx, :, :]) @ jnp.concatenate((V.T, u)) # (L+1)x1
        # term2 = jax.scipy.linalg.solve(Σ_s[idx, :, :], jnp.concatenate((V.T, u)), assume_a='her') # (L+1)x1
        return term1 @ term2 # Lx1
    # Apply λᵢ to each idx in [0,L-1], stack the L λᵢ's along axis 0 i.e. returns an LxL matrix:
    # λ_all = jax.jit(jax.vmap(λᵢ, 0, 0))
    λ_all = jax.vmap(λᵢ, 0, 0)
    λ_s = λ_all(idx_arr) # L signal indices along axis 0
    assert λ_s.shape == (L, L)
    # jax.debug.print(f"λ_s = {λ_s}")
    return Σ_s, λ_s

def form_Σ_λ_all(V, u, σ, ρ, ν, κ_T):
    """
    Calculate Σ = Cov([(V_{Θ,i}^t)^T, bar{Y_i}] | c_i=idx).
    Calculate λ_idx = E[Z_{B,i} | V_{Θ,i}^t=V, bar{Y}_i=u, c_i = idx] 
    for idx in [0,L-1].

    Inputs:
    V is 1xL (L, ), u is scalar.
    σ is scalar noise standard deviation, ρ, ν, κ_T in R^{LxL}.
    Output λ is Lx1. Checked. 
    """
    L = ρ.shape[0]
    idx_arr = jnp.arange(L)
    Σ_s = form_Σ_all(idx_arr, σ, ρ, ν, κ_T) # L signal indices along axis 0
    # Σ_s_is_all_psd = jnp.all(jnp.array([is_pos_semi_def_scipy(Σ_s[i]) for i in range(L)]))
    assert Σ_s.shape == (L, L+1, L+1)
   
    def λᵢ(idx):
        term1 = jnp.concatenate((ν, ρ[:, idx].reshape(L, 1)), axis=1) # Lx(L+1)
        term2 = jnp.linalg.pinv(Σ_s[idx, :, :]) @ jnp.concatenate((V.T, u)) # (L+1)x1
        # term2 = jax.scipy.linalg.solve(Σ_s[idx, :, :], jnp.concatenate((V.T, u)), assume_a='her') # (L+1)x1
        return term1 @ term2 # Lx1
    # Apply λᵢ to each idx in [0,L-1], stack the L λᵢ's along axis 0 i.e. returns an LxL matrix:
    # λ_all = jax.jit(jax.vmap(λᵢ, 0, 0))
    λ_all = jax.vmap(λᵢ, 0, 0)
    λ_s = λ_all(idx_arr) # L signal indices along axis 0
    assert λ_s.shape == (L, L)
    # jax.debug.print(f"λ_s = {λ_s}")
    return Σ_s, λ_s

def E_Z(V, u, ϕ_j, σ, ρ, ν, κ_T):
    """ 
    Calculate E[Z_{B,i} | V_{Theta,i}^t=V, bar{Y}_i=u].

    Inputs:
    V is 1xL, u is scalar.
    σ is scalar noise standard deviation, ρ, ν, κ_T in R^{LxL}.

    Output is 1xL, and can be positive or negative.
    """
    L = ρ.shape[0]
    ϕ_s = ϕ_j.reshape((L, 1)) 
    μ_s = V.reshape(L,1)
    σ_sq_s = (ρ-ν).diagonal()
    print(σ_sq_s)

    Σ_s, λ_s = form_Σ_λ_all(V, u, σ, ρ, ν, κ_T) # L signal indices along axis 0
    x = jnp.append(V.T, u).reshape(L+1, 1)  # Tested
    # jax.debug.print(f"Σ_s = {Σ_s}")
    # jax.debug.print(f"λ_s = {λ_s}")
    # jax.debug.print(f"x = {x}")
    # Σ_s is Lx(L+1)x(L+1), λ_s is LxL

    log_N_s = log_N_all(x, Σ_s).reshape((L, 1))
    # Subtract the max term to avoid numerical over/underflow:
    max_log_N = jnp.max(log_N_s, axis=0)
    log_N_s = log_N_s - max_log_N

    log_num_arr, sign_num_arr = logsumexp(
        a=log_N_s, axis=0, b=λ_s * ϕ_s, return_sign=True)
    log_denom, sign_denom_arr = logsumexp(a=log_N_s, b=ϕ_s, axis=0, return_sign=True)
    # jax.debug.print(f"sign_denom_arr = {sign_denom_arr}")
    # log_num_arr = jnp.maximum(log_num_arr, -1e5) # avoids numerical issues when ϕ_s contains a zero. 
    # log_denom = jnp.maximum(log_denom, -1e5) # avoids numerical issues
    res = (sign_num_arr * sign_denom_arr * jnp.exp(log_num_arr - log_denom)).reshape((1, L)) 
    return res


def g_j(V, u, ϕ_j, σ, ρ, ν, κ_T):
    """ 
    Calculate gt applied to one row: R^L * R -> R^L.
    V is 1-by-L, u is scalar.
    Output is 1-by-L.

    V, u are input arguments to g. And ρ, ν, κ_T are the parameters 
    defining the form of g. For this reason, ρ, ν, κ_T should be based
    on estimated signal prior.
    """
    # jax.debug.print("Actual u.shape: {x}", x = u.shape)
    # jax.debug.print("Actual u: {x}", x = u)
    ξ = jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T)
    # tmp tends to have very large condition number, so we use pinv instead of solve:
    # tmp = ν.T @ jax.scipy.linalg.inv(ρ) @ ν + κ_T
    return (E_Z(V, u, ϕ_j, σ, ρ, ν, κ_T) - V @ ξ.T @ ν.T) @ jnp.linalg.pinv(ρ - ν.T @ ξ.T @ ν)
    # return (E_Z(V, u, ϕ_j, σ, ρ, ν, κ_T) - V) @ jnp.linalg.pinv(ρ - ν)
# g = jax.jit(jax.vmap(g_j, in_axes=(
#     0, 0, 1, None, None, None, None), out_axes=0))
g = jax.jit(jax.vmap(g_j, in_axes=(
    0, 0, 1, None, None, None, None), out_axes=0))
# g = jax.vmap(g_j, in_axes=(
#     0, 0, 1, None, None, None, None), out_axes=0)

def dgdθ_j(V, u, ϕ_j, σ, ρ, ν, κ_T): 
    L = ρ.shape[0]
    return jax.jacfwd(g_j, 0)(V, u, ϕ_j, σ, ρ, ν, κ_T).reshape((L, L))
    # output from jax.jacfwd is LxL, where num cols = input (V) dim, num rows = output (g) dim
    # that is Jij = dgi/dVj
# dgdθ = jax.jit(jax.vmap(dgdθ_j, in_axes=(
#     0, 0, 1, None, None, None, None), out_axes=0)) # nxLxL
dgdθ = jax.jit(jax.vmap(dgdθ_j, in_axes=(
    0, 0, 1, None, None, None, None), out_axes=0))
# dgdθ = jax.vmap(dgdθ_j, in_axes=(
#     0, 0, 1, None, None, None, None), out_axes=0)

def g_Z_j(V, Z, c_j, ϕ_j, σ, ρ, ν, κ_T):
    L = ρ.shape[0]
    # jax.debug.print("q().shape: {x}", x = q_j_sample(Z, c_j, σ, L).shape)
    # jax.debug.print("q(): {x}", x = q_j_sample(Z, c_j, σ, L))
    return g_j(V, q_j_sample(Z, c_j, σ, L)[0], ϕ_j, σ, ρ, ν, κ_T)

def dgdZ_j(V_j, Z_j, c_j, ϕ_j, σ, ρ, ν, κ_T):
    L = ρ.shape[0]
    return jax.jacfwd(g_Z_j, 1)(V_j, Z_j, c_j, ϕ_j, σ, ρ, ν, κ_T).reshape((L, L)) 
dgdZ_j_jit = jax.jit(dgdZ_j, static_argnames=("σ")) # LxL
dgdZ = jax.jit(jax.vmap(dgdZ_j, in_axes=(0, 0, 0, 1, None, None, None, None), out_axes=0), static_argnames=("σ")) # nxLxL

def κ_B_marginal(ϕ, ρ, n, σ, ν, κ_T, ρ_est = None, num_samples = 1000):
    """
    ν̂ is E_{V, Z_B, Ψ}[1/n * g(V, Ȳ).T*g(V, Ȳ)] where V = Z_B*ρ^{-1}*ν + G
    with G ~ N(ν, κ_T) and Ȳ = q(Z_B, Ψ)).
    This function draws multiple samples of V and multiple samples of Z_B, and all
    changepoint configs Ψ to calculate ν̂.
    
    note E_{V, Z_B, Ψ}[1/n * g(V, Ȳ).T*g(V, Ȳ)] 
        = 1/n sum_{i=1}^n * E_{V, Z_B, Ψ}[g_i(V_i, Ȳ_i).T * g_i(V_i, Ȳ_i)]
    """

    if ρ_est is None:
        ρ_est = ρ
    L = ρ.shape[0]
    κ_T_symm = (κ_T + κ_T.T) / 2 # To avoid numerical issues.
    Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
    V = Z_B @ jnp.linalg.inv(ρ) @ ν + jnp.array(nprandom.multivariate_normal(
        jnp.zeros(L), κ_T_symm, size=(n, num_samples)))
    
    ϕ_and_j = jnp.concatenate((ϕ, jnp.arange(n).reshape((1, n)).astype(int)), axis=0) # (L+1)xn

    def E(ϕ_and_j_col_j): 
        """
        Calculates E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)] with 
        all samples of V_j and Z_B,j and all possible ϕ_j.

        Input is the j-th column of ϕ_and_j, of dim (L+1)x1.
        """
        ϕ_j = ϕ_and_j_col_j[:-1]
        j = (ϕ_and_j_col_j[-1]).astype(int)

        assert ϕ_j.shape == (L, )
        def sm(c_j):
            """Calculates one summand of E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)]
            i.e. for one changepoint config c_j."""
            # u = jax.vmap(q_j_sample, in_axes=(0, None, None), out_axes=0)(Z_B, c_j, σ, L)
            u = q_j_sample(Z_B[j, :, :], c_j, σ, L)
            # g_j_ = g_j(V[j, :].flatten(), q_j(Z_B[j, :].reshape((1, L)), c_j, σ).sample(), ϕ_j, σ, ρ, ν, κ_T).reshape((1, L))
            # NOTE: input V, u to g are sampled according to the true signal prior, but g itself
            # is defined using the estimated signal prior ρ_est.
            g_j_full = jax.vmap(g_j, in_axes=(0, 0, None, None, None, None, None),
                             out_axes=0)(
                V[j, :, :], u, ϕ_j, σ, jnp.array(ρ_est), ν, κ_T).reshape((num_samples, L))
            return ϕ_j[c_j] * (1 / num_samples * g_j_full.T @ g_j_full)

        return jnp.sum(jax.vmap(sm, 0, 0)(jnp.arange(L)), axis = 0) # sum over c_j
        # sum over all L signal indices c_j for the j-th observation. dim LxL

    return 1/n * jnp.sum(jax.vmap(E, 1, 0)(ϕ_and_j), axis=0) # average over all n g_j's. dim LxL

def ν̂_marginal(ϕ, ρ, n, σ, ν, κ_T, ρ_est = None, num_samples=1000):

    if ρ_est is None:
        ρ_est = ρ
    L = ρ.shape[0]
    κ_T_symm = (κ_T + κ_T.T) / 2 # To avoid numerical issues.
    # num_samples = 300
    Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
    V = Z_B @ jnp.linalg.inv(ρ) @ ν + jnp.array(nprandom.multivariate_normal(
        jnp.zeros(L), κ_T_symm, size=(n, num_samples)))

    ϕ_and_j = jnp.concatenate((ϕ, jnp.arange(n).reshape((1, n)).astype(int)), axis=0) # (L+1)xn

    def E(ϕ_and_j_col_j): 
        """
        Calculates E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)] with 
        all samples of V_j and Z_B,j and all possible ϕ_j.

        Input is the j-th column of ϕ_and_j, of dim (L+1)x1.
        """
        ϕ_j = ϕ_and_j_col_j[:-1]
        j = (ϕ_and_j_col_j[-1]).astype(int)

        assert ϕ_j.shape == (L, )
        def sm(c_j):
            dgdZ_AD_full = jax.vmap(dgdZ_j_jit, in_axes = (0, 0, None, None, None, None, None, None), out_axes=0)(V[j, :, :], Z_B[j, :, :], c_j, ϕ_j, σ, ρ_est, ν, κ_T).reshape((num_samples, L, L))
            return ϕ_j[c_j] * np.mean(dgdZ_AD_full, axis = 0)

        return jnp.sum(jax.vmap(sm, 0, 0)(jnp.arange(L)), axis = 0) # sum over c_j
        # sum over all L signal indices c_j for the j-th observation. dim LxL

    return 1/n * jnp.sum(jax.vmap(E, 1, 0)(ϕ_and_j), axis=0) # average over all n g_j's. dim LxL    

def GAMP_gaussian_clean(B̂_0, δ, p, ϕ_, L, σ, X, Y, ρ, T, prior = None, verbose=False, seed=None, tqdm_disable = False):
    """
    Using optimal denoisers, run GAMP for T iterations.
    B̃ is the ground truth
    B̂_0 is the initial estimate
    """
    if seed is not None:
        nprandom.seed(seed)

    ϕ = amp.signal_configuration.pad_marginal(ϕ_) # Pad the marginal's zeros for numerical stability
    n = int(δ * p)

    B̂ = B̂_0
    ν = jnp.zeros((L, L)) 
    κ_T = ρ  
    F = jnp.eye(L)  # Checked
    R̂ = jnp.zeros((n, L))  # Checked
    B̂_prev = B̂_0

    for t in tqdm(range(T), disable = tqdm_disable):
        # print(f"== GAMP iteration {t} ==")
        if verbose:
            print("ν: ", ν)
            print("κ_T: ", κ_T)

        ## -- AMP -- ##
        Θ_t = X @ B̂ - R̂ @ F.T

        ## -- g and its parameters -- ##
        R̂ = g(Θ_t, Y, ϕ, σ, ρ, ν, κ_T).reshape((n, L))

        if jnp.isnan(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains nans===')
            break
        elif jnp.isinf(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains infs===')
            break
        
        dgdθ_AD = dgdθ(Θ_t, Y, ϕ, σ, ρ, ν, κ_T) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
        assert dgdθ_AD.shape == (n, L, L)
        C_AD = 1/n * jnp.sum(dgdθ_AD, axis=0) # sum over n gives LxL matrix,
        C = C_AD
        if verbose:
            print("C_AD: ", C_AD)
      
        if verbose:
            print("C: ", C)
        B_t = X.T @ R̂ - B̂ @ C.T

        ## -- f and its parameters -- ##
        ν̂ = κ_B_marginal(ϕ, ρ, n, σ, ν, κ_T, num_samples = 400)
        assert not jnp.isnan(ν̂).any(), "ν̂ contains nans"
        atol = 1e-16 if config.jax_enable_x64 else 1e-7
        assert np.allclose(ν̂, ν̂.T, atol=atol), "ν̂ is not symmetric"

        if verbose:
            print("ν̂: ", ν̂)
        κ_B = ν̂

        B̂ = f(B_t, δ, ρ, ν̂, κ_B).reshape((p, L))
        if jnp.isnan(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains nans===')
            B̂ = B̂_prev
            break
        elif jnp.isinf(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains infs===')
            B̂ = B̂_prev
            break

        χ = ρ @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
        F = 1/δ * χ.T

        # Closed form gaussian case
        ν = ρ @ ν̂ @ χ.T
        κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν

        B̂_prev = B̂

    return B̂, Θ_t, ν, ν̂ 

def GAMP_full(B̃, δ, p, ϕ_, σ, X, Y, C_true, T = 10, B̂_0 = None, 
         true_signal_prior: SignalPrior=GaussianSignal(np.eye(2)), 
         est_signal_prior = None, st_ζ = None,
         verbose = False, seed = None, tqdm_disable = False):
    """
    signal_prior is the estimated signal prior, which may differ from the true signal prior.
    Y is observation of the true signal.
    GAMP runs on Y assuming signal_prior.
    
    The denoising functions ft and gt are Bayes-optimal for est_signal_prior.

    st_ζ: the threshold for the soft-thresholding function when est_signal_prior
    is sparse.
    C_true is only used to construct fixed-C SE alongside our AMP.
    TODO: use C_true to run fixed-C SE (this is simply running SE alongside AMP
    for a fixed C).
    """

    if seed is not None:
        nprandom.seed(seed)

    if est_signal_prior is None:
        est_signal_prior = true_signal_prior

    if st_ζ is not None:
        assert type(est_signal_prior) == SparseGaussianSignal, \
            "st_ζ should only be used for sparse signal priors"

    n = int(δ * p)
    ϕ = amp.signal_configuration.pad_marginal(ϕ_) # Pad the marginal's zeros for numerical stability
    L = true_signal_prior.L
    ρ = jnp.array(1/δ * true_signal_prior.cov)
    ρ_est = jnp.array(1/δ * est_signal_prior.cov)

    if B̂_0 is None: 
        # NOTE: old version used true_signal_prior
        B̂ = est_signal_prior.sample(p) # Sample B̂_0 from the prior
    else:
        B̂ = B̂_0
    # ν = all zero implies initial estimate has zero correlation with true signal B:
    ν = jnp.zeros((L, L)) 
    # κ_T = 1/δ * covariance of f_0(B_0) = B̂_0:
    # NOTE: old version used κ_T = ρ instead of ρ_est
    κ_T = ρ_est 
    F = jnp.eye(L)  # Checked
    R̂ = jnp.zeros((n, L))  # Checked

    sq_corr_amp = np.zeros((L, T))
    sq_corr_se = np.zeros((L, T))
    ν_arr = np.zeros((T+1, L, L))
    κ_T_arr = np.zeros((T+1, L, L))
    ν̂_arr = np.zeros((T, L, L))
    κ_B_arr = np.zeros((T, L, L))
    ν_arr[0] = ν
    κ_T_arr[0] = κ_T

    # Initialise for fixed-C SE:
    ν_fixed = jnp.zeros((L, L))
    κ_T_fixed = ρ_est

    sq_corr_se_fixed = np.zeros((L, T))
    ν_fixed_arr = np.zeros((T+1, L, L))
    κ_T_fixed_arr = np.zeros((T+1, L, L))
    ν̂_fixed_arr = np.zeros((T, L, L))
    κ_B_fixed_arr = np.zeros((T, L, L))
    ν_fixed_arr[0] = ν_fixed
    κ_T_fixed_arr[0] = κ_T_fixed
    Θ_t_arr = np.zeros((T, n, L))
    for t in tqdm(range(T), disable = tqdm_disable):

        ## -- AMP -- ##
        Θ_t = X @ B̂ - R̂ @ F.T # nxL
        Θ_t_arr[t] = Θ_t

        ## -- g and its parameters -- unchanged for sparse or sparse difference prior ##
        # NOTE: old version used ρ instead of ρ_est
        R̂ = g(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T).reshape((n, L))

        if jnp.isnan(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains nans===')
            break
        elif jnp.isinf(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains infs===')
            break
        
        # NOTE: old version used ρ instead of ρ_est
        dgdθ_AD = dgdθ(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
        assert dgdθ_AD.shape == (n, L, L)
        C = 1/n * jnp.sum(dgdθ_AD, axis=0) # sum over n gives LxL matrix,
        B_t = X.T @ R̂ - B̂ @ C.T

        ## -- f and its parameters -- adapted for sparse or sparse difference prior ##
        # ν̂ = ν̂_marginal(ϕ, ρ, n, σ, ν, κ_T)
        # NOTE: old version did not input ρ_est
        κ_B = κ_B_marginal(ϕ, ρ, n, σ, ν, κ_T, ρ_est)
        # print(f'κ_B avg = {κ_B}')
        ν̂ = κ_B
        ν̂_arr[t] = ν̂
        κ_B_arr[t] = κ_B
        #################### fixed-C ν̂, κ_B ######################
        num_samples = 2000
        
        # Draw input arguments to g according to fixed-C params:
        Z_B_fixed = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
        V_θ_fixed = Z_B_fixed @ jnp.linalg.inv(ρ) @ ν_fixed + jnp.array(nprandom.multivariate_normal(
            jnp.zeros(L), κ_T_fixed, size=(n, num_samples)))
        # Note ϕ is the full marginal for defining g, rather than as an input argument to g.
        # All params defining g are the random-C version:
        assert Z_B_fixed.shape == (n, num_samples, L)
        assert V_θ_fixed.shape == (n, num_samples, L)
        assert C_true.shape == (n, )
        def sm(ϕ_j, j, c_j):
            """Calculates one summand of E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)]
            i.e. for one changepoint config c_j."""
            # Z_B_fixed[j, :, :] is num_samples x L
            u = q_j_sample(Z_B_fixed[j, :, :], c_j, σ, L)
            # g_j_ = g_j(V[j, :].flatten(), q_j(Z_B[j, :].reshape((1, L)), c_j, σ).sample(), ϕ_j, σ, ρ, ν, κ_T).reshape((1, L))
            # NOTE: input V, u to g are sampled according to the true signal prior, but g itself
            # is defined using the estimated signal prior ρ_est.
            g_j_full = jax.vmap(g_j, in_axes=(0, 0, None, None, None, None, None), 
                            out_axes=0)(
            V_θ_fixed[j, :, :], u, ϕ_j, σ, ρ_est, ν, κ_T).reshape((num_samples, L))
            return 1 / num_samples * g_j_full.T @ g_j_full # LxL
        κ_B_fixed_elements = jax.vmap(sm, in_axes=0, out_axes=0)(ϕ.T, jnp.arange(n), C_true)
        assert κ_B_fixed_elements.shape == (n, L, L)
        κ_B_fixed = jnp.mean(κ_B_fixed_elements, axis=0)
        assert κ_B_fixed.shape == (L, L)
        if False:
            def estimate_κ_B(V, Z):
                g_fixed = g(V, q(Z, C_true, σ), ϕ, σ, ρ_est, ν, κ_T).reshape((n, L))
                return 1/n * g_fixed.T @ g_fixed
            κ_B_fixed = jnp.mean(jax.vmap(estimate_κ_B, in_axes=(1, 1), \
                                out_axes=0)(V_θ_fixed, Z_B_fixed), axis=0)
        if False:
            κ_B_fixed = 1/n * R̂.T @ R̂
        
        # print(f'κ_B_fixed = {κ_B_fixed}')
        # Since g is Bayes optimal for random-C instead of fixed C,
        # ν̂_fixed doesnt equal to κ_B_fixed:
        def estimate_ν̂(V, Z):
            dgdZ_ = dgdZ(V, Z, C_true, ϕ, σ, ρ_est, ν, κ_T)
            return 1/n * jnp.sum(dgdZ_, axis=0)
        ν̂_fixed = jnp.mean(jax.vmap(estimate_ν̂, in_axes=(1, 1), out_axes=0)(V_θ_fixed, Z_B_fixed), axis=0)

        κ_B_fixed_arr[t] = κ_B_fixed
        ν̂_fixed_arr[t] = ν̂_fixed
        ####################################################
        # TODO: make below a function
        if type(est_signal_prior) == SparseDiffSignal:
            B̂ = f_sparse_diff(B_t, est_signal_prior, ν̂, κ_B).reshape((p, L))    
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                B̂ = f_sparse(B_t, δ, est_signal_prior, ν̂, κ_B).reshape((p, L))
            else:
                B̂ = f_st(B_t, ν̂, κ_B, st_ζ).reshape((p, L))
        elif type(est_signal_prior) == GaussianSignal:
            B̂ = f(B_t, δ, ρ_est, ν̂, κ_B).reshape((p, L))
        else:
            raise ValueError(f"prior {est_signal_prior} not recognized")
          
        if jnp.isnan(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains nans===')
            break
        elif jnp.isinf(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains infs===')
            break

        # Calculate F 
        if type(est_signal_prior) == SparseDiffSignal:
            dfdB_AD = dfdB_sparse_diff(B_t, est_signal_prior, ν̂, κ_B)
            assert dfdB_AD.shape == (p, L, L)
            F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                dfdB_AD = dfdB_sparse(B_t, δ, est_signal_prior, ν̂, κ_B)
                assert dfdB_AD.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
            else:
                dfdB = dfdB_st(B_t, ν̂, κ_B, st_ζ)
                assert dfdB.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB, axis=0) # sum over p LxL matrices
                
        elif type(est_signal_prior) == GaussianSignal:
            # NOTE: old version used ρ instead of ρ_est.
            # The derivative of f doesnt involve the input to f so
            # the derivative only depends on the parameters defining f. 
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            F = 1/δ * χ.T # this simplification only holds for Gaussian prior

        # Better to use MC for both ν and κ_T.
        if False:
        # if type(est_signal_prior) == GaussianSignal:
            # Closed form gaussian case
            # NOTE: old version used ρ instead of ρ_est
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            ν = ρ @ ν̂ @ χ.T # it is ρ not ρ_est due to the true signal inputs
            κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν # it is ρ not ρ_est due to the true signal inputs

            # ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, signal_prior)
        else:
            ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, true_signal_prior, st_ζ=st_ζ)
            # ν = 1/δ * 1/p * B̂.T @ B̂ # This causes numerical issues, does not seem to improve results. 
            # κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν
        
        if verbose:
            print("κ_T_mc: ", κ_T)
            # NOTE: below should be valid but sometimes gives all negative entries
            κ_T_old = ν - ν.T @ jnp.linalg.inv(ρ) @ ν 
            print("κ_T optimal f* closed form: ", κ_T_old)
        
        ν_arr[t+1] = ν
        κ_T_arr[t+1] = κ_T
        #################### fixed-C ν, κ_T ######################
        num_samples = 1000
        B_i_samples = true_signal_prior.sample(num_samples * p)
        assert B_i_samples.shape == (num_samples * p, L)
        V_B_i_samples = B_i_samples @ ν̂_fixed + \
            np.random.multivariate_normal(np.zeros(L), κ_B_fixed, size=num_samples * p)
        if type(est_signal_prior) == GaussianSignal:
            f_V = f(V_B_i_samples, δ, ρ_est, ν̂, κ_B).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                f_V = f_sparse(V_B_i_samples, δ, est_signal_prior, ν̂, κ_B).reshape((num_samples*p, L))
            else:
                f_V = f_st(V_B_i_samples, ν̂, κ_B, st_ζ).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseDiffSignal:
            f_V = f_sparse_diff(V_B_i_samples, est_signal_prior, ν̂, κ_B).reshape((num_samples*p, L))
       
        ν_fixed = 1/δ * 1/(num_samples*p) * B_i_samples.T @ f_V
        # only f itself uses the estimated signal prior, all other terms use the true signal prior
        diff = f_V - B_i_samples @ jnp.linalg.inv(ρ) @ ν_fixed 
        κ_T_fixed = 1/δ * (1/(num_samples*p)) * diff.T @ diff
        ν_fixed_arr[t+1] = ν_fixed
        κ_T_fixed_arr[t+1] = κ_T_fixed

        ####################################################
        # if jnp.all(κ_T < 0):
        #     print("κ_T_gau: ", κ_T)
        #     print('=== EARLY TERMINATION: all entries of κ_T <0 ===')
        
        # if not jnp.allclose(κ_T, κ_T.T, atol=atol):
        #     print("κ_T_gau: ", κ_T)
        #     print('=== EARLY TERMINATION: κ_T is not symmetric ===')    
        for l in range(L):
            sq_corr_amp[l, t] = norm_sq_corr(B̂[:, l], B̃[:, l])
            sq_corr_se[l, t] = ν[l,l] ** 2 / \
                (ρ[l,l] * (κ_T + ν.T @ np.linalg.inv(ρ) @ ν)[l,l]) # involving true signal so use ρ
            # sq_corr_se[l, t] = ν[l,l] / ρ[l,l]
            sq_corr_se_fixed[l, t] = ν_fixed[l,l] ** 2 / \
                (ρ[l,l] * (κ_T_fixed + ν_fixed.T @ np.linalg.inv(ρ) @ ν_fixed)[l,l]) # involving true signal so use ρ
    return B̂, Θ_t, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr, ν_fixed_arr, κ_T_fixed_arr, ν̂_fixed_arr, κ_B_fixed_arr

def GAMP(B̃, δ, p, ϕ_, σ, X, Y, T = 10, B̂_0 = None, 
         true_signal_prior: SignalPrior=GaussianSignal(np.eye(2)), 
         est_signal_prior = None, st_ζ = None, num_samples=1000,
         verbose = False, seed = None, tqdm_disable = False):
    """
    signal_prior is the estimated signal prior, which may differ from the true signal prior.
    Y is observation of the true signal.
    GAMP runs on Y assuming signal_prior.
    
    The denoising functions ft and gt are Bayes-optimal for est_signal_prior.

    st_ζ: the threshold for the soft-thresholding function when est_signal_prior
    is sparse.

    num_samples are for estimating ν̂, κ_B, ν and κ_T.
    """

    if seed is not None:
        nprandom.seed(seed)

    if est_signal_prior is None:
        est_signal_prior = true_signal_prior

    if st_ζ is not None:
        assert type(est_signal_prior) == SparseGaussianSignal, \
            "st_ζ should only be used for sparse signal priors"

    n = int(δ * p)
    ϕ = amp.signal_configuration.pad_marginal(ϕ_) # Pad the marginal's zeros for numerical stability
    L = true_signal_prior.L
    ρ = jnp.array(1/δ * true_signal_prior.cov)
    ρ_est = jnp.array(1/δ * est_signal_prior.cov)

    if B̂_0 is None: 
        # NOTE: old version used true_signal_prior
        B̂ = est_signal_prior.sample(p) # Sample B̂_0 from the prior
    else:
        B̂ = B̂_0
    # ν = all zero implies initial estimate has zero correlation with true signal B:
    ν = jnp.zeros((L, L)) 
    # κ_T = 1/δ * covariance of f_0(B_0) = B̂_0:
    # NOTE: old version used κ_T = ρ instead of ρ_est
    κ_T = ρ_est 
    F = jnp.eye(L)  # Checked
    R̂ = jnp.zeros((n, L))  # Checked

    sq_corr_amp = np.zeros((L, T))
    sq_corr_se = np.zeros((L, T))
    ν_arr = np.zeros((T+1, L, L))
    κ_T_arr = np.zeros((T+1, L, L))
    ν̂_arr = np.zeros((T, L, L))
    κ_B_arr = np.zeros((T, L, L))
    ν_arr[0] = ν
    κ_T_arr[0] = κ_T

    Θ_t_arr = np.zeros((T, n, L))
    for t in tqdm(range(T), disable = tqdm_disable):

        ## -- AMP -- ##
        Θ_t = X @ B̂ - R̂ @ F.T # nxL
        Θ_t_arr[t] = Θ_t

        ## -- g and its parameters -- unchanged for sparse or sparse difference prior ##
        # NOTE: old version used ρ instead of ρ_est
        R̂ = g(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T).reshape((n, L))

        if jnp.isnan(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains nans===')
            break
        elif jnp.isinf(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains infs===')
            break
        
        # NOTE: old version used ρ instead of ρ_est
        dgdθ_AD = dgdθ(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
        assert dgdθ_AD.shape == (n, L, L)
        C = 1/n * jnp.sum(dgdθ_AD, axis=0) # sum over n gives LxL matrix,
        B_t = X.T @ R̂ - B̂ @ C.T

        ## -- f and its parameters -- adapted for sparse or sparse difference prior ##
        # ν̂ = ν̂_marginal(ϕ, ρ, n, σ, ν, κ_T)
        # NOTE: old version did not input ρ_est
        κ_B = κ_B_marginal(ϕ, ρ, n, σ, ν, κ_T, ρ_est, num_samples)
        # print(f'κ_B avg = {κ_B}')
        ν̂ = κ_B
        ν̂_arr[t] = ν̂
        κ_B_arr[t] = κ_B
        # TODO: make below a function
        if type(est_signal_prior) == SparseDiffSignal:
            B̂ = f_sparse_diff(B_t, est_signal_prior, ν̂, κ_B).reshape((p, L))    
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                B̂ = f_sparse(B_t, δ, est_signal_prior, ν̂, κ_B).reshape((p, L))
            else:
                B̂ = f_st(B_t, ν̂, κ_B, st_ζ).reshape((p, L))
        elif type(est_signal_prior) == GaussianSignal:
            B̂ = f(B_t, δ, ρ_est, ν̂, κ_B).reshape((p, L))
        else:
            raise ValueError(f"prior {est_signal_prior} not recognized")
          
        if jnp.isnan(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains nans===')
            break
        elif jnp.isinf(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains infs===')
            break

        # Calculate F 
        if type(est_signal_prior) == SparseDiffSignal:
            dfdB_AD = dfdB_sparse_diff(B_t, est_signal_prior, ν̂, κ_B)
            assert dfdB_AD.shape == (p, L, L)
            F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                dfdB_AD = dfdB_sparse(B_t, δ, est_signal_prior, ν̂, κ_B)
                assert dfdB_AD.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
            else:
                dfdB = dfdB_st(B_t, ν̂, κ_B, st_ζ)
                assert dfdB.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB, axis=0) # sum over p LxL matrices
                
        elif type(est_signal_prior) == GaussianSignal:
            # NOTE: old version used ρ instead of ρ_est.
            # The derivative of f doesnt involve the input to f so
            # the derivative only depends on the parameters defining f. 
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            F = 1/δ * χ.T # this simplification only holds for Gaussian prior

        # Better to use MC for both ν and κ_T.
        if False:
        # if type(est_signal_prior) == GaussianSignal:
            # Closed form gaussian case
            # NOTE: old version used ρ instead of ρ_est
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            ν = ρ @ ν̂ @ χ.T # it is ρ not ρ_est due to the true signal inputs
            κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν # it is ρ not ρ_est due to the true signal inputs

            # ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, signal_prior)
        else:
            ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, true_signal_prior, st_ζ=st_ζ, 
                              num_samples=num_samples)
            # ν = 1/δ * 1/p * B̂.T @ B̂ # This causes numerical issues, does not seem to improve results. 
            # κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν
        
        if verbose:
            print("κ_T_mc: ", κ_T)
            # NOTE: below should be valid but sometimes gives all negative entries
            κ_T_old = ν - ν.T @ jnp.linalg.inv(ρ) @ ν 
            print("κ_T optimal f* closed form: ", κ_T_old)
        
        ν_arr[t+1] = ν
        κ_T_arr[t+1] = κ_T
        for l in range(L):
            sq_corr_amp[l, t] = norm_sq_corr(B̂[:, l], B̃[:, l])
            sq_corr_se[l, t] = ν[l,l] ** 2 / \
                (ρ[l,l] * (κ_T + ν.T @ np.linalg.inv(ρ) @ ν)[l,l]) # involving true signal so use ρ
    if False:
        plt.figure()
        for l in range(L):
            plt.plot(sq_corr_amp[l, :], label = f"β{l} AMP", color=f'C{l}')
            plt.plot(sq_corr_se[l, :], label = f"β{l} SE", color=f'C{l}', linestyle='--')
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Squared correlation")
        plt.title("AMP")
        plt.show()
    # if jnp.all(κ_T < 0):
    #     print("κ_T_gau: ", κ_T)
    #     print('=== EARLY TERMINATION: all entries of κ_T <0 ===')
    
    # if not jnp.allclose(κ_T, κ_T.T, atol=atol):
    #     print("κ_T_gau: ", κ_T)
    #     print('=== EARLY TERMINATION: κ_T is not symmetric ===')    
    return B̂, Θ_t_arr, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr


def GAMP_real_data(δ, p, ϕ_, σ, X, Y, T,
         est_signal_prior, st_ζ = None, num_samples=1000,
         verbose = False, seed = None, tqdm_disable = False):
    """
    For processing real data.
    est_signal_prior cannot be None, true_signal_prior doesnt exist.
    this function doesnt run fixed-C SE.

    Y is observation of the true signal.
    GAMP runs on Y assuming signal_prior.
    
    The denoising functions ft and gt are Bayes-optimal for est_signal_prior.

    st_ζ: the threshold for the soft-thresholding function when est_signal_prior
    is sparse.
    C_true is only used to construct fixed-C SE alongside our AMP.
    """

    if seed is not None:
        nprandom.seed(seed)

    if st_ζ is not None:
        assert type(est_signal_prior) == SparseGaussianSignal, \
            "st_ζ should only be used for sparse signal priors"

    n = int(δ * p)
    ϕ = amp.signal_configuration.pad_marginal(ϕ_, ϵ=1e-6) # Pad the marginal's zeros for numerical stability
    L = est_signal_prior.L
    ρ_est = jnp.array(1/δ * est_signal_prior.cov)

    B̂ = est_signal_prior.sample(p) # Sample B̂_0 from the prior
  
    # ν = all zero implies initial estimate has zero correlation with true signal B:
    ν = jnp.zeros((L, L)) 
    # κ_T = 1/δ * covariance of f_0(B_0) = B̂_0:
    # NOTE: old version used κ_T = ρ instead of ρ_est
    κ_T = ρ_est 
    F = jnp.eye(L)  # Checked
    R̂ = jnp.zeros((n, L))  # Checked

    # sq_corr_amp = np.zeros((L, T))
    sq_corr_se = np.zeros((L, T))
    ν_arr = np.zeros((T+1, L, L))
    κ_T_arr = np.zeros((T+1, L, L))
    ν̂_arr = np.zeros((T, L, L))
    κ_B_arr = np.zeros((T, L, L))
    ν_arr[0] = ν
    κ_T_arr[0] = κ_T

    for t in tqdm(range(T), disable = tqdm_disable):

        ## -- AMP -- ##
        Θ_t = X @ B̂ - R̂ @ F.T

        ## -- g and its parameters -- unchanged for sparse or sparse difference prior ##
        R̂ = g(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T).reshape((n, L))

        if jnp.isnan(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains nans===')
            break
        elif jnp.isinf(R̂).any():
            print('=== EARLY TERMINATION: R̂ contains infs===')
            break
        if t ==0:
            dgdθ_AD = np.zeros((n, L, L))
        else:
            dgdθ_AD = dgdθ(Θ_t, Y, ϕ, σ, ρ_est, ν, κ_T) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
        assert dgdθ_AD.shape == (n, L, L)
        C = 1/n * jnp.sum(dgdθ_AD, axis=0) # sum over n gives LxL matrix,
        B_t = X.T @ R̂ - B̂ @ C.T

        ## -- f and its parameters -- adapted for sparse or sparse difference prior ##
        # ν̂ = ν̂_marginal(ϕ, ρ, n, σ, ν, κ_T)
        # NOTE: below uses ρ in GAMP, but ρ is not available so we use ρ_est instead.
        # κ_B = 1/n * R̂.T @ R̂
        κ_B = κ_B_marginal(ϕ, ρ_est, n, σ, ν, κ_T, ρ_est, num_samples)
        # κ_B = 1/n * R̂.T @ R̂       
        if verbose:
            print(f'κ_B avg = {κ_B}')
        
        if verbose:
            κ_B0 = 1/n * R̂.T @ R̂
            print(f'κ_B [R̂.T @ R̂] = {κ_B0}')
        # ν̂ = κ_B
        ν̂ = ν̂_marginal(ϕ, ρ_est, n, σ, ν, κ_T, ρ_est, num_samples)
        if verbose:
            print(f'ν̂ avg = {ν̂} (expect this to = κ_B)')
        ν̂_arr[t] = ν̂
        κ_B_arr[t] = κ_B
        # TODO: make below a function
        if type(est_signal_prior) == SparseDiffSignal:
            B̂ = f_sparse_diff(B_t, est_signal_prior, ν̂, κ_B).reshape((p, L))    
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                B̂ = f_sparse(B_t, δ, est_signal_prior, ν̂, κ_B).reshape((p, L))
            else:
                B̂ = f_st(B_t, ν̂, κ_B, st_ζ).reshape((p, L))
        elif type(est_signal_prior) == GaussianSignal:
            B̂ = f(B_t, δ, ρ_est, ν̂, κ_B).reshape((p, L))
        else:
            raise ValueError(f"prior {est_signal_prior} not recognized")
          
        if jnp.isnan(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains nans===')
            break
        elif jnp.isinf(B̂).any():
            print('=== EARLY TERMINATION: B̂ contains infs===')
            break

        # Calculate F 
        if type(est_signal_prior) == SparseDiffSignal:
            dfdB_AD = dfdB_sparse_diff(B_t, est_signal_prior, ν̂, κ_B)
            assert dfdB_AD.shape == (p, L, L)
            F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                dfdB_AD = dfdB_sparse(B_t, δ, est_signal_prior, ν̂, κ_B)
                assert dfdB_AD.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB_AD, axis=0) # sum over p LxL matrices
            else:
                dfdB = dfdB_st(B_t, ν̂, κ_B, st_ζ)
                assert dfdB.shape == (p, L, L)
                F = 1/n * jnp.sum(dfdB, axis=0) # sum over p LxL matrices
                
        elif type(est_signal_prior) == GaussianSignal:
            # NOTE: old version used ρ instead of ρ_est.
            # The derivative of f doesnt involve the input to f so
            # the derivative only depends on the parameters defining f. 
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            F = 1/δ * χ.T # this simplification only holds for Gaussian prior

        # Better to use MC for both ν and κ_T.
        if False:
        # if type(est_signal_prior) == GaussianSignal:
            # Closed form gaussian case
            # NOTE: old version used ρ instead of ρ_est
            χ = ρ_est @ ν̂ @ jnp.linalg.pinv(ν̂.T @ ρ_est @ ν̂ + 1/δ * κ_B) # derivative of ft; symmetric by definition
            ν = ρ_est @ ν̂ @ χ.T # it is ρ not ρ_est due to the true signal inputs
            κ_T = ν - ν.T @ jnp.linalg.inv(ρ_est) @ ν # it is ρ not ρ_est due to the true signal inputs

            # ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, signal_prior)
        else:
            # NOTE: below uses true_signal_prior in GAMP.
            ν, κ_T = ν_κ_T_mc(δ, p, κ_B, ν̂, est_signal_prior, st_ζ=st_ζ, 
                              num_samples=num_samples)
            # ν = 1/δ * 1/p * B̂.T @ B̂ # This causes numerical issues, does not seem to improve results. 
            # κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν
        
        if verbose:
            print("κ_T_mc: ", κ_T)
            # NOTE: below should be valid but sometimes gives all negative entries
            # NOTE: below uses ρ in GAMP.
            κ_T_old = ν - ν.T @ jnp.linalg.inv(ρ_est) @ ν 
            print("κ_T optimal f* closed form: ", κ_T_old)
        
        ν_arr[t+1] = ν
        κ_T_arr[t+1] = κ_T
        
        # if jnp.all(κ_T < 0):
        #     print("κ_T_gau: ", κ_T)
        #     print('=== EARLY TERMINATION: all entries of κ_T <0 ===')
        
        # if not jnp.allclose(κ_T, κ_T.T, atol=atol):
        #     print("κ_T_gau: ", κ_T)
        #     print('=== EARLY TERMINATION: κ_T is not symmetric ===')    
        for l in range(L):
            # NOTE: below uses ρ in GAMP.
            # sq_corr_amp[l, t] = norm_sq_corr(B̂[:, l], B̃[:, l])
            sq_corr_se[l, t] = ν[l,l] ** 2 / \
                (ρ_est[l,l] * (κ_T + ν.T @ np.linalg.inv(ρ_est) @ ν)[l,l]) # involving true signal so use ρ
    
    if True:
        plt.figure()
        for l in range(L):
            # plt.plot(sq_corr_amp[l, :], label = f"β{l} AMP", color=f'C{l}')
            plt.plot(sq_corr_se[l, :], label = f"β{l} SE", color=f'C{l}', linestyle='--')
            # plt.plot(sq_corr_se_fixed[l, :], label = f"β{l} SE fixed-C", color=f'C{l}', linestyle=':')
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Squared correlation")
        plt.title("AMP")
        # plt.show()
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(f"singlecell/SE_sq_corr_{timestamp}.pdf")
    return B̂, Θ_t, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr

def ν_num_int(δ, ρ, p, ν̂, κ_B):
    """ Compute ν using numerical integration. Scipy only allows returning a scalar from the integral. """

    L = ρ.shape[0]
    def gaussian_pdf(x, mean, cov):
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    def fn(*s, δ, ρ, ν̂, κ_B, i, j): 
        f_ = np.array(f_j(s, δ, ρ, ν̂, κ_B).reshape((1, L)))
        return (f_.T @ f_)[i, j] * gaussian_pdf(s, np.zeros(ρ.shape[0]), np.array(ν̂.T @ (δ * ρ) @ ν̂ + κ_B))

    res = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            full_res = scipy.integrate.nquad(fn, [(-10**2, 10**2) for l in range(L)], args=(δ, ρ, ν̂, κ_B, i, j))
            abs_err = full_res[1]
            jax.debug.print("abs_err: {x}", x = abs_err)
            res[i, j] = full_res[0]
    return res

def ν_κ_T_mc(δ, p, κ_B, ν̂, true_signal_prior: SignalPrior, \
             est_signal_prior = None, st_ζ= None, num_samples=1000):
    """
    Estimate ν, κ_T using Monte Carlo sampling.
    V_{B,i} = B_iν̂ + G_{B,i} where G_{B,i} ~ N(0, κ_B).

    NOTE: the estimate of κ_T uses ν.
    """

    if est_signal_prior is None:
        est_signal_prior = true_signal_prior
    if st_ζ is not None:
        assert type(est_signal_prior) == SparseGaussianSignal, \
            "st_ζ should only be used for sparse signal priors"
    
    L = κ_B.shape[0]
    # num_samples = 1000
    ρ = 1/δ * true_signal_prior.cov
    # Input arguments to f should be drawn from the true signal prior
    B_i_samples = true_signal_prior.sample(num_samples * p)
    assert B_i_samples.shape == (num_samples * p, L)
    V_B_i_samples = B_i_samples @ ν̂ + \
        np.random.multivariate_normal(np.zeros(L), κ_B, size=num_samples * p)
    assert V_B_i_samples.shape == (num_samples * p, L)
    if type(est_signal_prior) == SparseDiffSignal:
        f_V = f_sparse_diff(V_B_i_samples, est_signal_prior, ν̂, κ_B).reshape((num_samples*p, L))
    elif type(est_signal_prior) == SparseGaussianSignal:
        if st_ζ is None:
            f_V = f_sparse(V_B_i_samples, δ, est_signal_prior, ν̂, κ_B).reshape((num_samples*p, L))
        else:
            f_V = f_st(V_B_i_samples, ν̂, κ_B, st_ζ).reshape((num_samples*p, L))
    elif type(est_signal_prior) == GaussianSignal:
        f_V = f(V_B_i_samples, δ, (1/δ) * est_signal_prior.cov, ν̂, κ_B).reshape((num_samples*p, L))

    ############### calculate ν_mc = 1/δ * E[B_i.T*f_i(V_{B,i})] ###############
    ν = 1/δ * 1/(num_samples*p) * B_i_samples.T @ f_V
    # ν = 1/δ * (1/(num_samples*p)) * f_V.T @ f_V # TODO: check why this is not equal to B_i_samples.T @ f
    # jax.debug.print("ν_est_B: {x}", x = ν_est_B)
    # jax.debug.print("ν: {x}", x = ν)
    # jax.debug.print("np.max(ν_est_B - ν): {x}", x = np.max(ν_est_B - ν)) # Differ on the order of 1e-3 for SparseGaussianSignal, regardless of α and num_samples. Differs on the same order for GaussianSignal, and the AMP-SE still matches in that case.
    # ν = ν_est_B
    assert ν.shape == (L, L)

    ############### Using ν_mc, calculate κ_T_mc ###############
    # 1/δ * E[ (f_i(V_{B,i}) - B_iρ^{-1}ν).T * (f_i(V_{B,i}) - B_iρ^{-1}ν)]
    diff = f_V - B_i_samples @ jnp.linalg.inv(ρ) @ ν
    κ_T = 1/δ * (1/(num_samples*p)) * diff.T @ diff
    # assert np.allclose(κ_T, κ_T.T), "κ_T should be symmetric"
    # Closed form κ_T due to optimality of f
    # κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν
    # Symmetrize κ_T
    # κ_T = (κ_T + κ_T.T) / 2

    return ν, κ_T

def SE_fixed_C(C, true_signal_prior: SignalPrior, \
               est_signal_prior: SignalPrior, δ, p, ϕ_, L, σ, T, \
                tqdm_disable = False, st_ζ=None):
    """
    Use SE_fixed_C_v1 instead, which reuses results from GAMP so runs faster.    
        
    This computes the state evolution given a fixed changepoint, but
    using the optimal denoisers from the C-agnostic AMP. 
    This function runs the C-agnostic SE along side of fixed-C SE just to define 
    the parameters of the denoisers.

    NOTE the V_B and V_Θ parameters are the fixed-C state evolution parameters. 

    Moreover, when true_signal_prior != est_signal_prior, the optimal denoisers
    are designed according to the estimated signal prior. But the input arguments
    to the denoisers will be drawn from the true signal prior. 
    """

    ϕ = amp.signal_configuration.pad_marginal(ϕ_) # Pad the marginal's zeros for numerical stability
    n = int(δ * p)
    ρ = jnp.array(1/δ * true_signal_prior.cov)
    ρ_est = jnp.array(1/δ * est_signal_prior.cov)

    # Initialize with zero correlation
    ν_avg = jnp.zeros((L, L)) 
    κ_T_avg = ρ_est
    ν_fixed = jnp.zeros((L, L)) 
    κ_T_fixed = ρ_est 
    
    sq_corr_fixed_C = np.zeros((L, T))
    sq_corr_avg_C = np.zeros((L, T))
    for t in tqdm(range(T), disable = tqdm_disable):
        # SE params from averaged AMP, to be used as parameters for f^*, g^*. This does not change. 
        
        ################################## Parameters for g ################################
        # Estimate ν̂_fixed 
        # (for fixed C, g^* is no longer optimal because g^* is optimal for random C
        # so ν̂_fixed is no longer = κ_B.)
        num_samples = 2000
        # Actual inputs into g^*, V_θ, Z_B should follow the true signal prior. 
        # It is only g^* itself thats designed according to the estimated signal prior + random C.
        Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
        V_Θ = Z_B @ jnp.linalg.inv(ρ) @ ν_fixed + \
            jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T_fixed, size=(n, num_samples)))
        def estimate_ν̂(V, Z):
            # g designed using the estimated signal prior so ρ should be ρ_est. 
            # g designed assuming random C so ν should be ν_avg, κ_T should be κ_T_avg.
            # ϕ is just the ground truth (we're not assuming mismatch here).
            dgdZ_AD = dgdZ(V, Z, C, ϕ, σ, ρ_est, ν_avg, κ_T_avg) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
            # dgdZ_AD = dgdZ(V, Z, C, ϕ, σ, ρ, ν_fixed, κ_T_fixed) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
            ν̂_fixed_est = jnp.mean(dgdZ_AD, axis=0) # sum over n gives LxL matrix,
            assert ν̂_fixed_est.shape == (L, L)
            return ν̂_fixed_est
        # Only the input arguments (V_Θ, Z_B) to g are drawn from the true signal prior.
        ν̂_fixed = jnp.mean(jax.vmap(estimate_ν̂, in_axes=(1, 1), out_axes=0)(V_Θ, Z_B), axis=0)

        # Estimate ν̂_avg
        ν̂_avg = ν̂_marginal(ϕ, ρ, n, σ, ν_avg, κ_T_avg, ρ_est = ρ_est)

        # Estimate κ_B_fixed
        num_samples = 2000
        # Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
        # V_Θ = Z_B @ jnp.linalg.inv(ρ) @ ν_fixed + \
        #     jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T_fixed, size=(n, num_samples)))
        def estimate_κ_B(V, Z):
            # Again g is designed using the estimated signal prior so ρ should be ρ_est.
            # g is designed assuming random C so ν should be ν_avg, κ_T should be κ_T_avg.
            R̂ = g(V, q(Z, C, σ), ϕ, σ, ρ_est, ν_avg, κ_T_avg).reshape((n, L))
            κ_B_fixed_est = 1/n * R̂.T @ R̂
            assert κ_B_fixed_est.shape == (L, L)
            return κ_B_fixed_est
        # Only the input arguments (V_Θ, Z_B) to g are drawn from the true signal prior.
        κ_B_fixed = jnp.mean(jax.vmap(estimate_κ_B, in_axes=(1, 1), out_axes=0)(V_Θ, Z_B), axis=0)
        κ_B_fixed = (κ_B_fixed + κ_B_fixed.T)/2 # Force symmetry

        # Estimate κ_B_avg
        κ_B_avg = κ_B_marginal(ϕ, ρ, n, σ, ν_avg, κ_T_avg, ρ_est = ρ_est)

        ################################## Parameters for f ################################
        # Estimate χ_avg
        χ_avg = ρ @ ν̂_avg @ jnp.linalg.pinv(ν̂_avg.T @ ρ @ ν̂_avg + 1/δ * κ_B_avg) # derivative of ft in the Gaussian case; symmetric by definition

        # Estimate ν_fixed and κ_T_fixed
        # Sample from ground truth
        num_samples = 1000
        B_i_samples = true_signal_prior.sample(num_samples * p)
        assert B_i_samples.shape == (num_samples * p, L)
        V_B_i_samples = B_i_samples @ ν̂_fixed + \
            np.random.multivariate_normal(np.zeros(L), κ_B_fixed, size=num_samples * p)
        # Feed samples into f designed using the estimated signal prior
        if type(est_signal_prior) == GaussianSignal:
            f_V = f(V_B_i_samples, δ, ρ_est, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                f_V = f_sparse(V_B_i_samples, δ, est_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
            else:
                f_V = f_st(V_B_i_samples, ν̂_avg, κ_B_avg, st_ζ).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseDiffSignal:
            f_V = f_sparse_diff(V_B_i_samples, est_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        else:
            raise ValueError(f"est_signal_prior {est_signal_prior} not supported")
        # if type(true_signal_prior) == GaussianSignal: 
        #     # TODO: the line below should be est_signal_prior == true_signal_prior as an object.
        #     assert type(est_signal_prior) == GaussianSignal, \
        #         "Currently assume no mismatch in signal prior when true prior is GaussianSignal"

        #     # f designed assuming random C (so ν should be ν_avg, κ_T should be κ_T_avg),
        #     # and assuming true signal prior, no mismatch, (so ρ should be ρ).
        #     f_V = f(V_B_i_samples, δ, ρ, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))

        # elif type(true_signal_prior) == SparseGaussianSignal:
        #     if type(est_signal_prior) == GaussianSignal:
        #         f_V = f(V_B_i_samples, δ, ρ_est, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        #     else:
        #         assert type(est_signal_prior) == SparseGaussianSignal # no mismatch
        #         f_V = f_sparse(V_B_i_samples, δ, true_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
            
        # elif type(true_signal_prior) == SparseDiffSignal:
        #     if type(est_signal_prior) == GaussianSignal:
        #         f_V = f(V_B_i_samples, δ, ρ, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        #     else:
        #         assert type(est_signal_prior) == SparseDiffSignal # no mismatch
        #         f_V = f_sparse_diff(V_B_i_samples, true_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
            
        # else:
        #     raise ValueError(f"prior {true_signal_prior} not supported")
       
        # if type(true_signal_prior) == type(est_signal_prior):
        #     # no mismatch. f* is Bayes-optimal.
        #     ν_fixed = 1/δ * 1/(num_samples*p) * f_V.T @ f_V
        #     assert np.allclose(ν_fixed, ν_fixed.T), "ν_fixed should be symmetric"
        # else:
        ν_fixed = 1/δ * 1/(num_samples*p) * B_i_samples.T @ f_V
        # only f itself uses the estimated signal prior, all other terms use the true signal prior
        diff = f_V - B_i_samples @ jnp.linalg.inv(ρ) @ ν_fixed 
        κ_T_fixed = 1/δ * (1/(num_samples*p)) * diff.T @ diff

        # Estimate ν_avg
        if False:
        # if type(true_signal_prior) == GaussianSignal: 
            # No mismatch:
            ν_avg = ρ @ ν̂_avg @ χ_avg.T

            # Estimate κ_T_avg
            κ_T_avg = ν_avg - ν_avg.T @ jnp.linalg.inv(ρ) @ ν_avg # Can only use this equation because, for the fully averaged SE case, f is optimal. 
        else: 
            # Allow mismatch:
            ν_avg, κ_T_avg = ν_κ_T_mc(δ, p, κ_B_avg, ν̂_avg, \
                             true_signal_prior, est_signal_prior, st_ζ=st_ζ) # This we can keep as is, because it returns the averaged ν and κ_T (only used to construct f)
        for l in range(L):
            sq_corr_fixed_C[l, t] = (ν_fixed[l, l]**2 / \
                ( ρ[l, l] * (κ_T_fixed + ν_fixed.T @ np.linalg.inv(ρ) @ ν_fixed)[l, l]))
            sq_corr_avg_C[l, t] = (ν_avg[l, l]**2 / \
                ( ρ[l, l] * (κ_T_avg + ν_avg.T @ np.linalg.inv(ρ) @ ν_avg)[l, l]))
    if False:
        plt.figure()
        for l in range(L):
            plt.plot(sq_corr_fixed_C[l, :], label=f"β{l}")
        plt.xlabel("t")
        plt.ylabel("normalised squared correlation")
        plt.legend()
        plt.title("Fixed C SE")
        plt.show()

        plt.figure()
        for l in range(L):
            plt.plot(sq_corr_avg_C[l, :], label=f"β{l}")
        plt.xlabel("t")
        plt.ylabel("normalised squared correlation")
        plt.legend()
        plt.title("Averaged C SE")
        plt.show()

    return ν_fixed, κ_T_fixed

def SE_fixed_C_v1(C_true, true_signal_prior: SignalPrior, \
        est_signal_prior: SignalPrior, \
        δ, p, ϕ_, L, σ, T, \
        ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = None, tqdm_disable = False):
    """
    Compared to SE_fixed_C, this function takes in the averaged-C SE parameters
    without re-running the average-C SE.
    
    When st_ζ is not None and est_signal_prior is SparseGaussianSignal,
    apply soft thresholding f.
    """
    if st_ζ is not None:
        assert type(est_signal_prior) == SparseGaussianSignal, \
            "st_ζ (soft threshold) should only be specified for SparseGaussianSignal"
    n = int(δ * p)
    assert C_true.shape == (n, )
    ϕ = amp.signal_configuration.pad_marginal(ϕ_) # Pad the marginal's zeros for numerical stability
    ρ = jnp.array(1/δ * true_signal_prior.cov)
    ρ_est = jnp.array(1/δ * est_signal_prior.cov)
    ν_fixed = jnp.zeros((L, L)) 
    ν_fixed_arr = np.zeros((T+1, L, L))
    ν_fixed_arr[0] = ν_fixed

    κ_T_fixed = ρ_est 
    κ_T_fixed_arr = np.zeros((T+1, L, L))
    κ_T_fixed_arr[0] = κ_T_fixed
    sq_corr_fixed_C = np.zeros((L, T))
    num_samples = 2000

    for t in (range(T)):
        ############### Estimate ν̂_fixed and κ_B_fixed, which involves g only #########
        ν_avg = jnp.array(ν_avg_arr[t])
        κ_T_avg = jnp.array(κ_T_avg_arr[t])
        # Actual inputs into g^*, V_θ, Z_B should follow the true signal prior. 
        # It is only g^* itself thats designed according to the estimated signal prior + random C.
        Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, num_samples)))
        V_Θ = Z_B @ jnp.linalg.inv(ρ) @ ν_fixed + \
            jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T_fixed, size=(n, num_samples)))
        assert Z_B.shape == (n, num_samples, L)
        assert V_Θ.shape == (n, num_samples, L)
        def estimate_ν̂(V, Z):
            # g designed using the estimated signal prior so ρ should be ρ_est. 
            # g designed assuming random C so ν should be ν_avg, κ_T should be κ_T_avg.
            # ϕ is just the ground truth (we're not assuming mismatch here).
            dgdZ_AD = dgdZ(V, Z, C_true, ϕ, σ, ρ_est, ν_avg, κ_T_avg) # nxLxL, axis 1 = output (g) dim, axis 2 = input (V) dim,    
            ν̂_fixed_est = jnp.mean(dgdZ_AD, axis=0) # sum over n gives LxL matrix,
            assert ν̂_fixed_est.shape == (L, L)
            return ν̂_fixed_est
        ν̂_fixed = jnp.mean(jax.vmap(estimate_ν̂, in_axes=(1, 1), out_axes=0)(V_Θ, Z_B), axis=0)

        ### Method 1 for estimating κ_B_fixed ###
        if False:
            def estimate_κ_B(V, Z):
                # Again g is designed using the estimated signal prior so ρ should be ρ_est.
                # g is designed assuming random C so ν should be ν_avg, κ_T should be κ_T_avg.
                R̂ = g(V, q(Z, C_true, σ), ϕ, σ, ρ_est, ν_avg, κ_T_avg).reshape((n, L))
                κ_B_fixed_est = 1/n * R̂.T @ R̂
                assert κ_B_fixed_est.shape == (L, L)
                return κ_B_fixed_est
            # Only the input arguments (V_Θ, Z_B) to g are drawn from the true signal prior.
            κ_B_fixed = jnp.mean(jax.vmap(estimate_κ_B, in_axes=(1, 1), out_axes=0)(V_Θ, Z_B), axis=0)
        
        ### Method 2 for estimating κ_B_fixed ###
        def sm(ϕ_j, j, c_j):
            """Calculates one summand of E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)]
            i.e. for one changepoint config c_j."""
            # Z_B_fixed[j, :, :] is num_samples x L
            u = q_j_sample(Z_B[j, :, :], c_j, σ, L)
            # g_j_ = g_j(V[j, :].flatten(), q_j(Z_B[j, :].reshape((1, L)), c_j, σ).sample(), ϕ_j, σ, ρ, ν, κ_T).reshape((1, L))
            # NOTE: input V, u to g are sampled according to the true signal prior, but g itself
            # is defined using the estimated signal prior ρ_est.
            g_j_full = jax.vmap(g_j, in_axes=(0, 0, None, None, None, None, None), 
                            out_axes=0)(V_Θ[j, :, :], u, ϕ_j, σ, \
                            ρ_est, ν_avg, κ_T_avg).reshape((num_samples, L))
            return 1 / num_samples * g_j_full.T @ g_j_full # LxL
        κ_B_fixed_elements = jax.vmap(sm, in_axes=0, out_axes=0)(ϕ.T, jnp.arange(n), C_true)
        assert κ_B_fixed_elements.shape == (n, L, L)
        κ_B_fixed = jnp.mean(κ_B_fixed_elements, axis=0)
        assert κ_B_fixed.shape == (L, L)
        κ_B_fixed = (κ_B_fixed + κ_B_fixed.T)/2 # Force symmetry

        ######### Estimate ν_fixed and κ_T_fixed, which involves f only #########
        ν̂_avg = jnp.array(ν̂_avg_arr[t])
        κ_B_avg = jnp.array(κ_B_avg_arr[t])
        B_i_samples = true_signal_prior.sample(num_samples * p)
        assert B_i_samples.shape == (num_samples * p, L)
        V_B_i_samples = B_i_samples @ ν̂_fixed + \
            np.random.multivariate_normal(np.zeros(L), κ_B_fixed, size=num_samples * p)
        # Feed samples into f designed using the estimated signal prior
        if type(est_signal_prior) == GaussianSignal:
            f_V = f(V_B_i_samples, δ, ρ_est, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseGaussianSignal:
            if st_ζ is None:
                f_V = f_sparse(V_B_i_samples, δ, est_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
            else:
                f_V = f_st(V_B_i_samples, ν̂_avg, κ_B_avg, st_ζ).reshape((num_samples*p, L))
        elif type(est_signal_prior) == SparseDiffSignal:
            f_V = f_sparse_diff(V_B_i_samples, est_signal_prior, ν̂_avg, κ_B_avg).reshape((num_samples*p, L))
        else:
            raise ValueError(f"est_signal_prior {est_signal_prior} not supported")
        
        ν_fixed = 1/δ * 1/(num_samples*p) * B_i_samples.T @ f_V
        ν_fixed_arr[t+1] = ν_fixed
        # only f itself uses the estimated signal prior, all other terms use the true signal prior
        diff = f_V - B_i_samples @ jnp.linalg.inv(ρ) @ ν_fixed 
        κ_T_fixed = 1/δ * (1/(num_samples*p)) * diff.T @ diff
        κ_T_fixed_arr[t+1] = κ_T_fixed

        for l in range(L):
            sq_corr_fixed_C[l, t] = ν_fixed[l, l]**2 / \
                ( ρ[l, l] * (κ_T_fixed + ν_fixed.T @ np.linalg.inv(ρ) @ ν_fixed)[l, l])
    if False:
        plt.figure()
        for l in range(L):
            plt.plot(sq_corr_fixed_C[l, :], label=f"β{l}")
        plt.xlabel("t")
        plt.ylabel("normalised squared correlation")
        plt.legend()
        plt.title("Fixed C SE (my version)")
        plt.show()
    return ν_fixed_arr, κ_T_fixed_arr