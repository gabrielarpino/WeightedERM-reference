import numpy.random as nprandom
import random
from amp.fully_separable import f_full
import amp.covariances
from tqdm.auto import tqdm
# from amp import ϵ_0
from jax.scipy.special import logsumexp
from jax.experimental import sparse
import jax.numpy as jnp
import jax
from jax import lax
__all__ = [
    "q",
    "form_Σ",
    "form_Σ_submatrix_λ",
    "form_λ_log_N",
    "E_Z",
    "ν̂_one_samp", 
    "g_chgpt",
    "g_j_chgpt_full",
    "g_chgpt_full",
    "run_chgpt_GAMP_jax"
]

# Use jax.scipy later, when/if want to do automatic differentiation.
from jax.scipy.stats import multivariate_normal
from functools import partial
# from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import spsolve
from functools import partial
from jax import jit

class q():
    def __init__(self, Θ, C_s, σ, noise_vector = None):
        """The heterogeneous output function.
        Inputs:
        C_s is length-n vector, each entry indicates which signal influences data point i. 
        Θ=XB is of shape nxL.
        We have to implement this in Jax because the code will differentiate through it. 
        """
        self.key = jax.random.PRNGKey(random.randint(0, 100000))
        self.Θ = Θ
        self.n = self.Θ.shape[0]
        self.C_s = C_s
        self.σ = σ
        self.noise_vector = noise_vector

    def sample(self):
        if self.noise_vector is not None: 
            η = self.noise_vector.reshape((self.n, 1))
        else:
            η = nprandom.normal(loc = 0, scale = self.σ, size=(self.n, 1))
        self.η = jnp.array(η)

        A = self.Θ + self.η
        # Pick the correct entry from each row of A, according to C_s
        Y = jnp.apply_along_axis(lambda x: A[x[0], x[1]], 1, \
            jnp.block([jnp.arange(self.n).reshape((self.n, 1)), \
                        self.C_s.reshape((self.n, 1))]))
        Y = Y.reshape((self.n, 1))
        return Y

def form_Σ(C, j, L, σ, ρ, ν, κ_T):
    """
    Construct the (2L+n)-by-(2L+n) covariance matrix of
    [(Z_{B, j}, V_{Θ,j}^t)^T, bar{Y}] in R^{2L+n} conditioned on 
    C, the entire changepoints vector, (n, 1). 

    TESTED. 
    """
    assert C.dtype == 'int32' or C.dtype == 'int64' or C.dtype == int
    if C.ndim == 1:  # reshape to nx1 if not already
        C = C[:, jnp.newaxis]

    n = C.shape[0]

    # Form Cov(Zᵀ_{B, j}, Vᵀ_{Θ, j})
    Σ_Z_V = jnp.block(
        [
            [ρ, ν],
            [ν.T, ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T]
        ]
    )

    # Form Cov(Ȳ | C). O(n) complexity.
    Σ_Y = amp.covariances.Σ_Y(C, n, ρ, σ) # This gives errors for now

    # For Cov((Z_{B, j}, V_{Θ, j})ᵀ,  Ȳ | C)
    Σ_Z_V_Y = amp.covariances.Cov_ZV_Y(j, C, n, ρ, ν)

    # Form the full Σ_j
    Σ_j = jnp.block(
        [
            [Σ_Z_V, Σ_Z_V_Y],
            [Σ_Z_V_Y.T, Σ_Y]
        ]
    )

    return Σ_j

def form_Σ_submatrix_λ(C, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_):
    """
    Calculate λ_idx = E[Z_{B,j} | V_{Θ,j}^t=V, \bar{Y}=u, c_i = idx] 
    for idx j in [0,L-1].

    Input V is 1-by-L, u is n-by-1.
    Output λ is L-by-1.

    TESTED. And it seems to be pretty fast. 
    """
    assert V.shape == (1, L) and (u.shape[0] == C.shape[0])
    n = u.shape[0]
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    # Σⱼ = form_Σ(C, j, L, σ, ρ, ν, κ_T)
    # term1 = Σⱼ[:2, 2:]
    # Σ_j[2:, 2:] is not exactly block diagonal, so don't know how to simplify the following pinv unless we use sparse matrix numpy.
    # Σ_sparse = sparse.BCOO.fromdense(Σⱼ[2:, 2:])
    # Σ_sparse_pinv = spsolve(Σⱼ[2:, 2:], jnp.eye(Σⱼ[2:, 2:].shape[0]))
    # term2 = Σ_sparse_pinv @ jnp.concatenate((V.T, u))
    # term2 = jnp.matmul(Σ_sparse_pinv, jnp.concatenate((V.T, u)))
    # Σ_sparse_pinv, info = jax.scipy.sparse.linalg.cg(Σⱼ[2:, 2:], jnp.eye(Σⱼ[2:, 2:].shape[0]))
    # term2 = Σ_sparse_pinv[:, 0] * V[0, 0] + Σ_sparse_pinv[:, 1] * V[0, 1] + jnp.sum(Σ_sparse_pinv[:, 2:] @ jnp.diag(u.flatten()), axis=1)
    # print("Term2 shape: ", term2.shape)
    # term2 = spsolve(Σ_sparse, jnp.concatenate((V.T, u))) # Exploits sparsity of the matrix Σ_j, hopefully this complexity is less than O(n^3)
    # Use the jax experimental sparse module if this is still too slow.
    # eps = 1e-1
    # Σ_V_Y_inv = amp.covariances.Σ_V_Y_inv(j, C, n, ρ, σ, ν, κ_T)
    # term2 = Σ_V_Y_inv @ jnp.concatenate((V.T, u))
    # res_reg = Σ_V_Y_inv + eps * jnp.eye(L + n)
    # term2 = res_reg @ jnp.concatenate((V.T, u))
    # print("jnp.sum(term2_mine- term2): ", jnp.sum(term2_mine - term2))
    # print("term2.shape = ", term2.shape)
    # print("term2: ", term2)
    # Only way I can get autodiff to work: 

    # L = jax.scipy.linalg.cholesky(Σⱼ[2:, 2:])
    # term2 = jax.scipy.linalg.solve_triangular(L, jnp.concatenate((V.T, u)), lower=True) # Doesnt even work with cholesky!
    
    # L_chol = lax.linalg.cholesky(Σⱼ[2:, 2:])
    # term2 = jnp.vectorize(
    #     partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
    #     signature="(n,n),(n)->(n)"
    # )(L_chol, jnp.concatenate((V.T, u)).flatten()).flatten() # Doesnt even work with cholesky!


    # term2, info = jax.scipy.sparse.linalg.cg(Σⱼ[2:, 2:], jnp.concatenate((V.T, u)))  # Apparently good for sparse matrices
    
    # Σ_inv_x, log_det_ = amp.covariances.Σ_V_Y_inv_x(j, V, u, C, n, ρ, σ, ν, κ_T, ν_κ_inv, A_inv_x_)
    A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_ = amp.covariances.Σ_V_Y_inv_x(j, V, u, C, n, ρ, σ, ν, κ_T, ν_κ_inv, A_inv_x_)

    # term2 = jnp.linalg.solve(Σⱼ[2:, 2:], jnp.concatenate((V.T, u)))  # RETURNS ERRORS ON THE GRADIENT!
    # term2 = sparse.sparsify(jax.scipy.sparse.linalg.cg)(Σⱼ[2:, 2:], jnp.concatenate((V.T, u))) # Apparently good for sparse matrices
    # term2 = jnp.linalg.pinv(Σⱼ[2:, 2:]) @ jnp.concatenate((V.T, u)) # O(n^3) complexity. Could be simplified and divided into blocks, so that we have to invert only once. Also, the matrix is very sparse, so can use sparse matrix inversion in numpy.
    # Shouldn't this be transposed instead?
    # λ = jnp.matmul(term1, Σ_inv_x).reshape(2, 1)
    Σ_inv_x_L = A_inv_x_[:L] - A_inv_x_update_L
    Σ_inv_x_L_plus_j = A_inv_x_[L + j] - A_inv_x_update_L_plus_j
    λ = (ν @ Σ_inv_x_L).reshape((L, 1)) + (ρ[:, C[j][0]] * Σ_inv_x_L_plus_j).reshape((L, 1)) # Compute λ without doing a size n matrix multiplication!

    # λ = (ν @ Σ_inv_x[0:L]).reshape((2, 1)) + (ρ[:, C[j][0]] * Σ_inv_x[L + j]).reshape((2, 1)) # Compute λ without doing a size n matrix multiplication!

    # return Σⱼ[2:, 2:], λ, Σ_inv_x, log_det_
    # return λ, Σ_inv_x, log_det_
    return λ, A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_

def form_λ_log_N(C, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_, x_A_inv_x_):
    """ Created so that we can maximally vectorize the sum_log_exp operations. Tested. """

    x = jnp.concatenate((V.T, u))
    # λ, Σ_inv_x, log_det_ = form_Σ_submatrix_λ(C, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_)
    # return jnp.append(λ, amp.covariances.log_pdf_cg(x, log_det_, Σ_inv_x))

    λ, A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_ = form_Σ_submatrix_λ(C, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_)
    return jnp.append(λ, amp.covariances.log_pdf_cg_opt(x, log_det_, x_A_inv_x_, A_inv_x_update_L, A_inv_x_update_L_plus_j, L, j))

# @partial(jit, static_argnums=1)
def E_Z(V, j, u, L, σ, ρ, ν, κ_T, C_s):
    """ 
    Calculate E[Z_{B,i} | V_{Theta,i}^t=V, bar{Y}=u].
    V is 1-by-L, u is n-by-1. 
    Output is 1-by-L, and can be positive or negative, will depend on the index j. 
    Tested and vectorized. 
    """
    assert V.shape == (1, L) and not jnp.isscalar(u)
    n = u.shape[0]
    # assert jnp.isscalar(σ) and ρ[0, 0] == ρ[1, 1]
    if u.ndim == 1:  # reshape to nx1 if not already
        u = u[:, jnp.newaxis]

    # ASSUMPTION USED HERE: We CAN have the all ones vector as C. But not the all zeros vector.
    # C_s = jnp.triu(jnp.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1.
    # The following vmap causes the final result to differ depending on whether in_axis = 0 or =1, but it should be indifferent to this.
    # ITERATE OVER THE ROWS OF C_s
    # Iterate across rows of c: start with changepoint at index 0, then end with chgpt at index n-2
    
    # Pass in the inverse L×L covariance
    ν_κ_inv = jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T) # The pinv here instead of inv is important for automatic differentiation. 
    x = jnp.concatenate((V.T, u))
    A_inv_x_ = amp.covariances.A_inv_x_no_C(x, n, ρ, σ, ν, κ_T)
    x_A_inv_x_ = jnp.dot(x.flatten(), A_inv_x_.flatten())

    λ_log_N_s = jax.vmap(form_λ_log_N, (0, None, None, None, None,
                        None, None, None, None, None, None, None), 0)(C_s, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_, x_A_inv_x_)

    # Using map instead of vmap is slightly slower, but consumes less memory. 
    # def form_λ_log_N_wrapped(C_s):
    #     return form_λ_log_N(C_s, j, V, u, L, σ, ρ, ν, κ_T, ν_κ_inv, A_inv_x_, x_A_inv_x_)
    # λ_log_N_s = jax.lax.map(form_λ_log_N_wrapped, C_s)

    assert λ_log_N_s.shape == (C_s.shape[0], L + 1)
    λ_s = λ_log_N_s[:, :L]
    log_N_s = λ_log_N_s[:, L:]
    assert log_N_s.shape[0] == C_s.shape[0]
    log_N_s = jnp.float32(log_N_s)

    log_num_arr, sign_arr = logsumexp(
        a=log_N_s, axis=0, b=λ_s, return_sign=True)
    log_denom = logsumexp(a=log_N_s, axis=0, return_sign=False)
    res = jnp.multiply(sign_arr, jnp.exp(log_num_arr - log_denom))
    return res.reshape((1, L))  # This way the output is 1-by-L.

def g_chgpt(V, j, u, L, σ, ρ, ν, κ_T, C_s):
    """ 
    Calculate gt applied to one row: R^L * R -> R^L.
    V is 1-by-L, u is n-by-1. Will depend on the index, j. 
    Output is 1-by-L.
    """

    if V.ndim == 1:  # reshape to 1xL if not already
        V = V[:, jnp.newaxis].T
    assert hasattr(u, "__len__"), 'V is not 1xL or u is a scalar'
    ξ = jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T)
    return (E_Z(V, j, u, L, σ, ρ, ν, κ_T, C_s) - V @ ξ.T @ ν.T) @ jnp.linalg.pinv(ρ - ν.T @ ξ.T @ ν)

def g_chgpt_full():
    """ Returns the vectorized version of g_chpt, to be always applied to a fixed Y etc. """
    return jax.vmap(g_chgpt, (0, 0, None, None, None, None, None, None, None), 0)
g_chgpt_full_ = g_chgpt_full()
g_chgpt_tensorized = jax.jit(g_chgpt_full(), static_argnames=('L'))

def g_j_chgpt_full():
    """ Keeps the j fixed. This is used for the approximation of C^t. """
    return jax.vmap(g_chgpt, (0, None, None, None, None, None, None, None), 0)
g_j_chgpt_tensorized = jax.jit(g_j_chgpt_full(), static_argnames=('L'))

def ν̂_double_samp(C_s, Θ_t, indx, L, ρ, n, σ, ν, κ_T, Y = None):
    
    indx = jnp.arange(0, Θ_t.shape[0]).reshape((Θ_t.shape[0], 1))
    def wrapper(a):
        return ν̂_samp(C_s, Θ_t, indx, L, ρ, n, σ, ν, κ_T, Y = None)

    return 1/n * jnp.sum(jax.vmap(wrapper, 0, 0)(indx), axis=0)

def ν̂_samp(C_s, Θ_t, indx, L, ρ, n, σ, ν, κ_T, Y = None):
    """ Approximate ν̂ by sampling. Tested with separable and with Y. Should replace nested vmaps with map otherwise too memory intensive."""

    def ν̂_samp_(C):
        Z_B_s = nprandom.multivariate_normal(jnp.zeros(L), ρ, size=(n, n)) # n × n × 2, increasing number of samples here does not seem to help. 
        κ_T_symm = (κ_T + κ_T.T) / 2
        G_Θ = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T_symm, size=(n, n)))
        V_Θ_s = Z_B_s @ jnp.linalg.inv(ρ) @ ν + G_Θ 
        def g_samp(V_Θ, Z_B):
            if Y is not None:
                res = g_chgpt_tensorized(V_Θ, indx, Y, L, σ, ρ, ν, κ_T, C_s, None).reshape((n, L)) 
            else:
                res = g_chgpt_tensorized(V_Θ, indx, q(Z_B, C, σ).sample(), L, σ, ρ, ν, κ_T, C_s, None).reshape((n, L))
            return 1/n * res.T @ res
        g_samp_tensorized = jax.vmap(g_samp, (0, 0), 0)
        return g_samp_tensorized(V_Θ_s, Z_B_s)
    E_Z_B = lambda C: 1/n * jnp.sum(ν̂_samp_(C), axis=0)
    E_Z_B_tensorized = jax.vmap(E_Z_B, 0, 0)
    assert C_s.shape[1] == Θ_t.shape[0]
    ν̂ = 1/(C_s.shape[0]) * jnp.sum(E_Z_B_tensorized(C_s), axis=0)
    return ν̂

def ν̂_one_samp(C_s, indx, L, ρ, n, σ, ν, κ_T):
    """ Compute ν̂ using one sample. Tested. """

    Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=n))
    κ_T_symm = (κ_T + κ_T.T) / 2 # To avoid numerical issues.
    V_Θ = Z_B @ jnp.linalg.inv(ρ) @ ν + jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T_symm, size=n))
    def inner(C):
        g_ = g_chgpt_tensorized(V_Θ, indx, q(Z_B, C, σ).sample(), L, σ, ρ, ν, κ_T, C_s).reshape((n, L)) # Double check that this reshaping is okay. 
        return 1/n * g_.T @ g_
    inner_tensorized = jax.vmap(inner, 0, 0)
    assert C_s.shape[1] == indx.shape[0]
    ν̂ = 1/(C_s.shape[0]) * jnp.sum(inner_tensorized(C_s), axis=0)
    return ν̂

def run_chgpt_GAMP_jax(C_s, B̂_0, δ, p, L, σ, X, Y, ρ, T, verbose=False, seed=None, post=False):
    if seed is not None:
        nprandom.seed(seed)
    n = int(δ * p)

    B̂ = B̂_0
    ν = jnp.zeros((L, L)) 
    κ_T = ρ  
    ν̂ = jnp.zeros((L, L)) # Not necessary
    F = jnp.eye(L)  # Checked
    R̂ = jnp.zeros((n, L))  # Checked
    Θ_t = jnp.zeros((n, L))

    # Tensorize and jit compile desired functions.
    # dgdθ = jax.jacfwd(g_chgpt, 0)
    dgdθ = jax.jacrev(g_chgpt, 0)

    def jacobian(V, j, u, ν, κ_T): return dgdθ(
        V, j, u, L, σ, ρ, ν, κ_T, C_s).reshape((L, L))

    def C_fun(Θ_, Y_, ν, κ_T):
        indx = jnp.arange(0, Θ_.shape[0]).reshape((Θ_.shape[0], 1))

        # If want to use vmap: 
        jac_map = jax.vmap(jacobian, (0, 0, None, None, None), 0) # jacobian stays the same except the shape of u changes, so maybe could jit compile this and label u as the only arg that changes.
        jac_mapped = jac_map(Θ_, indx, Y_, ν, κ_T)

        # If want to use map instead of vmap as it uses less memory: 
        # def jacobian_wrapped(V_j): 
        #     if V_j.shape != (1, L + 1): # Does this make the code slow for JIT compilation?  
        #         V_j = V_j.reshape((1, L + 1))
        #     return jacobian(V_j[0, :L].flatten(), V_j[0, -1].astype(int).flatten(), Y_, ν, κ_T)
        # jac_mapped = jax.lax.map(jacobian_wrapped, jnp.hstack((Θ_, indx)))

        return 1/n * jnp.sum(jac_mapped, axis=0)

    C_tensorized = jax.jit(C_fun)

    for t in tqdm(range(T)):
        if verbose:
            print("ν: ", ν)
            print("κ_T: ", κ_T)

          ## -- AMP -- ##
        Θ_t = X @ B̂ - R̂ @ F.T

        ## -- g and its parameters -- ##
        indx = jnp.arange(0, Θ_t.shape[0]).reshape((Θ_t.shape[0], 1))
        R̂ = g_chgpt_tensorized(Θ_t, indx, Y, L, σ, ρ,
                                ν, κ_T, C_s).reshape((n, L))

        # If want to use map instead of vmap, with JIT:
        # def g_chgpt_map_wrapper(V_j):
        #     if V_j.shape != (1, L + 1): # Does this make the code slow for JIT compilation?  
        #         V_j = V_j.reshape((1, L + 1))
        #     return g_chgpt_jitted(V_j[0, :L].flatten(), V_j[0, -1].astype(int).flatten(), Y, L, σ, ρ, ν, κ_T, C_s)

        # If want to use map but without JIT: 
        # def g_chgpt_map_wrapper(V_j):
        #     if V_j.shape != (1, L + 1): # Does this make the code slow for JIT compilation?  
        #         V_j = V_j.reshape((1, L + 1))
        #     return g_chgpt(V_j[0, :L].flatten(), V_j[0, -1].astype(int).flatten(), Y, L, σ, ρ, ν, κ_T, C_s)
        # R̂ = jax.lax.map(g_chgpt_map_wrapper, jnp.hstack((Θ_t, indx))).reshape((n, L))

        # if (jnp.isnan(R̂).any() or jnp.isinf(R̂).any()):
        #     print('=== EARLY STOPPAGE R̂===')
        #     print("R̂: ", R̂)
        #     break
        
        ### --- Same C as fully sep case (for testing) --- ###
        # Ω = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
        # # C = (jnp.linalg.pinv(Ω) @ (1/n * Θ_t.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T
        # C_fully_sep = (jnp.linalg.pinv(Ω) @ (1/n * Θ_t.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T

        ### --- C^t with AD (seems to be the best one for now, fastest and doesn't blow up) --- ###
        C = C_tensorized(Θ_t, Y, ν, κ_T)
        # Ω = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
        # C = (jnp.linalg.pinv(Ω) @ (1/n * Θ_t.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T # Replacing this with autodiff might make it more numerically stable. 
        # C = C_fun(Θ_t, Y, ν, κ_T) # if don't want to JIT 
        if verbose: 
        #    print("C - C_fully_sep: ", C - C_fully_sep) # difference is not too large, on the order of 1e-1, and considering that C_fully_sep is an approximation this kind of makes sense. 
           print("C: ", C)
        if (jnp.isnan(C).any() or jnp.isinf(C).any()):
            print('=== EARLY STOPPAGE C ===')
            print("C: ", C)
            print("Θ_t: ", Θ_t)
            print("Y: ", Y)
            break
            

        ### --- C^t short approximation (from experiments this is most definitely blowing up, blows up at δ = 0.5 in the all ones edge case) --- ###
        # Ω = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
        # def inner(i): return 1/n * Θ_t.T @ g_j_chgpt_tensorized(Θ_t,
        #                                                         i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
        # inner_map = jax.vmap(inner, 0, 0)
        # indx = jnp.arange(0, Θ_t.shape[0]).reshape((Θ_t.shape[0], ))
        # # Should double check whether the theory is correct.
        # C = 1/n * jnp.sum(inner_map(indx), axis=0).T @ jnp.linalg.pinv(Ω).T # I don't think this is correct. 
        # if verbose: print("C: ", C)

        ### --- C^t long approximation (from experiments this is most definitely blows up for all δ in all ones edge case) --- ###
        # Ω = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
        # def inner1(i): return 1/n * Θ_t.T @ g_j_chgpt_tensorized(Θ_t,
        #                                                 i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
        # def inner2(i): 
        #     g_res = g_j_chgpt_tensorized(Θ_t, i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
        #     return 1/n * g_res.T @ g_res
        # inner1_map = jax.vmap(inner1, 0, 0)
        # inner2_map = jax.vmap(inner2, 0, 0)
        # indx = jnp.arange(0, Θ_t.shape[0]).reshape((Θ_t.shape[0], ))
        # C =  (jnp.linalg.pinv(Ω) @ (1/n * jnp.sum(inner1_map(indx), axis=0) - ν.T @ (1/n * jnp.sum(inner2_map(indx), axis=0)) )).T # I don't think this is correct. 
        # if verbose: print("C: ", C)

        B_t = X.T @ R̂ - B̂ @ C.T

        ν̂ = ν̂_one_samp(C_s, indx, L, ρ, n, σ, ν, κ_T) # This is not the problem, signal blows up even with ν̂_samp
        # ν̂ = ν̂_samp(C_s, Θ_t, indx, L, ρ, n, σ, ν, κ_T)
        # ν̂ = 1/n * R̂.T @ R̂

        # Z_B = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), ρ, size=n))
        # G_Θ = jnp.array(nprandom.multivariate_normal(jnp.zeros(L), κ_T, size=n))
        # V_Θ = Z_B @ jnp.linalg.inv(ρ) @ ν + G_Θ
        # R̂_2 = g_chgpt_tensorized(V_Θ, indx, q(Z_B, C_s[0].reshape((1, n)), σ).sample(), L, σ, ρ, ν, κ_T, C_s).reshape((n, L)) # Double check that C_s[0] is the right way to pass in
        # ν̂_sep = 1/n * R̂_2.T @ R̂_2

        if verbose:
            print("ν̂: ", ν̂)
            # print("ν̂_sep: ", ν̂_sep)
        κ_B = ν̂

        ## -- f and its parameters. The same as the fully separable case. -- ##
        B̂ = f_full(B_t, δ, L, ν̂, κ_B, ρ)
        # print("ν̂_κ: ", ν̂.T @ ρ @ ν̂ + 1/δ * κ_B)
        # print("ν̂_κ pinv: ", jnp.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B))
        χ = (ρ @ ν̂) @ jnp.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) # type: ignore
        F = 1/δ * χ.T
        # F = (p/n) * χ.T # Slightly more accurate

        # Closed form gaussian case (could help in troubleshooting as it makes the code more stable)
        χ = (ρ @ ν̂) @ jnp.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) # type: ignore
        ν = ρ @ ν̂ @ χ.T
        κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν
        # ν_true = ν
        # print("ν_true: ", ν_true)

        # ν = 1/δ * 1/p * B̂.T @ B̂ # Try this with the true B, and see. 
        # print("ν_emp: ", ν)
        # print("ν_emp - ν_true: ", ν - ν_true)
        # # ν = 1/n * B̂.T @ B̂ 
        # κ_T = ν - ν.T @ jnp.linalg.inv(ρ) @ ν  # Checked

    if not post:
        return B̂, ν, ν̂
    else:
        return B̂, ν, ν̂, Θ_t