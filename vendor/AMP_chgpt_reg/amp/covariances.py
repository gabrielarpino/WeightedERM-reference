import numpy.random as nprandom
import random
from amp.fully_separable import f_full
from tqdm import tqdm
from . import ϵ_0
from jax.scipy.special import logsumexp
from jax.experimental import sparse
import jax.numpy as jnp
import jax
from jax.scipy.sparse.linalg import cg as cg
import jax.scipy.sparse.linalg
# import numpy as np
# import scipy

__all__ = [
    "woodbury",
    "woodbury_C_diag",
    "woodbury_optimized_L_2",
    "sherman_morrison",
    "Σ_V_full", 
    "Σ_Y",
    "Cov_Z_Y",
    "Cov_V_Y", 
    "Cov_V_Y_full", 
    "Σ_V_Y_full",
    "Σ_V_Y_inv_x"
]

def sherman_morrison(A_inv, u, v):
    """ Return the inverse of A + u @ v^T. Could potentially be more optimized. """
    return A_inv - (A_inv @ u @ v.T @ A_inv) / (1 + v.T @ A_inv @ u)

def woodbury(A_inv, U, C, V):
    """ U is n × k, C is k × k, V is k × n."""
    # assert U.shape[0] == A_inv.shape[0]
    # assert V.shape[1] == A_inv.shape[0]
    return A_inv - A_inv @ U @ jnp.linalg.inv(jnp.linalg.inv(C) + V @ A_inv @ U) @ V @ A_inv

def woodbury_C_diag(A_inv, U, C_diag, V):
    """ U is n × k, C_diag is a scalar, V is k × n. Slightly faster. """
    # assert U.shape[0] == A_inv.shape[0]
    # assert V.shape[1] == A_inv.shape[0]
    return A_inv - A_inv @ U @ jnp.linalg.pinv(1/C_diag * jnp.eye(V.shape[0]) + V @ A_inv @ U) @ V @ A_inv

def woodbury_optimized_L_2(j, A_inv, U, C_diag, V):
    """ U is n × k, C_diag is a scalar, V is k × n. This is highly optimized, returns the full inverse in O(n) time, works only for linear regression and L = 2. """
    assert U.shape[0] == A_inv.shape[0]
    assert V.shape[1] == A_inv.shape[0]
    if not jnp.isscalar(j): j = j[0]
    k = 2
    L = k
    n = A_inv.shape[0] - L

    A_inv_U = jnp.vstack((A_inv[:, 0] * U[0, 0] + A_inv[:, 1] * U[1, 0] + A_inv[:, L + j] * U[L + j, 0], A_inv[:, 0] * U[0, 1] + A_inv[:, 1] * U[1, 1] + A_inv[:, L + j] * U[L + j, 1])).T
    V_A_inv = jnp.vstack((A_inv[0, :] * V[0, 0] + A_inv[1, :] * V[0, 1] + A_inv[L + j, :] * V[0, L + j],A_inv[0, :] * V[1, 0] + A_inv[1, :] * V[1, 1] + A_inv[L + j, :] * V[1, L + j]))

    V_A_inv_U = jnp.vstack((A_inv_U[0, :] * V[0, 0] + A_inv_U[1, :] * V[0, 1] + A_inv_U[L + j, :] * V[0, L + j], A_inv_U[0, :] * V[1, 0] + A_inv_U[1, :] * V[1, 1] + A_inv_U[L + j, :] * V[1, L + j]))

    ζ = jnp.linalg.pinv(1/C_diag * jnp.eye(V.shape[0]) + V_A_inv_U) # This is k × k

    ζ̃ = jnp.zeros((n, k))
    ζ̃ = ζ̃.at[0:k, 0:k].set(A_inv_U[0:k, 0:k] @ ζ)
    ζ̃ = ζ̃.at[L + j, 0].set(jnp.dot(A_inv_U[L + j, :], ζ[:, 0]))
    ζ̃ = ζ̃.at[L + j, 1].set(jnp.dot(A_inv_U[L + j, :], ζ[:, 1]))

    sres = jnp.zeros((L + n, L + n))
    sres = sres.at[0:k, 0:k].set(ζ̃[0:k, 0:k] @ V_A_inv[0:k, 0:k])
    sres = sres.at[0:k, L + j].set(ζ̃[0:k, 0:k] @ V_A_inv[:, L + j])
    sres = sres.at[L + j, 0:k].set(ζ̃[L + j, :] @ V_A_inv[0:k, 0:k])
    sres = sres.at[L + j, L + j].set(ζ̃[L + j, :] @ V_A_inv[:, L + j])

    return A_inv - sres

def Σ_V_full(n, ρ, ν, κ_T):
    """ Return a nL × nL block diagonal matrix with ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T on the diagonals. """
    τ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
    return jax.scipy.linalg.block_diag(*[τ for i in range(0, n)])

def Σ_Z_full(n, ρ):
    """ Return a nL × nL block diagonal matrix with ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T on the diagonals. """
    τ = ρ
    return jax.scipy.linalg.block_diag(*[τ for i in range(0, n)])

def Σ_Y(C, n, ρ, σ):
    """ Return an n × n matrix with ρ[c_i, c_i] + σ**2 in its i-th diagonal. """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    def f(c): return ρ[c, c] + σ**2
    ρ_σ_diag = jax.vmap(f, 0, 0)(C)
    # ρ_σ_diag = jnp.apply_along_axis(f, 1, C).reshape((n,)) # See which of these two is faster
    return jax.scipy.linalg.block_diag(*ρ_σ_diag)

def Σ_Y_inv_diag(C, n, ρ, σ):
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    def f(c): return 1/(ρ[c, c] + σ**2)
    ρ_σ_diag = jax.vmap(f, 0, 0)(C)
    return ρ_σ_diag

def Σ_Y_inv(C, n, ρ, σ):
    ρ_σ_diag = Σ_Y_inv_diag(C, n, ρ, σ)
    res = jax.scipy.linalg.block_diag(*ρ_σ_diag)
    assert res.shape == (n, n)
    return res

def Σ_Y_inv_diag_no_C(n, ρ, σ):
    return 1/(ρ[0, 0] + σ**2) * jnp.ones((n,))

def Cov_Z_Y(j, C, n, ρ):
    """ Return an L × n matrix of covariances between Z_B and Ȳ = q(Z_B, Ψ). """
    L = ρ.shape[0]
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    Cov_Z_Y_ = jnp.zeros((L, n))
    Cov_Z_Y_ = Cov_Z_Y_.at[:, j].set(ρ[:, C[j][0]])
    return Cov_Z_Y_

def Cov_V_Y(j, C, n, ρ, ν):
    """ Return an L × n matrix of covariances between V_Θ[j] and Ȳ. """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    Cov_Z_Y_ = Cov_Z_Y(j, C, n, ρ)
    return ν.T @ jnp.linalg.inv(ρ) @ Cov_Z_Y_

def Σ_V_j_Y_j(C_j, ρ, σ, ν, κ_T):
    """Covariance of V_j ∈ R^L, Y_j ∈ R given C_j"""
    L = ρ.shape[0]
    ν_ρ_inv = ν.T @ jnp.linalg.inv(ρ)
    A = jnp.block([
        [ν_ρ_inv @ ν + κ_T, (ν_ρ_inv @ ρ[:, C_j]).reshape((L, 1)) ],
        [(ν_ρ_inv @ ρ[:, C_j]).reshape((1, L)), ρ[C_j, C_j] + σ**2]
    ])
    return A # (L+1)x(L+1) matrix

def A_inv_x(x, C, n, ρ, σ, ν, κ_T):
    # x = jnp.concatenate((V.T, u))
    L = ν.shape[0]
    ν_κ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
    # A_inv_V = jnp.linalg.pinv(ν_κ) @ x[0:L] # This is already unstable
    A_inv_V, info = cg(ν_κ, x[0:L]) # This is the key step which makes the gradient stable!
    A_inv_u = jnp.multiply((Σ_Y_inv_diag(C, n, ρ, σ)).flatten(), x[L:].flatten()).reshape((n, 1))
    A_inv_x = jnp.concatenate((A_inv_V, A_inv_u))
    return A_inv_x
    
def A_inv_x_no_C(x, n, ρ, σ, ν, κ_T):
    """ Useful for when we can assume ρ[0, 0] == ρ[1, 1] and σ is a scalar. """
    # assert ρ[0, 0] == ρ[1, 1]
    L = ν.shape[0]
    ν_κ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T
    # A_inv_V = jnp.linalg.pinv(ν_κ) @ x[0:L] # This is already unstable
    A_inv_V, info = cg(ν_κ, x[0:L]) # This is the key step which makes the gradient stable!
    A_inv_u = ((1/(ρ[0, 0] + σ**2)) * x[L:]).reshape((n, 1))
    A_inv_x = jnp.concatenate((A_inv_V, A_inv_u))
    return A_inv_x

def woodbury_x(A_inv_x_, U, V, C_diag, C, n, ρ, σ, ν, κ_T):
    """ Compute a specialized woodbury inverse multiplication using a conjugate gradient solution which is more numerically stable. 
    Can be further optimized, like in the woodbury_optimized function, or using cholesky decomposition.
    The sensitive numerical step is inverting this ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T, and pinv or cgrad for that seems to help.
    """
    L = U.shape[1]
    A_inv = jnp.block([
        [jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T), jnp.zeros((L, n))],
        [jnp.zeros((L, n)).T, Σ_Y_inv(C, n, ρ, σ)]
    ])
    x = A_inv @ U @ jnp.linalg.pinv(1/C_diag * jnp.eye(L) + V @ A_inv @ U) @ V # The inv vs pinv in this line doesnt seem to be a numerical problem
    return A_inv_x_ - x @ A_inv_x_

def woodbury_x_optimized(j, A_inv_x_, U_L, U_L_j, V_L, V_L_j, C_diag, ρ, σ, ν_κ_inv):
    """ Optimized woodbury computation that does not compute large matrix multiplications when we know j. This update runs in O(1), notice there are no matrix multiplications of size n. """
    L = U_L.shape[0]

    # ν_κ_inv = jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T)
    # Σ_Y_inv_diag_ = Σ_Y_inv_diag(C, n, ρ, σ).flatten()
    Σ_Y_inv_diag_j = 1/(ρ[0, 0] + σ**2)


    # A_inv_U = jnp.zeros((L + n, L))
    # A_inv_U = A_inv_U.at[0:L, 0:L].set(ν_κ_inv @ U[0:L, 0:L])
    # A_inv_U = A_inv_U.at[L+j, :].set((Σ_Y_inv_diag_[j] * U[L+j, :]).flatten())

    # V_A_inv_U = V[0:L, 0:L] @ A_inv_U[0:L, 0:L] + jnp.outer(V[:, L+j], A_inv_U[L+j, :])

    # More memory efficient
    # V_A_inv_U = V_L @ ν_κ_inv @ U_L + jnp.outer(V_L_j, Σ_Y_inv_diag_[j] * U_L_j)
    
    # Even more memory efficient
    V_A_inv_U = V_L @ ν_κ_inv @ U_L + jnp.outer(V_L_j, Σ_Y_inv_diag_j * U_L_j)

    X2 = jnp.linalg.pinv(1/C_diag * jnp.eye(2) + V_A_inv_U)
    # X3 = V

    # ζ = jnp.zeros((L+n, L+n))
    # ζ = ζ.at[:L, :L].set(A_inv_U[:L, :L] @ X2 @ V[:L, :L])
    # ζ = ζ.at[:L, L+j].set(A_inv_U[:L, :L] @ X2 @ V[:, L+j])
    # ζ = ζ.at[L+j, :L].set(A_inv_U[L+j, :] @ X2 @ V[:L, :L])
    # ζ = ζ.at[L+j, L+j].set(jnp.dot(A_inv_U[L+j, :].flatten(), (X2 @ V[:, L+j]).flatten()))

    # # More memory efficient: 
    # ζ = jnp.zeros((L+n, L+n))
    # ζ = ζ.at[:L, :L].set(ν_κ_inv @ U[0:L, 0:L] @ X2 @ V[:L, :L])
    # ζ = ζ.at[:L, L+j].set(ν_κ_inv @ U[0:L, 0:L] @ X2 @ V[:, L+j])
    # ζ = ζ.at[L+j, :L].set(Σ_Y_inv_diag_[j] * U[L+j, :] @ X2 @ V[:L, :L])
    # ζ = ζ.at[L+j, L+j].set(jnp.dot((Σ_Y_inv_diag_[j] * U[L+j, :]).flatten(), (X2 @ V[:, L+j]).flatten()))

    # ζ_A_inv_x = jnp.zeros((L+n, 1))
    # ζ_A_inv_x = ζ_A_inv_x.at[:L].set(ζ[:L, :L] @ A_inv_x_[:L] + (ζ[:L, L+j] * A_inv_x_[L+j]).reshape((L, 1)))
    # ζ_A_inv_x = ζ_A_inv_x.at[L+j].set(ζ[L+j, :L] @ A_inv_x_[:L] + ζ[L+j, L+j] * A_inv_x_[L+j])

    # More memory efficient
    # ζ_A_inv_x = jnp.zeros((L+n, 1))
    # ζ_A_inv_x = ζ_A_inv_x.at[:L].set(ν_κ_inv @ U[0:L, 0:L] @ X2 @ V[:L, :L] @ A_inv_x_[:L] + (ν_κ_inv @ U[0:L, 0:L] @ X2 @ V[:, L+j] * A_inv_x_[L+j]).reshape((L, 1)))
    # ζ_A_inv_x = ζ_A_inv_x.at[L+j].set(Σ_Y_inv_diag_[j] * U[L+j, :] @ X2 @ V[:L, :L] @ A_inv_x_[:L] + jnp.dot((Σ_Y_inv_diag_[j] * U[L+j, :]).flatten(), (X2 @ V[:, L+j]).flatten()) * A_inv_x_[L+j])

    # A_inv_x_updated = A_inv_x_
    # A_inv_x_updated = A_inv_x_updated.at[:L].set(A_inv_x_updated[:L] - ζ_A_inv_x[:L])
    # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - ζ_A_inv_x[L + j])

    # More memory efficient
    # A_inv_x_updated = A_inv_x_
    # A_inv_x_updated = A_inv_x_updated.at[:L].set(A_inv_x_updated[:L] - (ν_κ_inv @ U_L @ X2 @ V_L @ A_inv_x_[:L] + (ν_κ_inv @ U_L @ X2 @ V[:, L+j] * A_inv_x_[L+j]).reshape((L, 1))))
    # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - (Σ_Y_inv_diag_[j] * U[L+j, :] @ X2 @ V[:L, :L] @ A_inv_x_[:L] + jnp.dot((Σ_Y_inv_diag_[j] * U[L+j, :]).flatten(), (X2 @ V[:, L+j]).flatten()) * A_inv_x_[L+j]))

    # Even more memory efficient
    # A_inv_x_updated = A_inv_x_
    # A_inv_x_updated = A_inv_x_updated.at[:L].set(A_inv_x_updated[:L] - (ν_κ_inv @ U_L @ X2 @ V_L @ A_inv_x_[:L] + (ν_κ_inv @ U_L @ X2 @ V_L_j * A_inv_x_[L+j]).reshape((L, 1))))
    # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - (Σ_Y_inv_diag_[j] * U_L_j @ X2 @ V_L @ A_inv_x_[:L] + jnp.dot((Σ_Y_inv_diag_[j] * U_L_j).flatten(), (X2 @ V_L_j).flatten()) * A_inv_x_[L+j]))
    
    term1 = U_L_j @ X2
    term2 = V_L @ A_inv_x_[:L]
    # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - (Σ_Y_inv_diag_[j] * jnp.dot(term1.flatten(), term2.flatten()) + jnp.dot((Σ_Y_inv_diag_[j] * U_L_j).flatten(), (X2 @ V_L_j).flatten()) * A_inv_x_[L+j]))
    
    # More memory efficient
    # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - (Σ_Y_inv_diag_j * jnp.dot(term1.flatten(), term2.flatten()) + jnp.dot((Σ_Y_inv_diag_j * U_L_j).flatten(), (X2 @ V_L_j).flatten()) * A_inv_x_[L+j]))

    # return A_inv_x_updated

    A_inv_x_update_L = (ν_κ_inv @ U_L @ X2 @ V_L @ A_inv_x_[:L] + (ν_κ_inv @ U_L @ X2 @ V_L_j * A_inv_x_[L+j]).reshape((L, 1)))
    A_inv_x_update_L_plus_j = (Σ_Y_inv_diag_j * jnp.dot(term1.flatten(), term2.flatten()) + jnp.dot((Σ_Y_inv_diag_j * U_L_j).flatten(), (X2 @ V_L_j).flatten()) * A_inv_x_[L+j])

    return A_inv_x_update_L, A_inv_x_update_L_plus_j

def Σ_V_Y_inv_x_j(j, V_, C_j, n, ρ, σ, ν, κ_T, ν_κ_inv, A_inv_x_):
    """ Only uses C_j and not all of C """
    L = V_.shape[1]
    ρ̃_ = (ν.T @ jnp.linalg.inv(ρ) @ ρ[:, C_j]).flatten()

    # Memory efficient and holds for all L
    U_L = jnp.zeros((L, 2))
    U_L = U_L.at[0:L, 0].set(ρ̃_)
    U_L = U_L.at[0:L, 1].set(jnp.zeros((L, )))
    U_L_j = jnp.zeros((1, 2))
    U_L_j = U_L_j.at[0, 0].set(0)
    U_L_j = U_L_j.at[0, 1].set(1)

    # Memory efficient and holds for all L
    V_L = jnp.zeros((2, L))
    V_L = V_L.at[0, 0:L].set(jnp.zeros((L, )))
    V_L = V_L.at[1, 0:L].set(ρ̃_)
    V_L_j = jnp.zeros((2, 1))
    V_L_j = V_L_j.at[0, 0].set(1)
    V_L_j = V_L_j.at[1, 0].set(0)

    C_diag = 1

    woodbury_x_opt_ = woodbury_x_optimized(j, A_inv_x_, U_L, U_L_j, V_L, V_L_j, C_diag, ρ, σ, ν_κ_inv)
    return woodbury_x_opt_[0], woodbury_x_opt_[1], log_det_A_U_V_optimized(j, U_L, U_L_j, V_L, V_L_j, C_diag, n, ρ, σ, ν, κ_T, ν_κ_inv)

def Σ_V_Y_inv_x(j, V_, u, C, n, ρ, σ, ν, κ_T, ν_κ_inv, A_inv_x_):
    if C.shape != (n, 1): 
        C = C.reshape((n, 1))

    # A_inv_x_ = A_inv_x(x, C, n, ρ, σ, ν, κ_T)

    L = V_.shape[1]
    ρ̃_ = (ν.T @ jnp.linalg.inv(ρ) @ ρ[:, C[j][0]]).flatten()
    # assert ρ̃.shape == (L,), ρ̃.shape

    # Create U for woodbury
    # U = jnp.zeros((L + n, L)) # Maybe these allocations are too much. 
    # U = U.at[0:L, 0].set(ρ̃_)
    # U = U.at[0:L, 1].set(-ρ̃_)
    # U = U.at[L + j, 0].set(1/jnp.sqrt(2))
    # U = U.at[L + j, 1].set(1/jnp.sqrt(2))
    # assert U.shape == (L + n, L), U.shape

    # More memory efficient: 
    # U_L = jnp.zeros((L, L))
    # U_L = U_L.at[0:L, 0].set(ρ̃_)
    # U_L = U_L.at[0:L, 1].set(-ρ̃_)
    # U_L_j = jnp.zeros((1, L))
    # U_L_j = U_L_j.at[0, 0].set(1/jnp.sqrt(2))
    # U_L_j = U_L_j.at[0, 1].set(1/jnp.sqrt(2))

    # Memory efficient and holds for all L
    U_L = jnp.zeros((L, 2))
    U_L = U_L.at[0:L, 0].set(ρ̃_)
    U_L = U_L.at[0:L, 1].set(jnp.zeros((L, )))
    U_L_j = jnp.zeros((1, 2))
    U_L_j = U_L_j.at[0, 0].set(0)
    U_L_j = U_L_j.at[0, 1].set(1)

    # # Create V for woodbury
    # V = jnp.zeros((L, L + n))
    # V = V.at[0, 0:L].set(ρ̃_)
    # V = V.at[1, 0:L].set(ρ̃_)
    # V = V.at[0, L + j].set(1/jnp.sqrt(2))
    # V = V.at[1, L + j].set(-1/jnp.sqrt(2))
    # assert V.shape == (L, L + n), V.shape

    # More memory efficient:
    # V_L = jnp.zeros((L, L))
    # V_L = V_L.at[0, 0:L].set(ρ̃_)
    # V_L = V_L.at[1, 0:L].set(ρ̃_)
    # V_L_j = jnp.zeros((L, 1))
    # V_L_j = V_L_j.at[0, 0].set(1/jnp.sqrt(2))
    # V_L_j = V_L_j.at[1, 0].set(-1/jnp.sqrt(2))

    # Memory efficient and holds for all L
    V_L = jnp.zeros((2, L))
    V_L = V_L.at[0, 0:L].set(jnp.zeros((L, )))
    V_L = V_L.at[1, 0:L].set(ρ̃_)
    V_L_j = jnp.zeros((2, 1))
    V_L_j = V_L_j.at[0, 0].set(1)
    V_L_j = V_L_j.at[1, 0].set(0)

    # C_diag = jnp.sqrt(2)/2
    C_diag = 1

    woodbury_x_opt_ = woodbury_x_optimized(j, A_inv_x_, U_L, U_L_j, V_L, V_L_j, C_diag, ρ, σ, ν_κ_inv)
    return woodbury_x_opt_[0], woodbury_x_opt_[1], log_det_A_U_V_optimized(j, U_L, U_L_j, V_L, V_L_j, C_diag, n, ρ, σ, ν, κ_T, ν_κ_inv)
    # return woodbury_x(A_inv_x_, U, V, C_diag, C, n, ρ, σ, ν, κ_T), log_det_A_U_V(U, V, C, n, ρ, σ, ν, κ_T)

def log_det_A_U_V(U, V, C, n, ρ, σ, ν, κ_T):
    """ Compute the log determinant of the UV perturbation of A, using the Matrix Inversion Lemma. """

    L = ν.shape[0]
    ν_κ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T 
    Σ_Y_inv_ = Σ_Y_inv(C, n, ρ, σ)
    A_inv = jnp.block([
        [jnp.linalg.pinv(ν_κ), jnp.zeros((L, n))],
        [jnp.zeros((L, n)).T, Σ_Y_inv_]
    ])
    A = jnp.block([
        [ν_κ, jnp.zeros((L, n))],
        [jnp.zeros((L, n)).T, Σ_Y(C, n, ρ, σ)]
    ])
    # det_ν_κ = ν_κ[0, 0] * ν_κ[1, 1] - ν_κ[0, 1] * ν_κ[1, 0] + 1e-7 #just trying to see what makes the gradient stable
    # sign, log_det_ν_κ = jnp.linalg.slogdet(ν_κ) 
    log_det_ν_κ = jnp.log(jnp.linalg.det(ν_κ) + 1e-7) # just trying to see what makes the gradient stable
    log_det_diag = jnp.sum(jnp.log(jnp.diag(Σ_Y(C, n, ρ, σ)).flatten()))
    log_det_A = log_det_ν_κ + log_det_diag

    # sign, term2 = jnp.linalg.slogdet(jnp.eye(L) + V @ A_inv @ U) 
    # A_inv_U_top = cg(ν_κ, U[0:L, 0:L])[0]
    # A_inv_U_bot = cg(Σ_Y(C, n, ρ, σ), U[L:, 0:L])[0]
    # A_inv_U = jnp.block([
    #     [A_inv_U_top],
    #     [A_inv_U_bot]
    # ])
    # sign, term2 = jnp.linalg.slogdet(jnp.eye(L) + V @ A_inv_U) 

    sign, term2 = jnp.linalg.slogdet(jnp.eye(L) + V @ A_inv @ U)  # highly optimizable based on prev calc
    # term2 = 0
    return log_det_A + term2

def log_det_A_U_V_optimized(j, U_L, U_L_j, V_L, V_L_j, C_diag, n, ρ, σ, ν, κ_T, ν_κ_inv):
    """ Compute the log determinant of the UV perturbation of A, using the Matrix Inversion Lemma. Optimized to not compute any matrix multiplications of size n. """

    L = ν.shape[0]
    ν_κ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T 
    # Σ_Y_inv_diag_ = Σ_Y_inv_diag(C, n, ρ, σ).flatten()
    Σ_Y_inv_diag_j = 1/(ρ[0, 0] + σ**2)

    log_det_ν_κ = jnp.log(jnp.linalg.det(ν_κ) + 1e-12) # this makes the gradient stable, should play around with this value. Problem: this is zero when number of signals < L. 
    # log_det_diag = -jnp.sum(jnp.log(Σ_Y_inv_diag_))
    log_det_diag = - n * (jnp.log(Σ_Y_inv_diag_j))
    log_det_A = log_det_ν_κ + log_det_diag

    # ν_κ_inv = jnp.linalg.pinv(ν_κ)
    # A_inv_U = jnp.zeros((L + n, L))
    # A_inv_U = A_inv_U.at[0:L, 0:L].set(ν_κ_inv @ U[0:L, 0:L])
    # A_inv_U = A_inv_U.at[L+j, :].set(Σ_Y_inv_diag_[j] * U[L+j, :])

    # V_A_inv_U = V[0:L, 0:L] @ A_inv_U[0:L, 0:L] + jnp.outer(V[:, L+j], A_inv_U[L+j, :])

    # Memory efficient
    # V_A_inv_U = V_L @ ν_κ_inv @ U_L + jnp.outer(V_L_j.flatten(), Σ_Y_inv_diag_[j] * U_L_j.flatten())
    
    # More memory efficient
    V_A_inv_U = V_L @ ν_κ_inv @ U_L + jnp.outer(V_L_j.flatten(), Σ_Y_inv_diag_j * U_L_j.flatten())

    sign, term2 = jnp.linalg.slogdet(jnp.eye(2) + C_diag * V_A_inv_U)  # highly optimized based on prev calc
    return log_det_A + term2

def log_det_A_U_V_opt_update_no_j(U_L, U_L_j, V_L, V_L_j, ρ, σ, ν_κ_inv):
    L = U_L.shape[0]

    Σ_Y_inv_diag_j = 1/(ρ[0, 0] + σ**2)

    # Memory efficient
    V_A_inv_U = V_L @ ν_κ_inv @ U_L + jnp.outer(V_L_j.flatten(), Σ_Y_inv_diag_j * U_L_j.flatten())

    sign, term2 = jnp.linalg.slogdet(jnp.eye(L) + jnp.sqrt(2)/2 * V_A_inv_U)  # highly optimized based on prev calc
    return term2

def log_det_A_inv_U_V(log_det_A, A_inv, U, V):
    return log_det_A + jnp.linalg.slogdet(jnp.eye(V.shape[0]) + V @ A_inv @ U)[1]

def log_pdf_cg_opt(x, log_det_, x_A_inv_x, A_inv_x_update_L, A_inv_x_update_L_plus_j, L, j):
    """ Further optimized to not compute a vector - matrix multiplication (and storage of such vectors) """  
    return - (x.shape[0])/2 * jnp.log(2 * jnp.pi) - 1/2 * log_det_ - 1/2 * (x_A_inv_x - jnp.dot(x[:L].flatten(), A_inv_x_update_L.flatten()) - x[L+j] * A_inv_x_update_L_plus_j)

def log_pdf_cg(x, log_det_, Σ_inv_x):
    """ Compute the log pdf of a multivariate normal distribution using the conjugate gradient solution. """  
    return - (x.shape[0])/2 * jnp.log(2 * jnp.pi) - 1/2 * log_det_ - 1/2 * jnp.matmul(x.T, Σ_inv_x)

def Σ_V_Y_inv(j, C, n, ρ, σ, ν, κ_T):
    if C.shape != (n, 1): # Does this make the code slow for JIT compilation?  
        C = C.reshape((n, 1))
    L = 2
    # if not jnp.isscalar(j): j = j[0]
    # j = j[0]

    ρ̃ = (ν.T @ jnp.linalg.inv(ρ) @ ρ[:, C[j][0]]).flatten()
    # assert ρ̃.shape == (L,), ρ̃.shape

    # Create U for woodbury
    U = jnp.zeros((L + n, L))
    U = U.at[0:L, 0].set(ρ̃)
    U = U.at[0:L, 1].set(-ρ̃)
    U = U.at[L + j, 0].set(1/jnp.sqrt(2))
    U = U.at[L + j, 1].set(1/jnp.sqrt(2))
    # assert U.shape == (L + n, L), U.shape

    # Create V for woodbury
    V = jnp.zeros((L, L + n))
    V = V.at[0, 0:L].set(ρ̃)
    V = V.at[1, 0:L].set(ρ̃)
    V = V.at[0, L + j].set(1/jnp.sqrt(2))
    V = V.at[1, L + j].set(-1/jnp.sqrt(2))
    # assert V.shape == (L, L + n), V.shape

    C_diag = jnp.sqrt(2)/2

    A_inv = jnp.block([
        [jnp.linalg.pinv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T), jnp.zeros((L, n))],
        [jnp.zeros((L, n)).T, Σ_Y_inv(C, n, ρ, σ)]
    ])
    # A_inv = jnp.zeros((L + n, L + n))
    # A_inv = A_inv.at[0:L, 0:L].set(jnp.linalg.inv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T))
    # A_inv = A_inv.at[L:, L:].set(Σ_Y_inv(C, n, ρ, σ))
    # A_inv = jnp.eye(L + n, L + n)
    # else:
    #     A_inv = jnp.block([
    #         [ν_κ_T_inv, jnp.zeros((L, n))],
    #         [jnp.zeros((L, n)).T, Σ_Y_inv(C, n, ρ, σ)]
    #     ])
    # assert A_inv.shape == (L + n, L + n), A_inv.shape
    # assert jnp.isscalar(j), str("j: " + str(j))

    return woodbury_optimized_L_2(j, A_inv, U, C_diag, V) # Gives errors in autograd for now
    # return woodbury_C_diag(A_inv, U, C_diag, V)

def Cov_Z_Y_full(C, n, ρ): 
    """ Return an nL × n matrix of covariances between Z_B and Ȳ. """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    L = ρ.shape[0]
    indx = jnp.arange(0,n).reshape((n, ))
    Σ_ = jax.vmap(Cov_Z_Y, (0, None, None, None, None), 0)(indx, C, n, ρ).reshape((n * L, n))
    return Σ_

def Cov_V_Y_full(C, n, ρ, ν): 
    """ Return an nL × n matrix of covariances between V_Θ and Ȳ. """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    L = ν.shape[0]
    indx = jnp.arange(0,n).reshape((n, ))
    Σ_ = jax.vmap(Cov_V_Y, (0, None, None, None, None), 0)(indx, C, n, ρ, ν).reshape((n * L, n))
    return Σ_

def Σ_V_Y_full(C, n, ρ, σ, ν, κ_T):
    """ Returns an n(L+1) × n(L+1) matrix representing the full covariance matrix of the variable (V_Θ, Ȳ). """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    Cov_V_Y_full_ = Cov_V_Y_full(C, n, ρ, ν)
    A = [
        [Σ_V_full(n, ρ, ν, κ_T), Cov_V_Y_full_],
        [Cov_V_Y_full_.T, Σ_Y(C, n, ρ, σ)]
    ]
    return jnp.block(A)

def Σ_Z_Y_full(C, n, ρ, σ):
    """ Returns an n(L+1) × n(L+1) matrix representing the full covariance matrix of the variable (Z_B, Ȳ). """
    if C.shape != (n, 1):
        C = C.reshape((n, 1))
    Cov_Z_Y_full_ = Cov_Z_Y_full(C, n, ρ)
    A = [
        [Σ_Z_full(n, ρ), Cov_Z_Y_full_],
        [Cov_Z_Y_full_.T, Σ_Y(C, n, ρ, σ)]
    ]
    return jnp.block(A)

def Cov_ZV_Y(j, C, n, ρ, ν):
    return jnp.vstack((Cov_Z_Y(j, C, n, ρ), Cov_V_Y(j, C, n, ρ, ν)))


