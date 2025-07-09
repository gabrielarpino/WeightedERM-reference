import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
import numpy as np
import jax.scipy.optimize as jso
from jax import random
import jax.lax
import scipy.optimize
from WERM.config import η_to_ψ

def prox_obj_no_idx(t, args): 
    z, b, π_i, v = args
    return (b * π_i * M_fixed(t, v) + (1/2) * jnp.linalg.norm(t - z)**2)

def prox(z, b, π_i, v, M_prime, M_double_prime, maxiter=30):
    """ Proximal operator with respect to the loss b π_i M """

    # def prox_obj(t): 
    #     """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
    #     return (b * π_i * M_fixed(t) + (1/2) * jnp.linalg.norm(t - z)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer

    # res = jso.minimize(prox_obj, jnp.array([1.0]), method="BFGS", options={"maxiter": 10000}) # Currently, this optimization is not always successful, even after increasing maxiter.
    # return res.x[0] # was giving nans

    # Solve using jaxopt, so that we can differentiate through the solution

    # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-5, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-7, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    # lb = jaxopt.LBFGS(
    #     fun=prox_obj_no_idx, 
    #     maxiter=1e10, 
    #     tol = 1e-7, 
    #     stepsize = 1e-1, 
    #     implicit_diff=True
    # ) # Seems a little more stable and robust to parameters than gradient descent
    # args = (z, b, π_i, v)
    # res = gd.run(0.9*z, args).params
    # return res

    # Simple newton's method, fully JAX compatible
    def body(carry, _):
        t = carry
        g = b * π_i * M_prime(t, v) + (t - z)
        H = b * π_i * M_double_prime(t, v) + 1.0
        update = g / H
        t_new = t - update
        return t_new, None

    t_final, _ = jax.lax.scan(body, z, xs=None, length=maxiter)
    return t_final

def g(z_i, v_θ_i, b, π_i, ψ_i, ε_i, q):
    ḡ = prox(v_θ_i, b, π_i, q(z_i, ψ_i, ε_i))
    return (ḡ - v_θ_i) / b
g_grad = jax.grad(g, argnums=0)
g_grad_mapped = jax.vmap(g_grad, in_axes=(0, 0, None, 0, 0, 0, None))

def general_fp(M, ρ, δ, η_, ϕ, ε, n, q):

    global M_fixed
    M_fixed = M
    
    # Define higher order derivatives of M
    global M_prime
    M_prime = jax.grad(M, argnums=0)
    global M_double_prime
    M_double_prime = jax.grad(M_prime, argnums=0)
    # M_partial_12 = jax.grad(M_double_prime, argnums=1)
    M_prime_mapped = jax.vmap(M_prime, in_axes=(0, 0))
    M_double_prime_mapped = jax.vmap(M_double_prime, in_axes=(0, 0))
    # M_partial_12_mapped = jax.vmap(M_partial_12, in_axes=(0, 0))
    prox_mapped = lambda z, b, π_i, v: jax.vmap(prox, in_axes=(0, None, 0, 0, None, None))(z, b, π_i, v, M_prime, M_double_prime)

    num_samples = n
    L = ρ.shape[0]

    if η_.size == 0:
        ψ = jnp.zeros(num_samples).astype(int) 
    else:
        ψ = η_to_ψ((η_ * num_samples).astype(int), num_samples) # Seems to be correct, converts chgpt to signal config vector

    key = random.PRNGKey(0)
    key1, key_G = random.split(key)

    num_samples_mult = 30 # Number of samples used to estimate a single expectation

    Z_full = random.multivariate_normal(key1, jnp.zeros((L, )), ρ, shape=(num_samples_mult, n,)) # num_samp \times n × L
    # Pre-sample standard normal G_raw
    G_raw_full = random.multivariate_normal(key_G, jnp.zeros((L, )), jnp.eye(L), shape=(num_samples_mult, n,)) # num_samp × n × L

    q_mapped = jax.vmap(q, in_axes=(0, 0, 0)) # q_mapped(Z, ψ, ε) = q(Z @ ψ + ε)

    # q_mapped_input_test
    assert q_mapped(Z_full[0, :, :], ψ, ε).shape == (n,), "q_mapped(Z_full[0, :, :], ψ, ε) should be of shape (n,)"

    def W_b(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_prime_mapped( prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)] 
        assert jnp.array(arr).T.shape == (n, L)
        return jnp.array(arr).T

    def W_b_prime(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_double_prime_mapped( prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)]
        assert jnp.array(arr).T.shape == (n, L)
        return jnp.array(arr).T

    def W_z(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-7 * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        # print("g_grad_mapped shape: ", g_grad_mapped(Z, Z @ ν̃[:, 0] + G_sample[:, 0], b[0], ϕ[0, :], ψ, ε, q).shape)
        arr = [g_grad_mapped(Z, Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], ψ, ε, q) for ℓ in range(L)]
        # print("arr.shape = ", jnp.array(arr).shape)
        jnparr = jnp.array(arr)
        jnparr = jnp.moveaxis(jnparr, 0, -1) # move so that: n × dim of gradient \times L
        assert jnparr.shape == (n, L, L)
        return jnparr
    
    def unpack_x(x):
        """ Optimize the cholesky factorization of κ̃ instead of the full matrix. """
        b = x[0:L]
        ν̃_flat = x[L:L+L**2]
        ν̃ = ν̃_flat.reshape((L, L))
        L_flat = x[L+L**2:]
        L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
        κ̃ = L_mat @ L_mat.T
        return b, ν̃, κ̃

    def obj_1(x): 
        b, ν̃, κ̃ = unpack_x(x)
    
        def body(carry, idx):
            Z = Z_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            Wbp = W_b_prime(b, ν̃, κ̃, Z, G_raw)
            term = jnp.pow(Wbp + 1, -1).T @ jnp.ones((n,)) / n
            return carry + term, None
        
        init = jnp.zeros((L,))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = (1 - 1/δ) * jnp.ones((L,))
        assert (lhs - rhs).shape == (L,)
        return lhs - rhs

    def obj_2(x): 
        b, ν̃, κ̃ = unpack_x(x)

        def body(carry, idx):
            Z = Z_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            term = Z.T @ W_b(b, ν̃, κ̃, Z, G_raw) / n
            return carry + term, None
        
        init = jnp.zeros((L, L))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = jnp.zeros((L, L))
        assert (lhs - rhs).shape == (L, L)
        return lhs - rhs

    def obj_2_v2(x): 
        """ Use the explicit gradient of g̃. Does not yield as good results as stein's lemma method. 
        Maybe would need to write it out more explicitly in code rather than differentiating 
        the entire prox operation. """
        b, ν̃, κ̃ = unpack_x(x)

        def body(carry, idx):
            Z = Z_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            term = jnp.sum(W_z(b, ν̃, κ̃, Z, G_raw), axis = 0) / n
            return carry + term, None
        
        init = jnp.zeros((L, L))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = ν̃
        assert (lhs - rhs).shape == (L, L)
        return lhs - rhs
        
    def obj_3(x):
        b, ν̃, κ̃ = unpack_x(x)

        def body(carry, idx):
            Z = Z_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            Wb_sample = W_b(b, ν̃, κ̃, Z, G_raw)
            term = δ * Wb_sample.T @ Wb_sample / n
            return carry + term, None
        
        init = jnp.zeros((L, L))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = κ̃
        assert (lhs - rhs).shape == (L, L)
        return lhs - rhs

    def fsolve_obj(x): 
        return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x).flatten()])
    fsolve_obj_jit = jax.jit(fsolve_obj)

    # Solve using fsolve, faster than minimize with the jacobian
    J_fsolve = jax.jacrev(fsolve_obj_jit, argnums=0)
    L0 = 0.75 * jnp.eye(L)
    L0_flat = L0[jnp.tril_indices(L)]

    x0 = jnp.concatenate([
        0.66 * jnp.ones((L,)),
        0.85 * jnp.ones((L, L)).flatten(),
        L0_flat
    ])
    # x0 = jnp.concatenate([
    #     0.66 * jnp.ones((L,)),
    #     0.85 * jnp.ones((L,L)).flatten(),
    #     0.75 * jnp.eye(L).flatten()
    # ])

    # Use constrained optimization instead of fsolve, seems to be more stable since enforcing b and diagonal of κ̃ to be positive
    lower_b = 1e-6 * jnp.ones((L,))
    lower_nu = -jnp.inf * jnp.ones((L * L,))
    L_indices = jnp.tril_indices(L)

    # Set bounds for each element in L
    lower_L = -jnp.inf * jnp.ones(len(L_indices[0]))
    for i in range(L):  # enforce positivity on L_ii
        diag_idx = jnp.where((L_indices[0] == i) & (L_indices[1] == i))[0][0]
        lower_L = lower_L.at[diag_idx].set(1e-6)

    lower = jnp.concatenate([lower_b, lower_nu, lower_L])
    upper = jnp.inf * jnp.ones_like(lower)

    # Solve:
    res = scipy.optimize.least_squares(
        fsolve_obj_jit,
        x0,
        jac=J_fsolve,
        bounds=(lower, upper),
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=200
    )

    print("Optimization Results: all = ", res, "success = ", res.success, ", message = ", res.message)
    b, ν̃, κ̃ = unpack_x(res.x)

    return b, ν̃, κ̃