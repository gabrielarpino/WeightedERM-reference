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
from tqdm.auto import tqdm
import WERM.posterior
from functools import partial

### --- Updated code made to be mapped by jax --- 

import jax
jax.config.update("jax_enable_x64", True) # Ensure float64 precision
import jax.numpy as jnp
import jaxopt
from jax import random
import jax.lax
import numpy as np # For CPU-based combination generation
import scipy.special # For combinations
import math # For math.ceil
import concurrent.futures # For CPU parallelization

# --- Proximal operator functions ---

def prox_obj_no_idx(t, args, M_fixed_func):
    """
    Objective function for the proximal operator.
    This is the function that prox_obj_no_idx minimizes.
    """
    z, b, π_i, v = args
    return (b * π_i * M_fixed_func(t, v) + (1/2) * jnp.linalg.norm(t - z)**2)

def prox(z, b, π_i, v, M_prime_func, M_double_prime_func, maxiter=30):
    """
    Proximal operator with respect to the loss b * π_i * M.
    Solves using a simple Newton's method.

    Args:
        z (jnp.ndarray): The 'z' parameter in the proximal objective.
        b (float): The 'b' parameter.
        π_i (float): The 'pi_i' parameter.
        v (jnp.ndarray): The 'v' parameter passed to M functions.
        M_prime_func (callable): First derivative of the M function.
        M_double_prime_func (callable): Second derivative of the M function.
        maxiter (int): Maximum iterations for Newton's method.

    Returns:
        jnp.ndarray: The solution of the proximal operator.
    """
    # Newton's method iteration body
    def body(carry, _):
        t = carry # Current estimate of t
        # Calculate gradient (g) and Hessian (H) of the proximal objective
        g = b * π_i * M_prime_func(t, v) + (t - z)
        H = b * π_i * M_double_prime_func(t, v) + 1.0
        update = g / H # Newton step
        t_new = t - update # New estimate of t
        return t_new, None # Return new t and discard scan carry output

    # Run Newton's method using jax.lax.scan for fixed number of iterations
    t_final, _ = jax.lax.scan(body, z, xs=None, length=maxiter)
    return t_final

# --- Helper functions for fixed-point problem setup (modified g_func) ---
def g_func_modified(z_i, v_θ_i, b, π_i, sample_idx_i, ε_i, q_func_dynamic_psi, M_prime_func, M_double_prime_func):
    """
    Computes the 'g' function used in the fixed-point equations,
    using sample_idx_i to dynamically derive psi_i.
    """
    # Call prox, which now uses q_func_dynamic_psi for its 'v' argument
    ḡ = prox(v_θ_i, b, π_i, q_func_dynamic_psi(z_i, sample_idx_i, ε_i), M_prime_func, M_double_prime_func)
    return (ḡ - v_θ_i) / b

# --- Fixed Point Problem Setup Function (JAX-traceable) ---

def _setup_fixed_point_problem(M_func, ρ, δ, η_, ϕ, ε, n, q_func, key, num_samples_mult=100):
    """
    Sets up the necessary JAX-compatible components (like random samples) for the
    fixed-point optimization problem for a single η_.
    This function is designed to be JAX-traceable and is called by vmap.

    Args:
        M_func (callable): The primary M function, M(t, v).
        ρ (jnp.ndarray): The ρ parameter (covariance matrix, L x L).
        δ (float): The δ parameter.
        η_ (jnp.ndarray): The change point vector (L-1 elements).
        ϕ (jnp.ndarray): The ϕ parameter (L x n matrix).
        ε (jnp.ndarray): The epsilon parameter (array of shape (n,)).
        n (int): The number of samples.
        q_func (callable): The q function, q(z_i, ψ_i, ε_i) -> scalar.
        key (jax.random.PRNGKey): A JAX random key for reproducibility.

    Returns:
        tuple: (L, Z_full, G_raw_full)
            L (int): The dimension L.
            Z_full (jnp.ndarray): Monte Carlo samples for Z.
            G_raw_full (jnp.ndarray): Monte Carlo samples for G_raw.
    """
    # Note: M_func, ρ, δ, ϕ, ε, n, q_func are passed through as is.
    # η_ and key are the vmapped arguments.

    L = ρ.shape[0] # Dimension L from the covariance matrix ρ

    # Split PRNG key for sampling
    key1, key_G = random.split(key)

    num_samples_mult = num_samples_mult # Number of samples used to estimate a single expectation

    # Sample Z_full and G_raw_full for Monte Carlo estimation
    Z_full = random.multivariate_normal(key1, jnp.zeros((L, )), ρ, shape=(num_samples_mult, n,))
    G_raw_full = random.multivariate_normal(key_G, jnp.zeros((L, )), jnp.eye(L), shape=(num_samples_mult, n,))

    return L, Z_full, G_raw_full

# --- Function to run a single fixed-point optimization ---
def _run_single_fp_optimization(M_func, ρ, δ, η_, ϕ, ε, n, q_func, L, Z_full, G_raw_full, tol=1e-6, maxiter=1000, lr=0.25):
    """
    Runs the fixed-point optimization for a single configuration using jaxopt.AndersonAcceleration.
    This function is NOT JAX-traceable due to potential CPU-based operations but the
    core solver is JIT-compiled.
    """
    # Define higher-order derivatives of M locally within this function
    M_prime_func = jax.grad(M_func, argnums=0)
    M_double_prime_func = jax.grad(M_prime_func, argnums=0)

    # Mapped versions of M derivatives and prox for batch operations
    M_prime_mapped = jax.vmap(M_prime_func, in_axes=(0, 0))
    M_double_prime_mapped = jax.vmap(M_double_prime_func, in_axes=(0, 0))

    # Partial application of prox_mapped to fix M_prime_func/M_double_prime_func
    prox_mapped = lambda z, b, π_i, v: jax.vmap(prox, in_axes=(0, None, 0, 0, None, None))(
        z, b, π_i, v, M_prime_func, M_double_prime_func
    )

    num_samples_mult = Z_full.shape[0] # Infer from Z_full

    # Helper to get psi_i from sample_idx and eta_vec, avoiding full psi array
    def get_psi_for_sample_idx(sample_idx, eta_vec):
        L_minus_1 = eta_vec.shape[0]
        if L_minus_1 == 0:
            return 0
        return jnp.searchsorted(eta_vec, sample_idx, side='right')

    # Wrapper for the user's q_func that computes psi_i on the fly
    def q_func_dynamic_psi(z_i, sample_idx_i, ε_scalar):
        psi_i = get_psi_for_sample_idx(sample_idx_i, η_)
        return q_func(z_i, psi_i, ε_scalar)

    q_mapped = jax.vmap(q_func_dynamic_psi, in_axes=(0, 0, 0))
    g_grad = jax.grad(g_func_modified, argnums=0)
    g_grad_mapped = jax.vmap(g_grad, in_axes=(0, 0, None, 0, 0, 0, None, None))

    # --- Functions for fixed-point equations (W_b, W_b_prime, W_z) ---
    def W_b(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, jnp.arange(n), ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_prime_mapped(
            prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input),
            q_mapped_with_input
        )) for ℓ in range(L)]
        return jnp.array(arr).T

    def W_b_prime(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, jnp.arange(n), ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_double_prime_mapped(
            prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input),
            q_mapped_with_input
        )) for ℓ in range(L)]
        return jnp.array(arr).T

    def W_z(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-7 * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        arr = [g_grad_mapped(Z, Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], jnp.arange(n), ε, q_func_dynamic_psi, M_prime_func, M_double_prime_func) for ℓ in range(L)]
        jnparr = jnp.array(arr)
        jnparr = jnp.moveaxis(jnparr, 0, -1)
        return jnparr

    def unpack_x(x_flat):
        b = x_flat[0:L]
        ν̃_flat = x_flat[L:L+L**2]
        ν̃ = ν̃_flat.reshape((L, L))
        L_flat = x_flat[L+L**2:]
        L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
        κ̃ = L_mat @ L_mat.T
        return b, ν̃, κ̃

    # --- Objective functions (residuals to be driven to zero) ---
    def obj_1(x):
        b, ν̃, κ̃ = unpack_x(x)
        body = lambda carry, idx: (carry + jnp.pow(W_b_prime(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]) + 1, -1).T @ jnp.ones((n,)) / n, None)
        rhs = jax.lax.scan(body, jnp.zeros((L,)), jnp.arange(num_samples_mult))[0] / num_samples_mult
        return (1 - 1/δ) * jnp.ones((L,)) - rhs

    def obj_2(x):
        b, ν̃, κ̃ = unpack_x(x)
        body = lambda carry, idx: (carry + Z_full[idx].T @ W_b(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]) / n, None)
        rhs = jax.lax.scan(body, jnp.zeros((L, L)), jnp.arange(num_samples_mult))[0] / num_samples_mult
        return jnp.zeros((L, L)) - rhs

    def obj_3(x):
        """Third fixed-point equation."""
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
        residual_matrix = κ̃ - rhs
        return residual_matrix[jnp.tril_indices(L)]

    def fsolve_obj(x):
        """
        Concatenates all objective functions into a single flattened array.
        """
        # obj_3 now returns a flat vector, so no need for .flatten() on it
        return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x)])

    fsolve_obj_jit = jax.jit(fsolve_obj)

    # --- Define the Fixed-Point Iteration with Projection ---
    # @jax.jit
    # def fixed_point_update(x):
    #     """
    #     Represents one step of the fixed-point iteration x = g(x),
    #     where g(x) = x - f(x), followed by a projection to the valid domain.
    #     """
    #     # Calculate the residual of the fixed-point equations
    #     residual = fsolve_obj_jit(x)
    #     # Propose the next iterate via a standard fixed-point step
    #     x_next = x - residual

    #     # --- Projection Step to Enforce Constraints ---
    #     # Unpack the flat vector into its components
    #     b_next = x_next[0:L]
    #     nu_tilde_flat_next = x_next[L:L+L**2]
    #     L_flat_next = x_next[L+L**2:]

    #     # Project b to be positive
    #     b_proj = jnp.maximum(1e-6, b_next)

    #     # Project the diagonal elements of the Cholesky factor L_mat to be positive
    #     tril_indices = jnp.tril_indices(L)
    #     is_diagonal_mask = (tril_indices[0] == tril_indices[1])
    #     L_flat_proj = jnp.where(is_diagonal_mask, jnp.maximum(1e-6, L_flat_next), L_flat_next)

    #     # Re-assemble the projected flat vector
    #     x_projected = jnp.concatenate([b_proj, nu_tilde_flat_next, L_flat_proj])
    #     return x_projected

    learning_rate = lr # New hyperparameter for fixed-point updates
    use_iterate_averaging = True # Whether to return the average of iterates or the final one
    # The fixed_point_update function now uses the learning_rate argument
    def fixed_point_update(x):
        residual = fsolve_obj_jit(x)
        x_next = x - learning_rate * residual
        
        # --- Projection Step ---
        b_next = x_next[0:L]
        nu_tilde_flat_next = x_next[L:L+L**2]
        L_flat_next = x_next[L+L**2:]
        b_proj = jnp.maximum(1e-6, b_next)
        tril_indices = jnp.tril_indices(L)
        is_diagonal_mask = (tril_indices[0] == tril_indices[1])
        L_flat_proj = jnp.where(is_diagonal_mask, jnp.maximum(1e-6, L_flat_next), L_flat_next)
        x_projected = jnp.concatenate([b_proj, nu_tilde_flat_next, L_flat_proj])
        return x_projected

    # --- Initial guess (x0) for the solver ---
    L0 = 0.75 * jnp.eye(L)
    L0_flat = L0[jnp.tril_indices(L)]
    x0 = jnp.concatenate([
        0.66 * jnp.ones((L,)),
        0.85 * jnp.ones((L, L)).flatten(),
        L0_flat
    ])

    # --- JIT-compiled solver loop for performance ---
    # We make use_iterate_averaging a static argument for efficient compilation
    @partial(jax.jit, static_argnames=("use_iterate_averaging",))
    def run_solver_loop(x0, use_iterate_averaging):
        # Solver is defined inside to capture hyperparams
        solver = jaxopt.AndersonAcceleration(
            fixed_point_fun=fixed_point_update,
            history_size=5,
            ridge=1e-8,
            maxiter=maxiter # Note: maxiter is used by fori_loop, not solver.run
        )

        # Define the body of the JIT-compatible loop
        def body_fun(k, state):
            x, x_avg, solver_state = state
            # Perform one update step
            x_next, solver_state_next = solver.update(params=x, state=solver_state)
            # Update the running average
            x_avg_next = (k * x_avg + x_next) / (k + 1)
            return x_next, x_avg_next, solver_state_next

        # Initialize the state for the loop
        initial_solver_state = solver.init_state(x0)
        initial_loop_state = (x0, x0, initial_solver_state)
        
        # Run the solver for a fixed number of iterations
        x_final, x_avg_final, _ = jax.lax.fori_loop(0, maxiter, body_fun, initial_loop_state)
        
        # Return either the averaged solution or the final iterate
        final_solution = jax.lax.cond(
            use_iterate_averaging,
            lambda: x_avg_final,
            lambda: x_final
        )
        return final_solution

    # Run the main solver loop
    final_solution = run_solver_loop(x0, use_iterate_averaging)

    # Unpack the results from the chosen solution
    b_res, nu_tilde_res, kappa_tilde_res = unpack_x(final_solution)

    # Check for convergence based on the final solution's residual norm
    success_status = jnp.linalg.norm(fsolve_obj_jit(final_solution)) < tol

    return b_res, nu_tilde_res, kappa_tilde_res, success_status

# --- Helper Function for constrained combinations (CPU-based) ---
def _unrank_combination_constrained(rank, k, effective_N_prime, effective_range_start, Delta):
    """
    Generates the k-combination of {0, ..., effective_N_prime-1} with the given lexicographical rank,
    and then applies the Delta constraint and boundary offset to transform it into the actual change points.
    """
    if k < 0 or k > effective_N_prime:
        raise ValueError(f"Invalid combination parameters: k={k}, effective_N_prime={effective_N_prime}")
    if rank < 0 or rank >= scipy.special.comb(effective_N_prime, k, exact=True):
        raise ValueError(f"Rank {rank} is out of bounds for comb({effective_N_prime}, {k}).")

    result_effective = np.zeros(k, dtype=np.int64)
    current_rank = rank
    j = effective_N_prime - 1

    for i in range(k, 0, -1):
        v = j
        while scipy.special.comb(v, i, exact=True) > current_rank:
            v -= 1

        result_effective[k - i] = v
        current_rank -= scipy.special.comb(v, i, exact=True)
        j = v - 1

    # Apply the transformation to get the actual change points
    eta_ = effective_range_start + np.sort(result_effective) + np.arange(k) * (Delta - 1)
    return eta_.astype(np.int32)

def compute_single_fixed_point(eta_vector, M_func, ρ, δ, ϕ, ε, n, q_func, master_key, num_samples_mult=100,maxiter=1000, lr=0.25):
    """
    Computes the fixed-point solution for a single, specified change-point vector.

    Args:
        eta_vector (jnp.ndarray): The specific change point vector to solve for.
        M_func (callable): The primary M function, M(t, v).
        ρ (jnp.ndarray): The ρ parameter (covariance matrix, L x L).
        δ (float): The δ parameter.
        ϕ (jnp.ndarray): The ϕ parameter (L x n matrix).
        ε (jnp.ndarray): The epsilon parameter (array of shape (n,)).
        n (int): The n parameter (num_samples in general_fp).
        q_func (callable): The q function, q(z_i, ψ_i, ε_i) -> scalar.
        master_key (jax.random.PRNGKey): A JAX random key for reproducibility.
        num_samples_mult (int): Number of Monte Carlo samples to use.

    Returns:
        tuple: A tuple containing (b, nu_tilde, kappa_tilde, success_status).
    """
    # Get dimension L from the covariance matrix ρ
    L = ρ.shape[0]

    # Setup the problem for this single case by generating the necessary random samples
    L_val, Z_full_val, G_raw_full_val = _setup_fixed_point_problem(
        M_func, ρ, δ, eta_vector, ϕ, ε, n, q_func, master_key, num_samples_mult=num_samples_mult
    )

    # Run the optimization for this single configuration
    b_res, nu_tilde_res, kappa_tilde_res, success_status = _run_single_fp_optimization(
        M_func, ρ, δ, eta_vector, ϕ, ε, n, q_func, L_val, Z_full_val, G_raw_full_val, maxiter=maxiter,lr=lr
    )

    return b_res, nu_tilde_res, kappa_tilde_res, success_status

# --- Wrapper function to compute fixed points for all valid eta's ---
def compute_all_fixed_points(n_total_samples, L, Delta, M_func, ρ, δ, ϕ, ε, n,
                             q_func, master_key, coarse_grain_factor=1,
                             batch_size=4096, use_cpu_parallelization=False, cpu_workers=None,
                             num_samples_mult=100):
    """
    Computes the general fixed point solution for a (potentially coarse-grained) set of
    valid change point vectors within the specified range and constraints.
    """
    num_change_points = L - 1 # k in combination terms

    if num_change_points == 0:
        # Handle L=1 case (no change points) by calling the single compute function
        print("L=1, no change points. Computing fixed point for single configuration (empty eta_).")
        eta_vec = jnp.array([], dtype=int)
        b_res, nu_tilde_res, kappa_tilde_res, success_status = compute_single_fixed_point(
            eta_vec, M_func, ρ, δ, ϕ, ε, n, q_func, master_key, num_samples_mult=num_samples_mult
        )
        return [(eta_vec, b_res, nu_tilde_res, kappa_tilde_res, success_status)]

    # Define the range for the transformed problem:
    effective_range_start = Delta
    effective_range_end = n_total_samples - Delta - (num_change_points - 1) * (Delta - 1)
    effective_N_prime = effective_range_end - effective_range_start + 1

    if effective_N_prime < num_change_points:
        print(f"No valid eta vectors possible for L={L}, Delta={Delta}, n_total_samples={n_total_samples}.")
        return []

    num_total_constrained_configs = int(scipy.special.comb(effective_N_prime, num_change_points, exact=True))
    print(f"Total valid constrained configurations for L={L}: {num_total_constrained_configs:,}")

    selected_ranks = list(range(0, num_total_constrained_configs, coarse_grain_factor))
    print(f"Processing {len(selected_ranks):,} configurations (coarse_grain_factor={coarse_grain_factor}).")

    all_results = []

    vmapped_setup_problem = jax.vmap(
        _setup_fixed_point_problem,
        in_axes=(None, None, None, 0, None, None, None, None, 0),
        out_axes=(None, 0, 0)
    )

    num_batches = math.ceil(len(selected_ranks) / batch_size)
    for i in range(num_batches):
        batch_start_idx = i * batch_size
        batch_end_idx = min((i + 1) * batch_size, len(selected_ranks))
        current_batch_ranks = selected_ranks[batch_start_idx:batch_end_idx]

        if not current_batch_ranks:
            continue

        if use_cpu_parallelization:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
                eta_list_np = list(executor.map(
                    lambda rank: _unrank_combination_constrained(rank, num_change_points, effective_N_prime, effective_range_start, Delta),
                    current_batch_ranks
                ))
            eta_batch_np = np.array(eta_list_np, dtype=np.int32)
        else:
            eta_batch_np = np.array([
                _unrank_combination_constrained(rank, num_change_points, effective_N_prime, effective_range_start, Delta)
                for rank in current_batch_ranks
            ], dtype=np.int32)

        eta_batch_jax = jnp.asarray(eta_batch_np)

        batch_keys_setup = jax.random.split(master_key, len(current_batch_ranks))
        master_key, _ = jax.random.split(master_key)

        L_val_batch, Z_full_batch, G_raw_full_batch = vmapped_setup_problem(
            M_func, ρ, δ, eta_batch_jax, ϕ, ε, n, q_func, batch_keys_setup
        )

        optimization_tasks = []
        if use_cpu_parallelization:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
                for j in range(len(current_batch_ranks)):
                    future = executor.submit(
                        _run_single_fp_optimization,
                        M_func, ρ, δ, eta_batch_jax[j], ϕ, ε, n, q_func,
                        L_val_batch, Z_full_batch[j], G_raw_full_batch[j]
                    )
                    optimization_tasks.append(future)

                for j, future in enumerate(optimization_tasks):
                    b_res, nu_tilde_res, kappa_tilde_res, success_status = future.result()
                    all_results.append((
                        eta_batch_np[j], b_res, nu_tilde_res, kappa_tilde_res, success_status
                    ))
        else:
            for j in range(len(current_batch_ranks)):
                b_res, nu_tilde_res, kappa_tilde_res, success_status = _run_single_fp_optimization(
                    M_func, ρ, δ, eta_batch_jax[j], ϕ, ε, n, q_func,
                    L_val_batch, Z_full_batch[j], G_raw_full_batch[j]
                )
                # Check if there are any NaNs in the results
                if jnp.isnan(b_res).any() or jnp.isnan(nu_tilde_res).any() or jnp.isnan(kappa_tilde_res).any():
                    print(f"Warning: NaN detected in results for eta {eta_batch_np[j]}. Marking as failed.")
                    success_status = False
                all_results.append((
                    eta_batch_np[j], b_res, nu_tilde_res, kappa_tilde_res, success_status
                ))

    return all_results

# --- Compute fixed points for all valid eta's naive (for loop) ---

def compute_all_fixed_points_naive(n, L, Delta_n, M, ρ, δ, ϕ, ε, q, coarse_grain_factor = 1, master_key=random.PRNGKey(0), est_b=False, X=None, y=None, num_samples_mult=100, maxiter=1000):
    """ Naive implementation of fixed point computation for all valid eta's, using a for loop.
    This function is not JAX-traceable, and is only kept for reference and testing purposes.
    It is not used in the main codebase, as it is too slow for large L and n.
    """
    
    chgpt_grid_absolute = WERM.posterior.generate_changepoint_grid(n, L, Delta_n, coarse_grain_factor=coarse_grain_factor)

    res_list = []
    for cp in tqdm(chgpt_grid_absolute):
        # b, nu, kappa = general_fp(M, ρ, δ, cp, ϕ, ε, n, q, est_b=est_b, X=X, y=y, Delta_n=Delta_n, num_samples_mult=num_samples_mult)
        # res_list.append((cp, b, nu, kappa))
        b, nu, kappa, success_status = compute_single_fixed_point(cp, M, ρ, δ, ϕ, ε, n, q, master_key=master_key, num_samples_mult=num_samples_mult, maxiter=maxiter)
        # Check if there are any NaNs in the results
        if jnp.isnan(b).any() or jnp.isnan(nu).any() or jnp.isnan(kappa).any():
            print(f"Warning: NaN detected in results for eta {cp}. Marking as failed.")
            success_status = False
        res_list.append((cp, b, nu, kappa, success_status))

    return res_list

# --- Compute all fixed points, traceable version (for vmap) ---

def _run_single_fp_iteration_traceable(
    M_func, ρ, δ, η_, ϕ, ε, n, q_func, L, Z_full, G_raw_full,
    learning_rate=0.25, max_iter=100, use_polyak_avg=True, early_stop_tol=1e-5
):
    """
    Runs a fixed number of naive fixed-point iterations for a single configuration.
    This function is JAX-traceable and designed to be used with jax.vmap.
    """
    # NOTE: All helper functions are defined *inside* this function.
    # This ensures that when jax.vmap traces this function, it captures
    # all the necessary logic and correctly handles closures over `η_`.

    # --- Start of Inner Function Definitions (copied from original) ---
    M_prime_func = jax.grad(M_func, argnums=0)
    M_double_prime_func = jax.grad(M_prime_func, argnums=0)
    M_prime_mapped = jax.vmap(M_prime_func, in_axes=(0, 0))
    M_double_prime_mapped = jax.vmap(M_double_prime_func, in_axes=(0, 0))
    prox_mapped = lambda z, b, π_i, v: jax.vmap(prox, in_axes=(0, None, 0, 0, None, None))(
        z, b, π_i, v, M_prime_func, M_double_prime_func
    )
    num_samples_mult = Z_full.shape[0]

    def get_psi_for_sample_idx(sample_idx, eta_vec):
        """
        Computes psi_i for a given sample index using jax.lax.cond
        in a way that is robust to JAX's tracer.
        """
        L_minus_1 = eta_vec.shape[0]

        # Define the two branches of the conditional as separate, named functions.
        # Both functions must accept the same arguments (operands).
        
        def true_branch(s_idx, _):
            """Branch for when eta_ is empty (L=1). Ignores eta_vec."""
            return jnp.zeros_like(s_idx, dtype=jnp.int32)

        def false_branch(s_idx, e_vec):
            """Branch for when eta_ has content (L>1)."""
            return jnp.searchsorted(e_vec, s_idx, side='right')

        # Call cond, explicitly passing the functions and the data they need.
        return jax.lax.cond(
            L_minus_1 == 0,
            true_branch,   # The function to run if true
            false_branch,  # The function to run if false
            sample_idx,    # First operand (passed as s_idx)
            eta_vec        # Second operand (passed as e_vec or _)
        )

    def q_func_dynamic_psi(z_i, sample_idx_i, ε_scalar):
        psi_i = get_psi_for_sample_idx(sample_idx_i, η_)
        return q_func(z_i, psi_i, ε_scalar)

    q_mapped = jax.vmap(q_func_dynamic_psi, in_axes=(0, 0, 0))
    g_grad = jax.grad(g_func_modified, argnums=0)
    g_grad_mapped = jax.vmap(g_grad, in_axes=(0, 0, None, 0, 0, 0, None, None))

    def W_b(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, jnp.arange(n), ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_prime_mapped(
            prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input),
            q_mapped_with_input
        )) for ℓ in range(L)]
        return jnp.array(arr).T

    def W_b_prime(b, ν̃, κ̃, Z, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + 1e-12 * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        q_mapped_with_input = q_mapped(Z, jnp.arange(n), ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_double_prime_mapped(
            prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input),
            q_mapped_with_input
        )) for ℓ in range(L)]
        return jnp.array(arr).T

    def unpack_x(x_flat):
        b = x_flat[0:L]
        ν̃_flat = x_flat[L:L+L**2]
        ν̃ = ν̃_flat.reshape((L, L))
        L_flat = x_flat[L+L**2:]
        L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
        κ̃ = L_mat @ L_mat.T
        return b, ν̃, κ̃

    def obj_1(x):
        b, ν̃, κ̃ = unpack_x(x)
        body = lambda carry, idx: (carry + jnp.pow(W_b_prime(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]) + 1, -1).T @ jnp.ones((n,)) / n, None)
        rhs = jax.lax.scan(body, jnp.zeros((L,)), jnp.arange(num_samples_mult))[0] / num_samples_mult
        return (1 - 1/δ) * jnp.ones((L,)) - rhs

    def obj_2(x):
        b, ν̃, κ̃ = unpack_x(x)
        body = lambda carry, idx: (carry + Z_full[idx].T @ W_b(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]) / n, None)
        rhs = jax.lax.scan(body, jnp.zeros((L, L)), jnp.arange(num_samples_mult))[0] / num_samples_mult
        return jnp.zeros((L, L)) - rhs

    def obj_3(x):
        b, ν̃, κ̃ = unpack_x(x)
        def body(carry, idx):
            term = δ * W_b(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]).T @ W_b(b, ν̃, κ̃, Z_full[idx], G_raw_full[idx]) / n
            return carry + term, None
        rhs = jax.lax.scan(body, jnp.zeros((L,L)), jnp.arange(num_samples_mult))[0] / num_samples_mult
        residual_matrix = κ̃ - rhs
        return residual_matrix[jnp.tril_indices(L)]

    # Flattened objective for the fixed-point iteration
    def fsolve_obj(x):
        return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x)])
    # --- End of Inner Function Definitions ---


    # --- Core Fixed-Point Iteration Logic ---
    def fixed_point_iteration_step(x):
        """Represents one step of the projected fixed-point iteration x_k+1 = g(x_k)."""
        residual = fsolve_obj(x)
        x_next = x - learning_rate * residual

        # --- Projection Step to Enforce Constraints ---
        b_next, nu_tilde_flat_next, L_flat_next = x_next[0:L], x_next[L:L+L**2], x_next[L+L**2:]
        b_proj = jnp.maximum(1e-9, b_next) # Project b to be positive
        tril_indices = jnp.tril_indices(L)
        is_diagonal_mask = (tril_indices[0] == tril_indices[1])
        L_flat_proj = jnp.where(is_diagonal_mask, jnp.maximum(1e-6, L_flat_next), L_flat_next)
        x_projected = jnp.concatenate([b_proj, nu_tilde_flat_next, L_flat_proj])
        return x_projected

    # 1. Define the condition function (when to continue the loop)
    def loop_cond(state):
        k, _, _, error = state
        # Continue if we're under max_iter AND the error is above the tolerance
        return jnp.logical_and(k < max_iter, error > early_stop_tol)

    # 2. Define the body function (one step of the iteration)
    def loop_body(state):
        k, x_k, x_avg_k, _ = state
        
        # Perform one fixed-point iteration step
        x_k_plus_1 = fixed_point_iteration_step(x_k)
        
        # Update Polyak-Ruppert average
        x_avg_k_plus_1 = (k * x_avg_k + x_k_plus_1) / (k + 1)
        
        # Calculate the new error as the norm of the change
        new_error = jnp.linalg.norm(x_k_plus_1 - x_k)
        
        return (k + 1, x_k_plus_1, x_avg_k_plus_1, new_error)

    # Initial guess (x0)
    L0 = 0.75 * jnp.eye(L)
    L0_flat = L0[jnp.tril_indices(L)]
    x0 = jnp.concatenate([
        0.66 * jnp.ones((L,)),
        0.85 * jnp.ones((L, L)).flatten(),
        L0_flat
    ])

    # 3. Set up the initial state for the loop
    # (iteration=0, x=x0, avg_x=x0, error=infinity)
    initial_state = (0, x0, x0, jnp.inf)

    # 4. Run the traceable while_loop
    _, x_final, x_avg_final, final_error = jax.lax.while_loop(loop_cond, loop_body, initial_state)

    # Select final or averaged solution
    final_solution = jax.lax.cond(
        use_polyak_avg,
        lambda: x_avg_final,
        lambda: x_final
    )

    # Unpack results and check final convergence status
    b_res, nu_tilde_res, kappa_tilde_res = unpack_x(final_solution)
    # The success status is now based on the loop's final error
    success_status = final_error < early_stop_tol

    return b_res, nu_tilde_res, kappa_tilde_res, success_status

def compute_all_fixed_points_vmap(n_total_samples, L, Delta, M_func, ρ, δ, ϕ, ε, n,
                             q_func, master_key, coarse_grain_factor=1,
                             batch_size=4096, # Batch size for memory management
                             num_samples_mult=100,
                             # New params for the traceable solver
                             max_iter=100,
                             fp_lr=0.25,
                             use_polyak_avg=True,
                             early_stop_tol=1e-5):
    """
    Computes fixed points for all valid eta configurations using jax.vmap
    for massive parallelization.
    """
    num_change_points = L - 1

    if num_change_points == 0:
        # L=1 case remains the same, but we call the new function
        print("L=1, no change points. Computing single configuration.")
        eta_vec = jnp.array([], dtype=jnp.int32)
        L_val, Z_full, G_raw = _setup_fixed_point_problem(M_func, ρ, δ, eta_vec, ϕ, ε, n, q_func, master_key, num_samples_mult)
        b_res, nu_res, kap_res, succ = _run_single_fp_iteration_traceable(
             M_func, ρ, δ, eta_vec, ϕ, ε, n, q_func, L_val, Z_full, G_raw, learning_rate=fp_lr, max_iter=max_iter, use_polyak_avg=use_polyak_avg, early_stop_tol=early_stop_tol
        )
        return [(eta_vec, b_res, nu_res, kap_res, succ)]


    # --- Generate all eta configurations (same as before) ---
    effective_range_start = Delta
    effective_range_end = n_total_samples - Delta - (num_change_points - 1) * (Delta - 1)
    effective_N_prime = effective_range_end - effective_range_start + 1

    if effective_N_prime < num_change_points:
        print(f"No valid eta vectors possible.")
        return []

    num_total_configs = int(scipy.special.comb(effective_N_prime, num_change_points, exact=True))
    selected_ranks = list(range(0, num_total_configs, coarse_grain_factor))
    print(f"Total valid configurations: {num_total_configs:,}. Processing {len(selected_ranks):,} of them.")

    all_results = []
    num_batches = math.ceil(len(selected_ranks) / batch_size)

    # --- JAX VMAP DEFINITIONS ---
    # vmap the setup function to generate random samples for each eta in a batch
    vmapped_setup = jax.vmap(
        _setup_fixed_point_problem,
        in_axes=(None, None, None, 0, None, None, None, None, 0),
        out_axes=(None, 0, 0)
    )

    # vmap the new traceable solver
    # We map over eta_, Z_full_batch, and G_raw_full_batch
    vmapped_solver = jax.vmap(
        partial(_run_single_fp_iteration_traceable, learning_rate=fp_lr, max_iter=max_iter, use_polyak_avg=use_polyak_avg, early_stop_tol=1e-5),
        in_axes=(None, None, None, 0, None, None, None, None, None, 0, 0)
    )


    # --- BATCH PROCESSING LOOP (for memory control) ---
    for i in tqdm(range(num_batches)):
        batch_start_idx = i * batch_size
        batch_end_idx = min((i + 1) * batch_size, len(selected_ranks))
        current_batch_ranks = selected_ranks[batch_start_idx:batch_end_idx]

        if not current_batch_ranks:
            continue

        # Generate eta vectors for the current batch (CPU)
        eta_batch_np = np.array([
            _unrank_combination_constrained(rank, num_change_points, effective_N_prime, effective_range_start, Delta)
            for rank in current_batch_ranks
        ], dtype=np.int32)
        eta_batch_jax = jnp.asarray(eta_batch_np)

        # Generate JAX keys for the batch
        batch_keys = jax.random.split(master_key, len(current_batch_ranks))
        master_key, _ = jax.random.split(master_key) # Get a new key for the next batch

        # 1. Setup problem for all etas in the batch at once
        L_val, Z_full_batch, G_raw_full_batch = vmapped_setup(
            M_func, ρ, δ, eta_batch_jax, ϕ, ε, n, q_func, batch_keys
        )

        # 2. Run the solver for all etas in the batch at once
        b_batch, nu_tilde_batch, kappa_tilde_batch, success_batch = vmapped_solver(
            M_func, ρ, δ, eta_batch_jax, ϕ, ε, n, q_func,
            L, Z_full_batch, G_raw_full_batch
        )

        # Check for NaNs in the batch results and mark failures
        for j in range(len(current_batch_ranks)):
            if jnp.isnan(b_batch[j]).any() or jnp.isnan(nu_tilde_batch[j]).any() or jnp.isnan(kappa_tilde_batch[j]).any():
                print(f"Warning: NaN detected in batch results for eta {eta_batch_np[j]}. Marking as failed.")
                success_batch = success_batch.at[j].set(False)

        # 3. Store results for the batch
        for j in range(len(current_batch_ranks)):
            all_results.append((
                eta_batch_np[j],
                b_batch[j],
                nu_tilde_batch[j],
                kappa_tilde_batch[j],
                success_batch[j]
            ))

    return all_results

### --- Original Code (still used) ---

# def prox_obj_no_idx(t, args): 
#     z, b, π_i, v = args
#     return (b * π_i * M_fixed(t, v) + (1/2) * jnp.linalg.norm(t - z)**2)

# # def prox(z, b, π_i, v, M_prime, M_double_prime, maxiter=30):
# #     """ Proximal operator with respect to the loss b π_i M """

# #     # def prox_obj(t): 
# #     #     """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
# #     #     return (b * π_i * M_fixed(t) + (1/2) * jnp.linalg.norm(t - z)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer

# #     # res = jso.minimize(prox_obj, jnp.array([1.0]), method="BFGS", options={"maxiter": 10000}) # Currently, this optimization is not always successful, even after increasing maxiter.
# #     # return res.x[0] # was giving nans

# #     # Solve using jaxopt, so that we can differentiate through the solution

# #     # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-5, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
# #     # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-7, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
# #     # lb = jaxopt.LBFGS(
# #     #     fun=prox_obj_no_idx, 
# #     #     maxiter=1e10, 
# #     #     tol = 1e-7, 
# #     #     stepsize = 1e-1, 
# #     #     implicit_diff=True
# #     # ) # Seems a little more stable and robust to parameters than gradient descent
# #     # args = (z, b, π_i, v)
# #     # res = gd.run(0.9*z, args).params
# #     # return res

# #     # Simple newton's method, fully JAX compatible
# #     def body(carry, _):
# #         t = carry
# #         g = b * π_i * M_prime(t, v) + (t - z)
# #         H = b * π_i * M_double_prime(t, v) + 1.0
# #         update = g / H
# #         t_new = t - update
# #         return t_new, None

# #     t_final, _ = jax.lax.scan(body, z, xs=None, length=maxiter)
# #     return t_final

def g(z_i, v_θ_i, b, π_i, ψ_i, ε_i, q):
    ḡ = prox(v_θ_i, b, π_i, q(z_i, ψ_i, ε_i))
    return (ḡ - v_θ_i) / b
g_grad = jax.grad(g, argnums=0)
g_grad_mapped = jax.vmap(g_grad, in_axes=(0, 0, None, 0, 0, 0, None))

def general_fp(M, ρ, δ, η_abs, ϕ, ε, n, q, est_b = False, X=None, y=None, Delta_n=None, num_samples_mult=100):
    """ η_abs is the change point vector with elements in {0, ..., n-1} and length L-1."""

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

    if η_abs.size == 0:
        ψ = jnp.zeros(num_samples).astype(int) 
    else:
        ψ = η_to_ψ(η_abs.astype(int), num_samples) # Seems to be correct, converts chgpt to signal config vector

    key = random.PRNGKey(0)
    key1, key_G = random.split(key)

    if est_b:
        chgpt_est, gof, B_hat = WERM.estimators.weighted_erm(
            X, y, M, L, Delta_n, 0.0, L_is_upper_bound=False,
            brute_force=False, return_gof=True, return_B_hat=True,
        )
        Theta_hat = X @ B_hat
        # If Theta_hat is provided, use it to compute b. 
        Theta_deb, b_hat = WERM.posterior.compute_debiased_theta(Theta_hat, y, M, ϕ, δ, output_b_hat=True)

    num_samples_mult = num_samples_mult # Number of samples used to estimate a single expectation, for posterior need roughly 100

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
        
    # def obj_3(x):
    #     b, ν̃, κ̃ = unpack_x(x)

    #     def body(carry, idx):
    #         Z = Z_full[idx, :, :]
    #         G_raw = G_raw_full[idx, :, :]
    #         Wb_sample = W_b(b, ν̃, κ̃, Z, G_raw)
    #         term = δ * Wb_sample.T @ Wb_sample / n
    #         return carry + term, None
        
    #     init = jnp.zeros((L, L))
    #     carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
    #     rhs = carry_final / num_samples_mult

    #     lhs = κ̃
    #     assert (lhs - rhs).shape == (L, L)
    #     return lhs - rhs

    def obj_3(x):
        """Third fixed-point equation."""
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

        # The full residual matrix
        residual_matrix = κ̃ - rhs
        
        # Return only the lower-triangular part to match the parameterization
        return residual_matrix[jnp.tril_indices(L)]

    def fsolve_obj(x): 
        return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x).flatten()])
    fsolve_obj_jit = jax.jit(fsolve_obj)

    # Solve using fsolve, faster than minimize with the jacobian
    # J_fsolve = jax.jacrev(fsolve_obj_jit, argnums=0)
    # L0 = 0.75 * jnp.eye(L)
    # L0_flat = L0[jnp.tril_indices(L)]

    # if est_b:
    #     b_0 = b_hat
    # else:
    #     b_0 = 1e-1 * jnp.ones((L,))
    # x0 = jnp.concatenate([
    #     b_0,
    #     0.85 * jnp.ones((L, L)).flatten(),
    #     L0_flat
    # ])
    # # x0 = jnp.concatenate([
    # #     0.66 * jnp.ones((L,)),
    # #     0.85 * jnp.ones((L,L)).flatten(),
    # #     0.75 * jnp.eye(L).flatten()
    # # ])

    # # Use constrained optimization instead of fsolve, seems to be more stable since enforcing b and diagonal of κ̃ to be positive
    # if est_b: 
    #     lower_b = b_hat * 0.9
    #     upper_b = b_hat * 1.1
    #     print(f"Using estimated b_hat = {b_hat} to constrain b in [{lower_b}, {upper_b}]")
    # else:
    #     lower_b = 1e-6 * jnp.ones((L,))
    #     upper_b = jnp.inf * jnp.ones((L,))

    # lower_nu = -jnp.inf * jnp.ones((L * L,))
    # L_indices = jnp.tril_indices(L)

    # # Set bounds for each element in L
    # lower_L = -jnp.inf * jnp.ones(len(L_indices[0]))
    # for i in range(L):  # enforce positivity on L_ii
    #     diag_idx = jnp.where((L_indices[0] == i) & (L_indices[1] == i))[0][0]
    #     lower_L = lower_L.at[diag_idx].set(1e-6)

    # lower = jnp.concatenate([lower_b, lower_nu, lower_L])
    
    # upper = jnp.concatenate([upper_b, jnp.inf * jnp.ones((L * L,)), jnp.inf * jnp.ones(len(L_indices[0]))])

    # # Solve:
    # res = scipy.optimize.least_squares(
    #     fsolve_obj_jit,
    #     x0,
    #     jac=J_fsolve,
    #     bounds=(lower, upper),
    #     xtol=1e-6,
    #     ftol=1e-6,
    #     max_nfev=200
    # )

    # print("Optimization Results: all = ", res, "success = ", res.success, ", message = ", res.message)
    # b, ν̃, κ̃ = unpack_x(res.x)

    # --- Define the Fixed-Point Iteration with Projection ---
    @jax.jit
    def fixed_point_update(x):
        """
        Represents one step of the fixed-point iteration x = g(x),
        where g(x) = x - f(x), followed by a projection to the valid domain.
        """
        # Calculate the residual of the fixed-point equations
        residual = fsolve_obj_jit(x)
        # Propose the next iterate via a standard fixed-point step
        x_next = x - residual

        # --- Projection Step to Enforce Constraints ---
        # Unpack the flat vector into its components
        b_next = x_next[0:L]
        nu_tilde_flat_next = x_next[L:L+L**2]
        L_flat_next = x_next[L+L**2:]

        # Project b to be positive
        b_proj = jnp.maximum(1e-6, b_next)

        # Project the diagonal elements of the Cholesky factor L_mat to be positive
        tril_indices = jnp.tril_indices(L)
        is_diagonal_mask = (tril_indices[0] == tril_indices[1])
        L_flat_proj = jnp.where(is_diagonal_mask, jnp.maximum(1e-6, L_flat_next), L_flat_next)

        # Re-assemble the projected flat vector
        x_projected = jnp.concatenate([b_proj, nu_tilde_flat_next, L_flat_proj])
        return x_projected
        # return x_next

    # --- Initial guess (x0) for the solver ---
    L0 = 0.75 * jnp.eye(L)
    L0_flat = L0[jnp.tril_indices(L)]

    x0 = jnp.concatenate([
        1e-1 * jnp.ones((L,)),
        0.85 * jnp.ones((L, L)).flatten(),
        L0_flat
    ])

    # --- Solve using jaxopt.AndersonAcceleration ---
    solver = jaxopt.AndersonAcceleration(
        fixed_point_fun=fixed_point_update,
        tol=1e-6,
        maxiter=100,
        history_size=3 # Can be tuned
    )

    sol = solver.run(init_params=x0)
    
    # Unpack the results
    b_res, nu_tilde_res, kappa_tilde_res = unpack_x(sol.params)
    
    # The solver has converged if the final error is below tolerance
    # and it finished in less than maxiter steps.
    success_status = sol.state.error < 1e-6

    return b_res, nu_tilde_res, kappa_tilde_res, sol.state.error
