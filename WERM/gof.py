import jax
import jax.numpy as jnp
import numpy as np
from WERM.model import ζ
import matplotlib.pyplot as plt
from WERM.config import η_to_ψ_jax_mapped, η_to_ψ_jax_combinations_mapped
import math
from functools import partial
import scipy.special
from tqdm.auto import tqdm # For a nice progress bar

### --- New Binary Segmentation Code --- ###
### --- 1. NEW JIT-compiled Inner Helper Function --- ###
# This new function contains the logic we want to speed up.
# It operates on pre-sliced arrays, so their shapes are static from its perspective.
@partial(jax.jit, static_argnames=['M', 'L'])
def _jitted_cost_computation(segment_y, segment_z_all_prototypes, M, L):
    """
    JIT-compiled helper to compute the minimum cost for a given segment.
    
    Args:
        segment_y: A JAX array of shape (segment_length,).
        segment_z_all_prototypes: A JAX array of shape (segment_length, L).
        M: The dissimilarity function.
        L: The static number of prototypes.
    """
    def cost_for_prototype(l):
        # Slice out the l-th prototype for the segment
        segment_z_prototype = segment_z_all_prototypes[:, l]
        dissimilarity = M(segment_z_prototype, segment_y)
        return jnp.sum(dissimilarity)

    all_costs = jax.vmap(cost_for_prototype)(jnp.arange(L))
    return jnp.min(all_costs)
### --- 2. MODIFIED `_calculate_segment_cost` (NO JIT) --- ###
# This function is now a regular Python function. Its job is to do the
# dynamic slicing and then call our fast, jitted helper.
# REMOVED: @partial(jax.jit, static_argnames=['M', 'L'])
def _calculate_segment_cost(start, end, Z, y, M, L):
    """
    Calculates the cost for a single segment [start, end).
    This is now a non-jitted function that handles dynamic slicing.
    """
    # Perform dynamic slicing in standard Python/Numpy mode
    segment_y = y[start:end]
    segment_z_all = Z[start:end, :] # Slice all L prototypes for the segment

    # Call the jitted helper with the now statically-sized slices
    return _jitted_cost_computation(segment_y, segment_z_all, M, L)

# User's original GoF calculator, used only at the end if gof_opt_output is True
@partial(jax.jit, static_argnames=['M', 'γ', 'n'])
def _calculate_final_gof(η, Z, y, M, γ, log_prior, n):
    """
    JIT-compiled function to calculate GoF for a single change point configuration.
    Unchanged from your original code.
    """
    def η_to_ψ(η_in, n_in):
        boundaries = jnp.concatenate([jnp.array([0]), η_in, jnp.array([n_in])])
        indices = jnp.arange(n_in)
        return jnp.searchsorted(boundaries, indices, side='right') - 1

    ψ = η_to_ψ(η, n)
    # Z has shape (n, L), ψ has shape (n,). We need to select the right prototype for each point.
    # This assumes the segment index from ψ corresponds to the prototype index in Z.
    # A safer assumption is to re-calculate the best prototype for each segment.
    # For simplicity here, we stick to a simplified interpretation.
    # A more robust implementation would re-fit prototypes to segments post-hoc.
    z_for_y = Z[jnp.arange(n), ψ]
    dissimilarity = M(z_for_y, y)
    gof = jnp.sum(dissimilarity) + γ * log_prior
    return gof

### --- Main Binary Segmentation Function (with tqdm) --- ###
def chgpt_binary_segmentation(M, Z, y, n, L, min_window_size=30, 
                              gamma=0.0, # 1. USE GAMMA AS THE THRESHOLD
                              gof_opt_output=False):
    """
    Performs change point detection using the standard Binary Segmentation algorithm.

    Args:
        gamma (float): A penalty threshold for the gain. A new change point is
            added only if the gain from the split is strictly greater than gamma.
            - To find AT MOST L-1 change points, use gamma >= 0 (default is 0).
            - To find a FIXED number of L-1 change points, set gamma to a large
              negative number (e.g., -np.inf).
        gof_opt_output (bool): If True, returns the final GoF score.
    """
    k = L - 1
    if k <= 0:
        return np.array([], dtype=int)

    changepoints = []
    segments_to_check = [(0, n)]
    
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)

    pbar = tqdm(range(k), desc="Finding change points")
    for _ in pbar:
        max_gain = -np.inf
        best_split_info = None

        for i, (start, end) in enumerate(segments_to_check):
            if (end - start) < 2 * min_window_size:
                continue

            parent_cost = _calculate_segment_cost(start, end, Z_jax, y_jax, M, L)
            
            for split_point in range(start + min_window_size, end - min_window_size):
                cost_child1 = _calculate_segment_cost(start, split_point, Z_jax, y_jax, M, L)
                cost_child2 = _calculate_segment_cost(split_point, end, Z_jax, y_jax, M, L)
                
                current_gain = parent_cost - (cost_child1 + cost_child2)

                if current_gain > max_gain:
                    max_gain = current_gain
                    best_split_info = {
                        "split_point": split_point,
                        "parent_segment_index": i,
                        "new_segments": [(start, split_point), (split_point, end)]
                    }
        
        # --- 2. UNIFIED STOPPING CONDITION USING GAMMA ---
        # This single condition now handles all cases:
        # 1. No split is possible (max_gain remains -inf, which is <= gamma).
        # 2. The best possible split is not good enough (max_gain <= gamma).
        if max_gain <= gamma:
            tqdm.write(
                f"Stopping: Best gain ({max_gain:.2f}) "
                f"did not exceed threshold ({gamma:.2f})."
            )
            break
        # --- END OF NEW LOGIC ---

        changepoints.append(best_split_info["split_point"])
        parent_idx = best_split_info["parent_segment_index"]
        segments_to_check.pop(parent_idx)
        segments_to_check.extend(best_split_info["new_segments"])
        
        # Update the progress bar's description with the latest gain and threshold
        pbar.set_postfix(max_gain=f"{max_gain:.2f}", threshold=gamma)

    pbar.close()
    best_η = np.sort(np.array(changepoints, dtype=int))

    if gof_opt_output:
        # Note: The final GoF is calculated without an explicit penalty term here,
        # as the model selection was already performed by the gamma threshold.
        final_gof = _calculate_final_gof(jnp.asarray(best_η), Z_jax, y_jax, M, 1.0, 0.0, n)
        return best_η, final_gof
    else:
        return best_η


### --- Efficient GoF Search for Change Points --- ###

# --- JIT-compiled GoF calculator (unchanged) ---
@partial(jax.jit, static_argnames=['M', 'γ', 'n'])
def _calculate_gof_single(η, Z, y, M, γ, log_prior, n):
    """
    JIT-compiled function to calculate GoF for a single change point configuration.
    """
    def η_to_ψ(η_in, n_in):
        boundaries = jnp.concatenate([jnp.array([0]), η_in, jnp.array([n_in])])
        indices = jnp.arange(n_in)
        return jnp.searchsorted(boundaries, indices, side='right') - 1
    
    ψ = η_to_ψ(η, n)
    similarity = M(Z[jnp.arange(n), ψ], y)
    gof = jnp.sum(similarity) + γ * log_prior 
    return gof

# --- Main memory-efficient function (using the corrected helper) ---
def chgpt_gof_binary_seg_mem_eff(M, Z, y, n, L, Delta, γ=1.0, min_window_size=30, gof_opt_output=False):
    """
    Memory-efficient GoF search that generates configurations on the fly.
    """
    k = L - 1
    if k == 0:
        # if gof_list_output: return np.array([], dtype=int), np.array([0.0])
        if gof_opt_output: 
            similarity = M(Z, y)
            gof = jnp.sum(similarity) + γ * jnp.log(1.0)  # log(1.0) is zero
            return np.array([], dtype=int), gof
        else: return np.array([], dtype=int)
        
    num_positions = n - 1
    effective_positions = num_positions - (k - 1) * (Delta - 1) # This is correct
    
    if effective_positions < k:
        raise ValueError(f"Impossible to place {k} change points with separation {Delta} in {n} points.")

    num_total_configs = scipy.special.comb(effective_positions, k, exact=True)
    log_prior_penalty = jnp.log(num_total_configs)

    def get_η_from_index(idx):
        if idx >= num_total_configs:
            raise IndexError("Configuration index out of bounds.")
        
        y = _unrank_combination(idx, k, effective_positions)
        separations = (Delta - 1) * np.arange(k)
        η = y + separations + 1 
        return jnp.asarray(η, dtype=jnp.int32)

    num_levels = max(0, math.ceil(math.log2(num_total_configs / min_window_size)))
    start_idx, end_idx = 0, num_total_configs
    
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)

    for _ in range(num_levels):
        if end_idx - start_idx <= 1: break
            
        midpoint = (start_idx + end_idx) // 2
        q1_point = (start_idx + midpoint) // 2
        q3_point = (midpoint + end_idx) // 2
        
        test_indices = [q1_point, midpoint, q3_point]
        best_gof_step = jnp.inf
        best_region = 1
        
        for i, idx in enumerate(test_indices):
            η_test = get_η_from_index(idx)
            gof = _calculate_gof_single(η_test, Z_jax, y_jax, M, γ, log_prior_penalty, n)
            if gof < best_gof_step:
                best_gof_step = gof
                best_region = i
        
        if best_region == 0:
            end_idx = midpoint
        elif best_region == 1:
            start_idx, end_idx = q1_point, q3_point
        else:
            start_idx = midpoint

    best_gof = jnp.inf
    best_η = jnp.array([-1] * k, dtype=jnp.int32) # Initialize with correct shape
    final_gof_list = []

    for i in range(start_idx, end_idx):
        η = get_η_from_index(i)
        gof = _calculate_gof_single(η, Z_jax, y_jax, M, γ, log_prior_penalty, n)
        
        # if gof_list_output:
        #     final_gof_list.append(gof)
            
        if gof < best_gof:
            best_gof = gof
            best_η = η

    # if gof_list_output:
    #     return np.asarray(best_η), np.asarray(final_gof_list)
    if gof_opt_output:
        return np.asarray(best_η), best_gof 
    else:
        return np.asarray(best_η)

### --- Binary Segmentation Style GoF --- ###

def chgpt_gof_binary_seg(M, Z, y, η_arr, p_η_arr, γ = 1.0, combinations = True,
                         gof_list_output = False, min_window_size = 30):
    """
    Finds the best change point using a binary segmentation-style search.

    This function treats the ordered list of all possible change point 
    configurations as a search space. It iteratively halves this space until
    a small, promising window remains, which is then scanned exhaustively.

    Args:
        M: The model function.
        Z: The data.
        y: The target.
        η_arr: The array of change points.
        p_η_arr: The prior probability of each change point.
        γ: The penalization constant.
        combinations: Boolean flag for using combinations.
        gof_list_output: If True, returns the final GoF list from the last stage.
        min_window_size (int): The search stops subdividing when the window size
                               is smaller than this, performing a final full
                               scan on this window.

    Returns:
        The optimal change point array, and optionally the GoF list.
    """
    # --- FIX: Explicitly convert all NumPy inputs to JAX arrays ---
    # This ensures that all operations inside the traced function (fori_loop)
    # are performed on JAX-native types, preventing conversion errors.
    Z = jnp.asarray(Z)
    y = jnp.asarray(y)
    η_arr = jnp.asarray(η_arr)
    p_η_arr_ = jnp.asarray(p_η_arr) # Renamed to avoid shadowing

    L = Z.shape[1]
    if combinations:
        C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Z.shape[0], L)
    else:
        raise NotImplementedError("No Combinations not implemented for GoF function.")

    # Ensure C_s and η_idx are also JAX arrays before blocking
    xs = jnp.block([jnp.asarray(C_s), jnp.asarray(η_idx).reshape(-1, 1)])
    num_total_configs = xs.shape[0]

    def calculate_gof(config_vector):
        ψ, idx = config_vector[:-1], config_vector[-1]
        ψ_int = jnp.int32(ψ)
        
        similarity = M(Z[jnp.arange(Z.shape[0]), ψ_int], y)
        gof = jnp.sum(similarity, axis=0) - γ * jnp.log(p_η_arr_[jnp.int32(idx)])
        return gof

    num_levels = max(0, math.ceil(math.log2(num_total_configs / min_window_size)))

    def search_step(level, loop_state):
        start_idx, end_idx = loop_state
        midpoint = (start_idx + end_idx) // 2
        q1_point = (start_idx + midpoint) // 2
        q3_point = (midpoint + end_idx) // 2
        
        test_indices = jnp.array([q1_point, midpoint, q3_point])
        test_scores = jax.vmap(calculate_gof)(xs[test_indices])
        
        best_test_region = jnp.argmin(test_scores)
        possible_starts = jnp.array([start_idx, q1_point, midpoint])
        possible_ends = jnp.array([midpoint, q3_point, end_idx])
        
        new_start = possible_starts[best_test_region]
        new_end = possible_ends[best_test_region]
        return (new_start, new_end)

    initial_state = (0, num_total_configs)
    final_start, final_end = jax.lax.fori_loop(0, num_levels, search_step, initial_state)

    window_len = jnp.maximum(1, final_end - final_start)
    final_xs = jax.lax.dynamic_slice_in_dim(xs, final_start, window_len, axis=0)
    gof_list_final = jax.vmap(calculate_gof)(final_xs)
    
    best_local_idx = jnp.argmin(gof_list_final)
    i_opt = final_start + best_local_idx
    
    # We need the original numpy array for indexing to get the final result
    η_opt = np.asarray(η_arr)[np.asarray(η_idx)[i_opt]]
    
    η_final = η_opt[η_opt != -1]

    if gof_list_output:
        return η_final, np.array(gof_list_final)
    else:
        # Always return a tuple to prevent unpacking errors.
        return η_final, None

### --- Memory efficient brute force GoF --- ###

# --- Helper Function ---
def _unrank_combination(rank, k, n):
    """
    Generates the k-combination of {0, ..., n-1} with the given lexicographical rank.
    This is a standard, correct implementation using the combinadic method.
    It produces the combination in descending order.
    """
    if k < 0 or k > n:
        raise ValueError("Invalid combination parameters k, n")
    if rank < 0 or rank >= scipy.special.comb(n, k, exact=True):
        raise ValueError("Rank is out of bounds.")
        
    result = np.zeros(k, dtype=np.int64)
    current_rank = rank
    
    # Start with the largest possible element, n-1
    j = n - 1
    
    # Iterate from the largest element of the combination down to the smallest
    for i in range(k, 0, -1):
        # Find the largest element v (where v < j) for the current position
        v = j
        while scipy.special.comb(v, i, exact=True) > current_rank:
            v -= 1
        
        result[k - i] = v  # Store the element
        current_rank -= scipy.special.comb(v, i, exact=True)
        j = v - 1
            
    return np.sort(result) # Return in ascending order for consistency

# --- JIT-compiled GoF calculator for a BATCH of configurations ---
@partial(jax.jit, static_argnames=['M', 'γ', 'n'])
def _calculate_gof_for_batch(η_batch, Z, y, M, γ, log_prior, n):
    """
    Calculates GoF for a whole batch of η configurations using vmap.
    
    Args:
        η_batch: A JAX array of shape (batch_size, k)
    """
    # Inner function to process a single η.
    def _calculate_gof_single(η):
        # Convert η to segment assignment vector ψ
        boundaries = jnp.concatenate([jnp.array([0]), η, jnp.array([n])])
        indices = jnp.arange(n)
        ψ = jnp.searchsorted(boundaries, indices, side='right') - 1
        
        # Calculate GoF score
        similarity = M(Z[jnp.arange(n), ψ], y)
        gof = jnp.sum(similarity) + γ * log_prior 
        return gof
    
    # Vmap the single-config function over the batch axis of η_batch
    return jax.vmap(_calculate_gof_single)(η_batch)

def chgpt_gof_brute_force_mem_eff(M, Z, y, n, L, Delta, γ=1.0, batch_size=4096, gof_opt_output=False):
    """
    Memory-efficient GoF brute-force search that generates configurations on the fly.

    Args:
        M: The model function (e.g., loss function).
        Z: The data matrix (n x L).
        y: The target vector.
        n (int): The number of data points.
        L (int): The number of signals (implies L-1 change points).
        Delta (int): The minimum separation between change points.
        γ (float): The penalization constant.
        batch_size (int): The number of configurations to process in parallel.
                          Tune based on GPU memory. Larger is often faster.

    Returns:
        The optimal change point array.
    """
    k = L - 1
    # if gof_list_output: return np.array([], dtype=int), np.array([0.0])
    if k == 0:
        if gof_opt_output: 
            similarity = M(Z, y)
            gof = jnp.sum(similarity) + γ * jnp.log(1.0)  # log(1.0) is zero
            return np.array([], dtype=int), gof
        else: return np.array([], dtype=int)
        
    num_positions = n - 1
    effective_positions = num_positions - (k - 1) * (Delta - 1)
    
    if effective_positions < k:
        raise ValueError(f"Impossible to place {k} change points with separation {Delta} in {n} points.")

    num_total_configs = scipy.special.comb(effective_positions, k, exact=True)
    print(f"Starting brute-force search over {num_total_configs:,} configurations...")
    
    # Check if the search is feasible
    if num_total_configs > 100_000_000:
        print("\n--- WARNING ---")
        print(f"The number of configurations is extremely large. This search may take a very long time.")
        print("Consider using the binary search version or reducing the problem complexity.\n")

    log_prior_penalty = jnp.log(num_total_configs)
    
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)

    # --- Main Brute-Force Search Loop ---
    best_gof = jnp.inf
    best_η = np.array([-1] * k, dtype=np.int32)
    
    num_batches = math.ceil(num_total_configs / batch_size)

    for i in tqdm(range(num_batches), desc="Brute-force Search"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_total_configs)
        
        # 1. Generate a batch of configurations on the CPU
        indices = range(start_idx, end_idx)
        η_list = [_unrank_combination(idx, k, effective_positions) for idx in indices]
        
        # Add separation Delta and convert to JAX array
        separations = (Delta - 1) * np.arange(k)
        η_batch_np = np.array(η_list, dtype=np.int32) + separations.astype(np.int32) + 1
        η_batch_jax = jnp.asarray(η_batch_np)

        # 2. Calculate GoF for the entire batch on the GPU/TPU
        gof_scores_batch = _calculate_gof_for_batch(
            η_batch_jax, Z_jax, y_jax, M, γ, log_prior_penalty, n
        )
        
        # 3. Find the best result in the batch and update the overall best
        batch_best_idx = gof_scores_batch.argmin()
        batch_best_gof = gof_scores_batch[batch_best_idx]
        
        if batch_best_gof < best_gof:
            best_gof = batch_best_gof
            # Store the best configuration (as a NumPy array)
            best_η = η_batch_np[batch_best_idx]
            
    print(f"Search complete. Optimal GoF score: {best_gof:.4f}")
    if gof_opt_output:
        return best_η, best_gof
    else:
        return best_η if best_gof < jnp.inf else np.array([])

### --- General Goodness of Fit (GoF) --- ###

def chgpt_gof(M, Z, y, η_arr, p_η_arr, γ = 1.0, combinations = True, gof_list_output = False):
    """ General Goodness of Fit (GoF) function.
    M is the model, Z is the data, y is the target, η_arr is the array of change points,
    p_η_arr is the prior probability of each change point, and γ is the penalization constant."""
    
    L = Z.shape[1]
    if combinations:
        C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Z.shape[0], L)
    else:
        raise NotImplementedError("No Combinations not implemented for GoF function.")

    # Implement with jax.lax.scan. Uses less memory than vmap, and nearly as fast.
    gof_list = []
    p_η_arr_ = jnp.array(p_η_arr)

    def scan_fn(carry, x):
        y, Z = carry
        ψ, idx = x[:-1], x[-1]
        similarity = M(Z[jnp.arange(Z.shape[0]), ψ], y)
        # gof = jnp.linalg.norm(y - Z[jnp.arange(Z.shape[0]), ψ], ord=2) - γ * jnp.log(p_η_arr_[idx]) 
        gof = jnp.sum(similarity, axis=0) - γ * jnp.log(p_η_arr_[idx])
        return (y, Z), gof
    xs = jnp.block([C_s, η_idx.reshape(-1, 1)])
    _, gof_list = jax.lax.scan(scan_fn, (y, Z), xs)

    gof_list = np.array(gof_list)
    i_opt = gof_list.argmin()
    η_opt = η_arr[η_idx[i_opt]]

    if gof_list_output:
        return η_opt[η_opt != -1], gof_list
    else:
        return η_opt[η_opt != -1]

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