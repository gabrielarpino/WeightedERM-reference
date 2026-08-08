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
from joblib import Parallel, delayed

### --- Binary Segmentation Search --- TESTED. ###
def _chgpt_binary_segmentation(M, Z, y, n, L, Delta, γ=1.0, gof_opt_output=False, verbose=False):
    """
    Binary segmentation approach for change point detection.
    
    This finds exactly L-1 change points by greedily adding the change point
    that most improves the fit at each step.
    
    Args:
        M: The model function
        Z: The data matrix (n x L)
        y: The target vector
        n: Number of data points
        L: Number of signals (implies L-1 change points)
        Delta: Minimum separation between change points
        γ: Penalization constant (applied to final result only)
        gof_opt_output: Whether to return the GoF score
    
    Returns:
        Change point array (and optionally GoF score)
    """
    k = L - 1
    
    if k == 0:
        if gof_opt_output: 
            if Z.ndim == 1:
                Z = Z.reshape((n, 1))
            elif Z.ndim > 2 or Z.shape[1] != 1:
                raise ValueError("Z must be a vector or a matrix with one column.")
            # No log prior penalty for single segment
            loss = M(Z[:, 0], y).sum()
            return np.array([], dtype=int), loss
        else: 
            return np.array([], dtype=int)
    
    if verbose:
        print(f"Starting binary segmentation for exactly {k} change points...")
    
    # Initialize with no change points
    changepoints = []
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)
    
    # Iteratively add exactly k change points
    for iteration in range(k):
        if verbose:
            print(f"\nIteration {iteration + 1}: Looking for change point {iteration + 1}/{k}")
        
        # Calculate current loss (without log prior penalty during search)
        if len(changepoints) == 0:
            current_loss = _calculate_raw_loss(M, Z_jax, y_jax, jnp.array([]), n)
        else:
            current_loss = _calculate_raw_loss(M, Z_jax, y_jax, jnp.array(sorted(changepoints)), n)
        
        if verbose:
            print(f"  Current raw loss: {current_loss:.6f}")
        
        best_new_changepoint = None
        best_new_loss = current_loss
        
        # Try adding a change point at each valid position
        valid_positions_tested = 0
        improvements_found = []
        
        for candidate_pos in range(1, n):
            # Check if this position is valid (Delta constraints)
            valid = True
            for existing_cp in changepoints:
                if abs(candidate_pos - existing_cp) < Delta:
                    valid = False
                    break
            
            if not valid:
                continue
            
            valid_positions_tested += 1
            
            # Calculate loss with this new change point (no log prior)
            test_changepoints = sorted(changepoints + [candidate_pos])
            test_loss = _calculate_raw_loss(M, Z_jax, y_jax, jnp.array(test_changepoints), n)
            
            improvement = current_loss - test_loss  # Positive improvement = loss reduction
            improvements_found.append((candidate_pos, improvement, test_loss))
            
            # Update best if this is better (lower loss = better fit)
            if test_loss < best_new_loss:
                best_new_loss = test_loss
                best_new_changepoint = candidate_pos
        
        print(f"  Tested {valid_positions_tested} valid positions")
        if len(improvements_found) > 0:
            print(f"  Best 5 improvements: {sorted(improvements_found, key=lambda x: x[1], reverse=True)[:5]}")
        else:
            print(f"  No valid positions found! This suggests Delta={Delta} is too restrictive.")
            print(f"  Current changepoints: {changepoints}")
            print(f"  n={n}, need positions in range [1, {n-1}]")
        
        # Add the best change point (or least bad if no improvements)
        if best_new_changepoint is not None:
            changepoints.append(best_new_changepoint)
            changepoints.sort()
            improvement = current_loss - best_new_loss  # Positive = loss reduction
            if verbose:
                if improvement > 0:
                    print(f"  Added change point at position {best_new_changepoint}")
                    print(f"  Raw loss improved from {current_loss:.4f} to {best_new_loss:.4f} (improvement: {improvement:.4f})")
                else:
                    print(f"  Added change point at position {best_new_changepoint} (forced - no improvement available)")
                    print(f"  Raw loss changed from {current_loss:.4f} to {best_new_loss:.4f} (change: {improvement:.4f})")
        else:
            # Force finding a change point even if no improvement
            if verbose:
                print(f"  No change point improves loss - forcing selection of least bad option...")
            
            # Find the change point that increases loss the least
            least_bad_pos = None
            least_bad_loss = jnp.inf
            
            for candidate_pos in range(1, n):
                # Check Delta constraints
                valid = True
                for existing_cp in changepoints:
                    if abs(candidate_pos - existing_cp) < Delta:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate loss for this position
                test_changepoints = sorted(changepoints + [candidate_pos])
                test_loss = _calculate_raw_loss(M, Z_jax, y_jax, jnp.array(test_changepoints), n)
                
                if test_loss < least_bad_loss:
                    least_bad_loss = test_loss
                    least_bad_pos = candidate_pos
            
            if least_bad_pos is not None:
                changepoints.append(least_bad_pos)
                changepoints.sort()
                if verbose:
                    loss_increase = least_bad_loss - current_loss
                    print(f"  Forced change point at position {least_bad_pos}")
                    print(f"  Raw loss increased from {current_loss:.4f} to {least_bad_loss:.4f} (increase: {loss_increase:.4f})")
            else:
                # This should only happen if Delta constraints make it impossible
                if verbose:
                    print(f"  IMPOSSIBLE: Cannot place change point due to Delta constraints!")
                    print(f"  Delta={Delta}, n={n}, existing changepoints: {changepoints}")
                    print(f"  Stopping with {len(changepoints)} change points instead of {k}")
                break
    
    # Check if we found the requested number of change points
    if len(changepoints) < k:
        if verbose:
            print(f"WARNING: Only found {len(changepoints)} change points out of {k} requested")
            print("This is likely due to Delta constraints being too restrictive")
        
        # Try to force additional change points if possible
        remaining_needed = k - len(changepoints)
        if verbose:
            print(f"Attempting to force {remaining_needed} additional change points...")
        
        for _ in range(remaining_needed):
            # Find any valid position that satisfies Delta constraints
            forced_pos = None
            for candidate_pos in range(1, n):
                # Check Delta constraints
                valid = True
                for existing_cp in changepoints:
                    if abs(candidate_pos - existing_cp) < Delta:
                        valid = False
                        break
                
                if valid:
                    forced_pos = candidate_pos
                    break
            
            if forced_pos is not None:
                changepoints.append(forced_pos)
                changepoints.sort()
                if verbose:
                    print(f"  Forced change point at position {forced_pos}")
            else:
                if verbose:
                    print(f"  Cannot force more change points - Delta constraints prevent it")
                break
                
    elif len(changepoints) > k:
        # This shouldn't happen with our algorithm, but just in case
        if verbose:
            print(f"WARNING: Found {len(changepoints)} change points, truncating to {k}")
        changepoints = changepoints[:k]
    
    # Local refinement (still without log prior)
    if len(changepoints) > 0:
        print("\nApplying local refinement...")
        changepoints = _local_refinement_binary_seg(
            M, Z_jax, y_jax, n, changepoints, Delta
        )
    
    result = np.array(changepoints, dtype=int)
    
    if gof_opt_output and len(result) > 0:
        # Calculate final GoF with log prior penalty only at the end
        num_positions = n - 1
        effective_positions = num_positions - (len(result) - 1) * (Delta - 1)
        num_configs = scipy.special.comb(effective_positions, len(result), exact=True)
        log_prior = jnp.log(L * num_configs)  # Use L as Lmax
        
        final_gof = _calculate_gof_single_config(
            jnp.array(result), Z_jax, y_jax, M, γ, log_prior, n
        )
        return result, final_gof
    elif gof_opt_output:
        # Return infinite GoF if we couldn't find change points
        return result, jnp.inf
    else:
        return result

def _calculate_raw_loss(M, Z, y, changepoints, n):
    """
    Calculate the raw loss (without log prior penalty).
    Each segment uses the corresponding column of Z (segment 0 -> column 0, etc.)
    Lower loss = better fit (consistent with M function convention).
    """
    if len(changepoints) == 0:
        # No change points - single segment using column 0
        losses = M(Z[:, 0], y)
        return jnp.sum(losses)
    else:
        # Multiple segments
        boundaries = jnp.concatenate([jnp.array([0]), changepoints, jnp.array([n])])
        
    indices = jnp.arange(n)
    ψ = jnp.searchsorted(boundaries, indices, side='right') - 1
    
    # Each segment uses its corresponding column: segment i uses Z[:, i]
    losses = M(Z[jnp.arange(n), ψ], y)
    # return jnp.sum(losses)
    return jnp.mean(losses) # So that these GoFs can be compared over different scales

def _local_refinement_binary_seg(M, Z, y, n, changepoints, Delta, verbose=False, max_iterations=5):
    """
    Local refinement for binary segmentation (without log prior penalty).
    Works with loss convention (lower is better).
    """
    changepoints = list(changepoints)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(changepoints)):
            current_pos = changepoints[i]
            best_pos = current_pos
            
            # Calculate current raw loss
            current_loss = _calculate_raw_loss(M, Z, y, jnp.array(changepoints), n)
            best_loss = current_loss
            
            # Try moving this change point
            for offset in [-2, -1, 1, 2]:
                new_pos = current_pos + offset
                if new_pos < 1 or new_pos >= n:
                    continue
                
                # Check Delta constraint
                valid = True
                for j, other_pos in enumerate(changepoints):
                    if i != j and abs(new_pos - other_pos) < Delta:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Test this new position
                test_changepoints = changepoints.copy()
                test_changepoints[i] = new_pos
                
                test_loss = _calculate_raw_loss(M, Z, y, jnp.array(test_changepoints), n)
                
                if test_loss < best_loss:  # Lower loss is better
                    best_loss = test_loss
                    best_pos = new_pos
            
            # Update if we found an improvement
            if best_pos != current_pos:
                changepoints[i] = best_pos
                improved = True
                if verbose:
                    print(f"  Moved change point from {current_pos} to {best_pos}")
        
        changepoints.sort()
    
    return changepoints

def chgpt_binary_segmentation(M, Z, y, n, L, Delta, γ=1.0, L_is_upper_bound=False, gof_opt_output=False, verbose=False):
    """
    Main interface for binary segmentation change point detection.
    
    Note: When L_is_upper_bound=True, this will try different values of L
    and apply the log prior penalty for model selection. When False,
    it finds exactly L-1 change points without log prior considerations.
    """
    if not L_is_upper_bound:
        return _chgpt_binary_segmentation(M, Z, y, n, L, Delta, γ, gof_opt_output, verbose)
    else:
        # If L is an upper bound, search for the best L
        η_list = []
        gof_list = []
        
        for L_current in tqdm(range(1, L + 1), desc="Binary segmentation - searching for best L"):
            if verbose:
                print(f"\nBinary segmentation with L = {L_current}...")
            η_current, gof_current = _chgpt_binary_segmentation(
                M, Z[:, :L_current], y, n, L_current, Delta, γ, 
                gof_opt_output=True, verbose=verbose
            )
            
            assert gof_current.ndim == 0, "GoF current should be a scalar"
            η_list.append(η_current)
            gof_list.append(gof_current)
        
        if verbose:
            print("\nBinary Segmentation Results:")
            for i, (η, gof) in enumerate(zip(η_list, gof_list)):
                print(f"  L={i+1}: GoF={gof:.4f}, change points={η}")
        
        gof_list = np.array(gof_list)
        i_opt = gof_list.argmin()  # Lower GoF is better (includes penalty)
        best_η = η_list[i_opt]
        best_gof = gof_list[i_opt]
        
        if verbose:
            print(f"\nOptimal: GoF={best_gof:.4f} for L={i_opt + 1}")
            print(f"Optimal change points: {best_η}")
        
        if gof_opt_output:
            return best_η, best_gof
        else:
            return best_η

### --- Efficient, approximate divide and conquer approach - NOT FULLY CORRECT --- ###

@partial(jax.jit, static_argnames=['M', 'γ', 'n'])
def _calculate_gof_single_config(η, Z, y, M, γ, log_prior, n):
    """Calculate GoF for a single configuration."""
    boundaries = jnp.concatenate([jnp.array([0]), η, jnp.array([n])])
    indices = jnp.arange(n)
    ψ = jnp.searchsorted(boundaries, indices, side='right') - 1
    
    similarity = M(Z[jnp.arange(n), ψ], y)
    gof = jnp.sum(similarity) + γ * log_prior 
    return gof

# --- NEW: Approximate Divide-and-Conquer Method ---
def _find_best_single_changepoint(M, Z, y, start, end, Delta, γ, log_prior, n, existing_changepoints=None):
    """
    Find the best single change point in the range [start, end).
    
    Args:
        M: Model function
        Z, y: Data
        start, end: Range to search in
        Delta: Minimum separation
        γ: Penalty parameter
        log_prior: Prior penalty
        n: Total number of data points
        existing_changepoints: List of already placed change points
    
    Returns:
        best_position: Best change point position, or None if no valid position
        best_gof: Best GoF score achieved
    """
    if existing_changepoints is None:
        existing_changepoints = []
    
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)
    
    # Find valid positions (respecting Delta separation from existing change points)
    valid_positions = []
    for pos in range(start + 1, end):  # Change points are at positions 1 to n-1
        valid = True
        for existing_cp in existing_changepoints:
            if abs(pos - existing_cp) < Delta:
                valid = False
                break
        if valid:
            valid_positions.append(pos)
    
    if not valid_positions:
        return None, jnp.inf
    
    best_gof = jnp.inf
    best_position = None
    
    # Test each valid position
    for pos in valid_positions:
        # Create configuration with this change point
        test_changepoints = sorted(existing_changepoints + [pos])
        η = jnp.array(test_changepoints)
        
        gof = _calculate_gof_single_config(η, Z_jax, y_jax, M, γ, log_prior, n)
        
        if gof < best_gof:
            best_gof = gof
            best_position = pos
    
    return best_position, best_gof

def _chgpt_gof_divide_conquer_L(M, Z, y, n, L, Delta, γ=1.0, gof_opt_output=False, Lmax=None):
    """
    Approximate change point detection using divide-and-conquer approach.
    
    This method recursively finds the best single change point in each segment,
    which is much faster than brute force but may not find the global optimum.
    
    Args:
        M: The model function
        Z: The data matrix (n x L)
        y: The target vector
        n: Number of data points
        L: Number of signals (implies L-1 change points)
        Delta: Minimum separation between change points
        γ: Penalization constant
        gof_opt_output: Whether to return the GoF score
    
    Returns:
        Change point array (and optionally GoF score)
    """
    k = L - 1
    
    if k == 0:
        if gof_opt_output: 
            # Assert that Z is either a vector or a matrix with one column
            if Z.ndim == 1:
                Z = Z.reshape((n, 1))
            elif Z.ndim > 2 or Z.shape[1] != 1:
                raise ValueError("Z must be a vector or a matrix with one column.")
            gof = M(Z[:, 0], y).sum()
            return np.array([], dtype=int), gof
        else: return np.array([], dtype=int)
    
    if Lmax is None:
        Lmax = L

    # Check feasibility
    min_required_length = 1 + (k - 1) * Delta + 1  # 1 + gaps + 1
    if n < min_required_length:
        raise ValueError(f"Impossible to place {k} change points with separation {Delta} in {n} points.")
    
    # Calculate log prior penalty (approximation - use similar logic as brute force)
    num_positions = n - 1
    effective_positions = num_positions - (k - 1) * (Delta - 1)
    num_total_configs = scipy.special.comb(effective_positions, k, exact=True)
    log_prior_penalty = jnp.log(Lmax * num_total_configs)
    
    print(f"Starting divide-and-conquer search for {k} change points...")
    
    # Recursively find change points
    def find_changepoints_recursive(start, end, num_remaining):
        """Recursively find change points in the range [start, end)."""
        if num_remaining == 0:
            return []
        
        if num_remaining == 1:
            # Find the single best change point in this range
            best_pos, _ = _find_best_single_changepoint(
                M, Z, y, start, end, Delta, γ, log_prior_penalty, n, []
            )
            return [best_pos] if best_pos is not None else []
        
        # For multiple change points, divide the range
        mid = (start + end) // 2
        
        # Ensure we can fit the remaining change points
        left_points = num_remaining // 2
        right_points = num_remaining - left_points
        
        # Adjust mid to ensure feasibility
        min_left_end = start + left_points * Delta + 1
        max_left_end = end - right_points * Delta - 1
        mid = max(min_left_end, min(mid, max_left_end))
        
        # Recursively find change points in each half
        left_changepoints = find_changepoints_recursive(start, mid, left_points)
        right_changepoints = find_changepoints_recursive(mid, end, right_points)
        
        return left_changepoints + right_changepoints
    
    # Find initial change points
    changepoints = find_changepoints_recursive(0, n, k)
    changepoints = [cp for cp in changepoints if cp is not None]
    changepoints.sort()
    
    # If we didn't find enough change points, try a different approach
    if len(changepoints) < k:
        print(f"Warning: Only found {len(changepoints)} change points, expected {k}")
        print("Falling back to greedy approach...")
        
        # Greedy approach: keep adding the best single change point
        while len(changepoints) < k:
            best_pos = None
            best_gof = jnp.inf
            
            # Try all valid positions
            for pos in range(1, n):
                # Check if position is valid
                valid = True
                for existing_cp in changepoints:
                    if abs(pos - existing_cp) < Delta:
                        valid = False
                        break
                if not valid or pos in changepoints:
                    continue
                
                # Test this position
                test_changepoints = sorted(changepoints + [pos])
                η = jnp.array(test_changepoints)
                gof = _calculate_gof_single_config(
                    η, jnp.asarray(Z), jnp.asarray(y), M, γ, log_prior_penalty, n
                )
                
                if gof < best_gof:
                    best_gof = gof
                    best_pos = pos
            
            if best_pos is not None:
                changepoints.append(best_pos)
                changepoints.sort()
            else:
                break
    
    # Final optimization: local refinement
    if len(changepoints) == k:
        changepoints = _local_refinement(
            M, Z, y, n, changepoints, Delta, γ, log_prior_penalty
        )
    
    result = np.array(changepoints, dtype=int)
    
    if gof_opt_output and len(result) > 0:
        final_gof = _calculate_gof_single_config(
            jnp.array(result), jnp.asarray(Z), jnp.asarray(y), 
            M, γ, log_prior_penalty, n
        )
        return result, final_gof
    elif gof_opt_output:
        return result, jnp.inf
    else:
        return result

def _local_refinement(M, Z, y, n, changepoints, Delta, γ, log_prior, max_iterations=5):
    """
    Local refinement: try to improve each change point by moving it slightly.
    """
    changepoints = list(changepoints)
    Z_jax, y_jax = jnp.asarray(Z), jnp.asarray(y)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(changepoints)):
            current_pos = changepoints[i]
            best_pos = current_pos
            
            # Calculate current GoF
            η = jnp.array(changepoints)
            current_gof = _calculate_gof_single_config(
                η, Z_jax, y_jax, M, γ, log_prior, n
            )
            best_gof = current_gof
            
            # Try moving this change point
            for offset in [-2, -1, 1, 2]:
                new_pos = current_pos + offset
                if new_pos < 1 or new_pos >= n:
                    continue
                
                # Check Delta constraint
                valid = True
                for j, other_pos in enumerate(changepoints):
                    if i != j and abs(new_pos - other_pos) < Delta:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Test this new position
                test_changepoints = changepoints.copy()
                test_changepoints[i] = new_pos
                η = jnp.array(test_changepoints)
                
                gof = _calculate_gof_single_config(
                    η, Z_jax, y_jax, M, γ, log_prior, n
                )
                
                if gof < best_gof:
                    best_gof = gof
                    best_pos = new_pos
            
            # Update if we found an improvement
            if best_pos != current_pos:
                changepoints[i] = best_pos
                improved = True
        
        changepoints.sort()
    
    return changepoints

def chgpt_gof_divide_conquer(M, Z, y, n, L, Delta, γ=1.0, L_is_upper_bound=False, gof_opt_output=False):

    if not L_is_upper_bound:
        return _chgpt_gof_divide_conquer_L(M, Z, y, n, L, Delta, γ, gof_opt_output)
    else:
        # If L is an upper bound, we need to search for the best L
        best_η = np.array([], dtype=int)
        best_gof = jnp.inf

        # Rewrite the above using a candidate list
        η_list = []
        gof_list = []
        for L_current in tqdm(range(1, L + 1), desc="Searching for best L"):
            print(f"Searching for best L = {L_current}...")
            η_current, gof_current = _chgpt_gof_divide_conquer_L(
                M, Z[:, :L_current], y, n, L_current, Delta, γ, gof_opt_output=True, Lmax=L
            )
            # Assert gof_current is a scalar
            assert gof_current.ndim == 0, "GoF current should be a scalar"

            η_list.append(η_current)
            gof_list.append(gof_current)

        print("GOF List:", gof_list)
        gof_list = np.array(gof_list)
        # Print the gof_list and the chgpt_list
        print("L_is_upper_bd. GoF List:", gof_list)
        print("L_is_upper_bd. Chgpt List:", η_list)
        i_opt = gof_list.argmin()
        best_η = η_list[i_opt]
        best_gof = gof_list[i_opt]
        print(f"Optimal GoF score: {best_gof:.4f} for L = {i_opt + 1}")
        print(f"Optimal change points: {best_η}")

        if gof_opt_output:
            return best_η, best_gof
        else:
            return best_η
            return best_η

### --- Exact dynamic-programming GoF search --- ###

def zero_k_penalty(K):
    """Return the zero partition penalty for integer segment count K."""
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    return 0.0


def manuscript_log_binomial_k_penalty(n, penalty_weight):
    """Build p(K) = penalty_weight * log binom(n - 1, K - 1)."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(penalty_weight):
        raise ValueError("penalty_weight must be finite")

    def penalty(K):
        if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
            raise ValueError("K must be an integer in 1, ..., n")
        # The issue specification uses K - 1 changepoints.  No manuscript
        # source is committed here, so this is the formula that can be audited.
        log_count = (
            scipy.special.gammaln(n)
            - scipy.special.gammaln(K)
            - scipy.special.gammaln(n - K + 1)
        )
        return float(penalty_weight * log_count)

    return penalty


def legacy_gamma_k_penalty(n, Lmax, Delta, gamma):
    """Build the K-only penalty used by the live brute-force implementation."""
    for name, value in (("n", n), ("Lmax", Lmax), ("Delta", Delta)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")

    def penalty(K):
        if not isinstance(K, (int, np.integer)) or not 1 <= K <= Lmax:
            raise ValueError("K must be an integer in 1, ..., Lmax")
        if K == 1:
            return 0.0
        changepoints = K - 1
        effective_positions = (n - 1) - (changepoints - 1) * (Delta - 1)
        if effective_positions < changepoints:
            raise ValueError("K and Delta define an infeasible partition")
        log_configurations = (
            scipy.special.gammaln(effective_positions + 1)
            - scipy.special.gammaln(changepoints + 1)
            - scipy.special.gammaln(effective_positions - changepoints + 1)
        )
        # This matches gamma * log(Lmax * num_total_configs) in the live code.
        return float(gamma * (np.log(Lmax) + log_configurations))

    return penalty


def table_k_penalty(values):
    """Build a finite K-only penalty from a small mapping or sequence."""
    if hasattr(values, "items"):
        table = {int(K): float(value) for K, value in values.items()}
    else:
        table = {K: float(value) for K, value in enumerate(values, start=1)}
    if not table or any(K < 1 or not np.isfinite(value) for K, value in table.items()):
        raise ValueError("penalty table must contain finite values for positive K")

    def penalty(K):
        if K not in table:
            raise ValueError(f"penalty table has no value for K={K}")
        return table[K]

    return penalty


def _dp_loss_prefixes(M, Z, y, n, L):
    """Evaluate observation losses once and return column-wise prefix sums."""
    Z_array = np.asarray(Z)
    y_array = np.asarray(y)
    if Z_array.ndim == 1 and L == 1:
        Z_array = Z_array.reshape(n, 1)
    if Z_array.ndim != 2 or Z_array.shape[0] != n or Z_array.shape[1] < L:
        raise ValueError("Z must have shape (n, at least L)")
    if y_array.ndim != 1 or y_array.shape[0] != n:
        raise ValueError("y must have shape (n,)")

    losses = np.empty((n, L), dtype=float)
    for column in range(L):
        column_losses = np.asarray(M(Z_array[:, column], y_array), dtype=float)
        if column_losses.shape != (n,):
            raise ValueError("M must return one loss per observation")
        losses[:, column] = column_losses
    if not np.all(np.isfinite(losses)):
        raise ValueError("M produced a non-finite loss")
    return np.vstack([np.zeros((1, L)), np.cumsum(losses, axis=0)])


def chgpt_gof_dp(
    M,
    Z,
    y,
    n,
    L,
    Delta,
    *,
    L_is_upper_bound=False,
    penalty=None,
):
    """Exactly minimize the current partition objective by dynamic programming.

    This search matches the brute-force feasible set: endpoints have no Delta
    restriction, adjacent
    internal changepoints are at least Delta apart, and K uses ``Z[:, :K]``.
    A penalty is either a callable receiving only integer K or a finite table.

    Ties are deterministic: candidates are visited in increasing boundary order
    and the first equal minimum is retained.

    The recurrence uses O(L n^2) time and O(L n) memory.
    """
    for name, value in (("n", n), ("L", L), ("Delta", Delta)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if L > n:
        raise ValueError("L cannot exceed n")

    if penalty is None:
        penalty_function = zero_k_penalty
    elif callable(penalty):
        penalty_function = penalty
    elif hasattr(penalty, "__iter__"):
        penalty_function = table_k_penalty(penalty)
    else:
        raise TypeError("penalty must be a K-only callable or finite table")

    prefixes = _dp_loss_prefixes(M, Z, y, n, L)

    def segment_loss(start, end, column):
        return prefixes[end, column] - prefixes[start, column]

    # dp[k, t] is the best raw loss for observations 0, ..., t-1 using k
    # segments when t will be an internal boundary.  Keeping t as a boundary
    # state preserves the live code's lack of endpoint spacing.
    dp = np.full((L, n), np.inf)
    backpointer = np.full((L, n), -1, dtype=int)
    if L > 1:
        for t in range(1, n):
            dp[1, t] = segment_loss(0, t, 0)
        for j in range(2, L):
            for t in range(1, n):
                best = np.inf
                best_previous = -1
                for previous in range(1, t - Delta + 1):
                    candidate = dp[j - 1, previous] + segment_loss(
                        previous, t, j - 1
                    )
                    if candidate < best:
                        best = candidate
                        best_previous = previous
                dp[j, t] = best
                backpointer[j, t] = best_previous

    raw_by_K = {}
    changepoints_by_K = {}
    raw_by_K[1] = float(segment_loss(0, n, 0))
    changepoints_by_K[1] = np.array([], dtype=int)

    for K in range(2, L + 1):
        j = K - 1
        best = np.inf
        last_boundary = -1
        for t in range(1, n):
            candidate = dp[j, t] + segment_loss(t, n, K - 1)
            if candidate < best:
                best = candidate
                last_boundary = t
        if not np.isfinite(best):
            raw_by_K[K] = np.inf
            changepoints_by_K[K] = np.array([], dtype=int)
            continue

        changepoints = [last_boundary]
        boundary = last_boundary
        for current_j in range(j, 1, -1):
            boundary = backpointer[current_j, boundary]
            changepoints.append(boundary)
        changepoints.reverse()
        raw_by_K[K] = float(best)
        changepoints_by_K[K] = np.asarray(changepoints, dtype=int)

    penalty_by_K = {}
    total_by_K = {}
    for K in range(1, L + 1):
        if not np.isfinite(raw_by_K[K]):
            penalty_by_K[K] = np.nan
            total_by_K[K] = np.inf
            continue
        try:
            value = float(penalty_function(int(K)))
        except TypeError as error:
            raise TypeError(
                "partition penalty must be callable as penalty(K) with integer K only"
            ) from error
        if not np.isfinite(value):
            raise ValueError("partition penalties must be finite")
        penalty_by_K[K] = value
        total_by_K[K] = raw_by_K[K] + value

    candidate_K = range(1, L + 1) if L_is_upper_bound else (L,)
    feasible_K = [K for K in candidate_K if np.isfinite(total_by_K[K])]
    if not feasible_K:
        raise ValueError("K and Delta define an infeasible partition")
    selected_K = min(feasible_K, key=lambda K: (total_by_K[K], K))

    return {
        "changepoints": changepoints_by_K[selected_K].copy(),
        "K": selected_K,
        "raw_objective": raw_by_K[selected_K],
        "penalty": penalty_by_K[selected_K],
        "total_objective": total_by_K[selected_K],
        "raw_by_K": raw_by_K,
        "penalty_by_K": penalty_by_K,
        "total_by_K": total_by_K,
        "changepoints_by_K": changepoints_by_K,
    }

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
# @partial(jax.jit, static_argnames=['M'])
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

def _chgpt_gof_bf_mem_eff_exact_L(M, Z, y, n, L, Delta, γ=1.0, batch_size=4096, gof_opt_output=False, Lmax = None):
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
            # Assert that Z is either a vector or a matrix with one column
            if Z.ndim == 1:
                Z = Z.reshape((n, 1))
            elif Z.ndim > 2 or Z.shape[1] != 1:
                raise ValueError("Z must be a vector or a matrix with one column.")
            gof = M(Z[:, 0], y).sum()
            return np.array([], dtype=int), gof
        else: return np.array([], dtype=int)
        

    if Lmax is None:
        Lmax = L
        
    num_positions = n - 1
    effective_positions = num_positions - (k - 1) * (Delta - 1)
    
    if effective_positions < k:
        raise ValueError(f"Impossible to place {k} change points with separation {Delta} in {n} points.")

    num_total_configs = scipy.special.comb(effective_positions, k, exact=True)
    print(f"Starting brute-force search over {num_total_configs:,} configurations, with L = {L} and k = {k} change points.")
    
    # Check if the search is feasible
    if num_total_configs > 100_000_000:
        print("\n--- WARNING ---")
        print(f"The number of configurations is extremely large. This search may take a very long time.")
        print("Consider using the binary search version or reducing the problem complexity.\n")

    log_prior_penalty = jnp.log(Lmax * num_total_configs)
    
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
        # Assert batch_best_gof is a scalar
        assert batch_best_gof.ndim == 0, "Batch best GoF should be a scalar"

        if batch_best_gof < best_gof:
            best_gof = batch_best_gof
            # Store the best configuration (as a NumPy array)
            best_η = η_batch_np[batch_best_idx]
            
    print(f"Search complete. Optimal GoF score: {best_gof:.4f}")
    if gof_opt_output:
        return best_η, best_gof
    else:
        return best_η if best_gof < jnp.inf else np.array([])

def chgpt_gof_bf_mem_eff(M, Z, y, n, L, Delta, γ=1.0, L_is_upper_bound = False, batch_size=4096, gof_opt_output=False):
    if not L_is_upper_bound:
        return _chgpt_gof_bf_mem_eff_exact_L(M, Z, y, n, L, Delta, γ, batch_size, gof_opt_output)
    else:
        # If L is an upper bound, we need to search for the best L
        best_η = np.array([], dtype=int)
        best_gof = jnp.inf

        # for L_current in range(1, L + 1):
        #     η_current, gof_current = _chgpt_gof_bf_mem_eff_exact_L(
        #         M, Z[:, :L_current], y, n, L_current, Delta, γ, batch_size, gof_opt_output=True
        #     )
        #     if gof_current < best_gof:
        #         best_gof = gof_current
        #         best_η = η_current

        # Do it by filling a list of chgpts and gofs
        η_list = []
        gof_list = []
        for L_current in tqdm(range(1, L + 1), desc="Searching for best L"):
            η_current, gof_current = _chgpt_gof_bf_mem_eff_exact_L(
                M, Z[:, :L_current], y, n, L_current, Delta, γ, batch_size, gof_opt_output=True, Lmax=L
            )
            # Assert gof_current is a scalar
            assert gof_current.ndim == 0, "GoF current should be a scalar"
            η_list.append(η_current)
            gof_list.append(gof_current)
        print("GOF List:", gof_list)
        gof_list = np.array(gof_list)
        # Print the gof_list and the chgpt_list
        print("L_is_upper_bd. GoF List:", gof_list)
        print("L_is_upper_bd. Chgpt List:", η_list)
        i_opt = gof_list.argmin()
        best_η = η_list[i_opt]
        best_gof = gof_list[i_opt]

        if gof_opt_output:
            return best_η, best_gof
        else:
            return best_η
        

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