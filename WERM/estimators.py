import WERM.gof
import WERM.marginals
import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
import scipy.special
from functools import partial
from tqdm.auto import tqdm
from tqdm import trange

def chgpt_two_fold_cv(f, X, y, gamma_candidates, M_scorer, P_scorer = None):
    """
    Performs a 2-fold cross-validation using an even/odd index split to find the
    best hyperparameter for a changepoint detection function.

    The scoring method is based on Section A.1 of "Divide and Conquer Dynamic
    Programming..." by Li, Wang, and Rinaldo. For each fold, it:
    1. Finds changepoints on the training set (e.g., even indices).
    2. Uses these changepoints to segment the test set (e.g., odd indices).
    3. Fits a separate model within each test set segment by minimizing an
       empirical risk.
    4. Sums the minimized risk values across all segments to get the final score
       for the fold. A lower score is better.

    Args:
        f (callable): A black-box function `f(X, y, gamma)` that accepts
            training data and a hyperparameter, and returns an array of
            changepoint indices.
        X (np.ndarray): The full feature data (n_samples, n_features).
        y (np.ndarray): The full target data (n_samples,).
        gamma_candidates (list or np.ndarray): A list of gamma values to test.
        M_scorer (callable): The loss function part of the empirical risk,
            e.g., `M(y_pred, y_true)`.
        P_scorer (callable): The penalty function part of the empirical risk,
            e.g., `P(beta)`.

    Returns:
        float: The hyperparameter `gamma` that achieved the lowest mean score
               across the two folds.
    """
    n_samples, p = X.shape
    even_indices = np.arange(0, n_samples, 2)
    odd_indices = np.arange(1, n_samples, 2)

    if len(even_indices) == 0 or len(odd_indices) == 0:
        raise ValueError("Cannot create two folds. Input data is too small.")

    mean_scores = []

    # JIT-compiled function to calculate empirical risk for a given segment
    @partial(jax.jit, static_argnums=(0, 1))
    def empirical_risk(M_func, P_func, beta, X_data, y_data):
        # Ensure predictions and targets have compatible shapes for M_func
        y_pred = X_data @ beta
        # The risk is the sum of the loss and the model complexity penalty
        if P_func is None:
            # If no penalty function is provided, just return the loss
            return jnp.sum(M_func(y_pred, y_data))
        return jnp.sum(M_func(y_pred, y_data)) + jnp.sum(P_func(beta))

    # Helper function to compute the total risk for a test fold
    def _calculate_fold_score(chgpts, X_test, y_test):
        total_risk = 0.0
        # Segment the test data based on the provided changepoints
        # np.split will create segments from indices 0..n-1 at the chgpt locations
        test_segments = np.split(np.arange(len(X_test)), chgpts)

        for segment_indices in test_segments:
            if len(segment_indices) == 0:
                continue # Skip empty segments

            X_seg, y_seg = X_test[segment_indices], y_test[segment_indices]

            # Define the objective function for this segment
            objective_fn = lambda beta: empirical_risk(M_scorer, P_scorer, beta, X_seg, y_seg)
            
            # Minimize the empirical risk for this segment
            res = scipy.optimize.minimize(
                fun=objective_fn,
                x0=0.1*jnp.ones(p, dtype=jnp.float64), # Initial guess for beta
                jac=jax.grad(objective_fn),
                method='L-BFGS-B'
            )
            
            # Add the minimized risk of this segment to the total
            total_risk += res.fun
        
        return total_risk

    print(f"Running 2-fold CV for {len(gamma_candidates)} gamma candidates...")
    for gamma in tqdm(gamma_candidates, desc="CV Progress"):
        # --- Fold 1: Train on even indices, test on odd indices ---
        chgpt_estimate_1 = f(X[even_indices], y[even_indices], gamma)
        # The logic of the even/odd split implies that changepoints found on one
        # set can be used directly to partition the other set.
        score_1 = _calculate_fold_score(chgpt_estimate_1, X[odd_indices], y[odd_indices])

        # --- Fold 2: Train on odd indices, test on even indices ---
        chgpt_estimate_2 = f(X[odd_indices], y[odd_indices], gamma)
        score_2 = _calculate_fold_score(chgpt_estimate_2, X[even_indices], y[even_indices])
        
        # The final score for this gamma is the average of the two fold scores
        mean_scores.append(np.mean([score_1, score_2]))

    # Find the best gamma corresponding to the minimum score (lower risk is better)
    print("mean_scores:", mean_scores)
    best_idx = np.argmin(mean_scores)
    best_gamma = gamma_candidates[best_idx]
    
    print(f"CV finished. Best gamma found: {best_gamma:.4f} (score: {mean_scores[best_idx]:.4f})")
    
    return best_gamma

def weighted_erm(X, y, M, L, Delta_n, gamma, L_is_upper_bound=False, brute_force=False, return_gof=False, return_B_hat=False, **kwargs):
    """
    Computes the change point estimate using the Weighted ERM method for a single L.
    (This function was provided as complete and is used as a helper).
    """
    n, p = X.shape
    # Assuming WERM.marginals.compute_pi is a valid function in your library
    pi = WERM.marginals.compute_pi(
        L=L, n=n, Delta=Delta_n, L_is_upper_bound=L_is_upper_bound
    )
    L = pi.shape[0]
    B_hat = np.zeros((p, L))

    # --- 1. Estimate signal coefficients B_hat ---
    for l_ in range(L):
        sample_weight = pi[l_, :] / np.sum(pi[l_, :])
        
        @partial(jax.jit, static_argnums=(0,))
        def empirical_risk(M_func, beta, X_data, y_data, weights):
            return jnp.multiply(M_func(X_data @ beta, y_data), weights).sum()

        res = scipy.optimize.minimize(
            lambda beta: empirical_risk(M, beta, X, y, sample_weight),
            0.1*np.ones(p),
            jac=jax.grad(lambda beta: empirical_risk(M, beta, X, y, sample_weight)),
            method='L-BFGS-B'
        )
        B_hat[:, l_] = res.x

    # --- 2. Compute Change Point Estimates using Goodness-of-Fit ---
    Theta_hat = X @ B_hat

    # The original gof functions might not have a gof_opt_output argument
    # I'm assuming they can be modified to return both chgpts and gof_score
    if brute_force:
        # Assuming brute force can also return gof
        opt_chgpts, opt_gof = WERM.gof.chgpt_gof_brute_force_mem_eff(M, Theta_hat, y, n, L, Delta_n, γ=gamma, batch_size=4096, gof_opt_output=True)
    else:
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_binary_seg_mem_eff(M, Theta_hat, y, n, L, Delta_n, γ=gamma, min_window_size=30, gof_opt_output=True)
        opt_chgpts, opt_gof = WERM.gof.chgpt_binary_segmentation(M, Theta_hat, y, n, L, min_window_size=Delta_n, gamma=gamma, gof_opt_output=True, γ=gamma)

    if return_gof:
        return (opt_chgpts, opt_gof, B_hat) if return_B_hat else (opt_chgpts, opt_gof)
    return (opt_chgpts, B_hat) if return_B_hat else opt_chgpts

def weighted_ermCV(X, y, M, L, Delta_n, gamma_candidates=None, L_is_upper_bound = True, brute_force=False, **kwargs):
    """
    Run cross-validation to select the best gamma for WERM, and return
    the best change point configuration.
    """
    
    f = lambda X_train, y_train, gamma: weighted_erm(
        X_train, y_train, M, L=L, Delta_n=Delta_n, gamma=gamma,
        L_is_upper_bound=L_is_upper_bound, brute_force=brute_force, return_gof=False, return_B_hat=False, **kwargs
    )

    return chgpt_two_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=gamma_candidates if gamma_candidates is not None else np.logspace(-1, 1, 10),
        M_scorer=M,
        P_scorer=None
    )

def weighted_erm_extended_penalized(X, y, M, L, Delta_n, gamma, brute_force=False, **kwargs):
    """
    Computes the optimal change point configuration by searching over L_tilde <= L
    and selecting the best based on a penalized goodness-of-fit score.
    """
    n = X.shape[0]
    candidate_chgpts = []
    candidate_scores = []

    # Iterate from L_tilde = 1 up to the max L provided
    for L_tilde in tqdm(range(1, L + 1), desc="Penalized Search over L"):
        # Get the unpenalized GOF score by setting gamma=0 for the search
        chgpts, gof_score = weighted_erm(
            X, y, M, L=L_tilde, Delta_n=Delta_n, gamma=0.0,
            L_is_upper_bound=False, brute_force=brute_force, return_gof=True, **kwargs
        )

        if gof_score is None: continue

        # Calculate the penalty for this configuration
        s = len(chgpts[chgpts != -1])
        
        # Penalty term: gamma * log(C(n-1, s))
        if s >= n:
            log_penalty_term = np.inf
        else:
            log_penalty_term = np.log(scipy.special.comb(n - 1, s, exact=False) + 1e-9)

        penalized_score = gof_score + gamma * log_penalty_term
        
        candidate_chgpts.append(chgpts)
        candidate_scores.append(penalized_score)

    # Return the change point configuration with the minimum penalized score
    if not candidate_scores:
        return np.array([], dtype=int)
    
    best_idx = np.argmin(candidate_scores)
    return candidate_chgpts[best_idx]

def weighted_erm_extended_penalizedCV(X, y, M, L, Delta_n, gamma_candidates=None, brute_force=False, **kwargs):
    """
    Run cross-validation to select the best gamma for penalized WERM,
    and return the best change point configuration.
    """
    f = lambda X_train, y_train, gamma: weighted_erm_extended_penalized(
        X_train, y_train, M, L=L, Delta_n=Delta_n, gamma=gamma, brute_force=brute_force, **kwargs
    )

    return chgpt_two_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=gamma_candidates if gamma_candidates is not None else np.logspace(-1, 1, 10),
        M_scorer=M,
        P_scorer=None
    )

def weighted_erm_extended_elbow(X, y, M, Delta_n, L_max=7, brute_force=False, **kwargs):
    """
    Finds the optimal change points by automatically searching for an "elbow".

    This function iteratively increases the number of signals L, starting from 1.
    After each step, it checks the solution path so far to see if an elbow
    point has been reached, defined by the "decreasing average slope" method.
    The search stops as soon as an elbow is detected or when L_max is reached.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The response vector.
        M (callable): The loss function.
        Delta_n (int): The minimum distance between change points.
        L_max (int, optional): A safeguard to stop the search if no elbow is
            detected. Defaults to 10.
        brute_force (bool, optional): Whether to use brute-force search in the
            underlying W-ERM calls. Defaults to False.
        **kwargs: Additional arguments passed to the weighted_erm function.

    Returns:
        np.ndarray: The array of optimal change point locations found at the elbow.
    """
    solution_path = []
    
    # Use trange for a progress bar that can be manually updated
    pbar = trange(1, L_max + 1, desc="Searching for elbow...")

    for L_tilde in pbar:
        # Update the progress bar description
        pbar.set_description(f"Testing L={L_tilde}")
        
        # 1. Compute the W-ERM result for the current L
        chgpts, gof_score = weighted_erm(
            X, y, M, L=L_tilde, Delta_n=Delta_n, gamma=0.0,
            L_is_upper_bound=False, brute_force=brute_force, return_gof=True, **kwargs
        )

        if gof_score is not None:
            num_chgpts = len(chgpts[chgpts != -1])
            solution_path.append({
                'L_val': L_tilde,
                'num_chgpts': num_chgpts,
                'gof': gof_score,
                'chgpts': chgpts
            })

        # 2. Filter for unique solutions (based on number of changepoints)
        unique_solutions = []
        seen_chgpts_count = set()
        # Sort by number of changepoints to create a proper path
        for sol in sorted(solution_path, key=lambda x: x['num_chgpts']):
            if sol['num_chgpts'] not in seen_chgpts_count:
                unique_solutions.append(sol)
                seen_chgpts_count.add(sol['num_chgpts'])
        
        # 3. Check for an elbow if we have enough points
        if len(unique_solutions) < 3:
            # Not enough points to calculate slopes, continue to the next L
            continue

        # --- Implement the "decreasing average slope" check ---
        points = np.array([[s['num_chgpts'], s['gof']] for s in unique_solutions])
        avg_slopes = []

        # Calculate average slope at each interior point
        for i in range(1, len(points) - 1):
            p_prev, p_curr, p_next = points[i-1], points[i], points[i+1]
            dx1 = p_curr[0] - p_prev[0]
            dx2 = p_next[0] - p_curr[0]
            slope1 = (p_curr[1] - p_prev[1]) / dx1 if dx1 != 0 else np.inf
            slope2 = (p_next[1] - p_curr[1]) / dx2 if dx2 != 0 else np.inf
            avg_slopes.append((slope1 + slope2) / 2.0)
        
        # Check if the last calculated average slope is smaller in magnitude
        if len(avg_slopes) > 1:
            if abs(avg_slopes[-1]) < abs(avg_slopes[-2]):
                # Elbow detected! The elbow point corresponds to the last point
                # used to calculate the newest slope.
                elbow_solution = unique_solutions[len(avg_slopes)] # Index is i+1
                print(f"\nElbow detected at L={elbow_solution['L_val']} (num_chgpts={elbow_solution['num_chgpts']}). Stopping search.")
                pbar.close()
                return elbow_solution['chgpts']

    # 4. If loop finishes (L_max reached) without finding an elbow
    pbar.close()
    if not unique_solutions:
        print("\nWarning: No valid solutions found.")
        return np.array([], dtype=int)
        
    final_solution = unique_solutions[-1]
    print(f"\nL_max reached. No elbow detected. Returning solution for L={final_solution['L_val']} (num_chgpts={final_solution['num_chgpts']}).")
    return final_solution['chgpts']