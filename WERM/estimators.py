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
from joblib import Parallel, delayed

def chgpt_k_fold_cv(f, X, y, gamma_candidates, M_scorer, P_scorer=None, k=5):
    """
    Performs k-fold cross-validation using index-based splits to find the
    best hyperparameter for a changepoint detection function.

    The scoring method is based on Section A.1 of "Divide and Conquer Dynamic
    Programming..." by Li, Wang, and Rinaldo. For each fold, it:
    1. Finds changepoints on the training set (k-1 folds).
    2. Uses these changepoints to segment the test set (1 fold).
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
        k (int): Number of folds for cross-validation. Default is 5.

    Returns:
        tuple: (best_gamma, best_chgpt_est) where best_gamma is the hyperparameter
               that achieved the lowest mean score across all folds, and 
               best_chgpt_est is the changepoint estimate using best_gamma on full data.
    """
    n_samples, p = X.shape
    
    # Create k folds using index-based splits
    folds = []
    for fold_idx in range(k):
        fold_indices = np.arange(fold_idx, n_samples, k)
        if len(fold_indices) == 0:
            raise ValueError(f"Fold {fold_idx} is empty. Consider reducing k or increasing data size.")
        folds.append(fold_indices)
    
    # Check that all folds have data
    if any(len(fold) == 0 for fold in folds):
        raise ValueError(f"Cannot create {k} non-empty folds. Input data is too small.")

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

    print(f"Running {k}-fold CV for {len(gamma_candidates)} gamma candidates...")
    for gamma in tqdm(gamma_candidates, desc="CV Progress"):
        fold_scores = []
        
        for test_fold_idx in range(k):
            # Get test fold indices
            test_indices = folds[test_fold_idx]
            
            # Get training fold indices (all other folds combined)
            train_indices = np.concatenate([folds[i] for i in range(k) if i != test_fold_idx])
            
            # Sort indices to maintain temporal order
            train_indices = np.sort(train_indices)
            test_indices = np.sort(test_indices)
            
            # Train: Find changepoints on training data
            chgpt_estimate = f(X[train_indices], y[train_indices], gamma)
            
            # Test: Evaluate on test data using the found changepoints
            fold_score = _calculate_fold_score(chgpt_estimate, X[test_indices], y[test_indices])
            fold_scores.append(fold_score)
        
        # The final score for this gamma is the average across all folds
        mean_scores.append(np.mean(fold_scores))

    # Find the best gamma corresponding to the minimum score (lower risk is better)
    print("mean_scores:", mean_scores)
    best_idx = np.argmin(mean_scores)
    best_gamma = gamma_candidates[best_idx]
    
    print(f"CV finished. Best gamma found: {best_gamma:.4f} (score: {mean_scores[best_idx]:.4f})")

    # Fit the final model using the best gamma on the full dataset
    best_chgpt_est = f(X, y, best_gamma)
    
    return best_gamma, best_chgpt_est

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

    # Fit the final model using the best gamma on the full dataset
    # This is optional, depending on whether you want to return the best model
    best_chgpt_est = f(X, y, best_gamma)
    
    return best_gamma, best_chgpt_est

def weighted_erm(X, y, M, L, Delta_n, gamma, L_is_upper_bound=False, brute_force=False, return_gof=False, return_B_hat=False, **kwargs):
    """
    Computes the change point estimate using the Weighted ERM method for a single L.
    (This function was provided as complete and is used as a helper).
    """
    n, p = X.shape

    n_small = 50
    Delta_n_small = int(Delta_n * n_small/n)
    if L_is_upper_bound: 
        p_l = 1/L * np.ones(L) # puts equal weight on all number of signals, set None otherwise. Best for estimating size. 
        Lmin = 1
        Lmax = L
    else:
        p_l = None
        Lmin = L
        Lmax = L
    pi = WERM.config.extrapolated_marginals(Lmin, Lmax, Delta_n_small, n_small = n_small, n_large = n, p_l = p_l, verbose = False)

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

    if brute_force:
        opt_chgpts, opt_gof = WERM.gof.chgpt_gof_bf_mem_eff(
                M, Theta_hat, y, n, L, Delta_n, L_is_upper_bound=L_is_upper_bound,
                γ=gamma, batch_size=4096, gof_opt_output=True
            )
    else:
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_binary_seg_mem_eff(M, Theta_hat, y, n, L, Delta_n, γ=gamma, min_window_size=30, gof_opt_output=True)
        # opt_chgpts, opt_gof = WERM.gof.chgpt_binary_segmentation(M, Theta_hat, y, n, L, min_window_size=Delta_n, gamma=gamma, gof_opt_output=True, γ=gamma)
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_dc(M, Theta_hat, y, n, L, Delta_n, γ=1.0, L_is_upper_bound=L_is_upper_bound, gof_opt_output=True)
        opt_chgpts, opt_gof = WERM.gof.chgpt_gof_divide_conquer(M, Theta_hat, y, n, L, Delta_n, γ=gamma, L_is_upper_bound=L_is_upper_bound, gof_opt_output=True)

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

    best_gamma, best_model_output = chgpt_two_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=gamma_candidates if gamma_candidates is not None else np.logspace(-1, 1, 10),
        M_scorer=M,
        P_scorer=None
    )

    return best_gamma, best_model_output

def weighted_erm_extended(X, y, M, L, Delta_n, gamma, brute_force=False, **kwargs):
    """
    Computes the optimal change point configuration by searching over L_tilde <= L
    and selecting the best based on a penalized goodness-of-fit score.
    """

    n = X.shape[0]
    candidate_chgpts = []
    candidate_scores = []
    fit_scores = []
    penalty_scores = []

    # Iterate from L_tilde = 1 up to the max L provided
    for L_tilde in tqdm(range(1, L + 1), desc="Penalized Search over L"):
        # Get the unpenalized GOF score by setting gamma=0 for the search
        chgpts, fit_score = weighted_erm(
            X, y, M, L=L_tilde, Delta_n=Delta_n, gamma=0.0,
            L_is_upper_bound=False, brute_force=brute_force, return_gof=True, **kwargs
        )

        if fit_score is None: continue

        # Calculate the penalty for this configuration
        s = len(chgpts[chgpts != -1])
        
        # Penalty term: gamma * log(C(n-1, s))
        if s >= n:
            log_penalty_term = np.inf
        else:
            log_penalty_term = np.log(L * scipy.special.comb(n - 1, s, exact=False) + 1e-9) # Correct

        # Print the scale of the penalty term vs. the GOF score
        if fit_score > 0:
            print(f"fit score: {fit_score:.4f}, gamma * log-penalty term: {gamma * log_penalty_term:.4f} (gamma={gamma})")
            print(f"Change points: {chgpts}, Number of change points: {s}, L_tilde: {L_tilde}, n: {n}")
        penalized_score = fit_score + gamma * log_penalty_term
        
        candidate_chgpts.append(chgpts)
        candidate_scores.append(penalized_score)
        fit_scores.append(fit_score)
        penalty_scores.append(gamma * log_penalty_term)

    # Return the change point configuration with the minimum penalized score
    if not candidate_scores:
        return np.array([], dtype=int)
    
    return candidate_chgpts, fit_scores, penalty_scores

def weighted_erm_extended_penalized(X, y, M, L, Delta_n, gamma, brute_force=False, gof_output = False, **kwargs):
    """
    Computes the optimal change point configuration by searching over L_tilde <= L
    and selecting the best based on a penalized goodness-of-fit score.
    """

    candidate_chgpts, fit_scores, penalty_scores = weighted_erm_extended(
        X, y, M, L=L, Delta_n=Delta_n, gamma=gamma, brute_force=brute_force, **kwargs
    )
    candidate_scores = np.array(fit_scores) + gamma * np.array(penalty_scores)
    best_idx = np.argmin(candidate_scores)

    if gof_output:
        return candidate_chgpts[best_idx], candidate_scores
    return candidate_chgpts[best_idx]

def weighted_erm_extended_penalizedCV(X, y, M, L, Delta_n, gamma_candidates=None, brute_force=False, M_scorer = None, P_scorer = None, num_folds = 5, **kwargs):
    """
    Run cross-validation to select the best gamma for penalized WERM,
    and return the best change point configuration.
    """
    f = lambda X_train, y_train, gamma: weighted_erm_extended_penalized(
        X_train, y_train, M, L=L, Delta_n=Delta_n, gamma=gamma, brute_force=brute_force, **kwargs
    )



    best_gamma, best_chgpt_est = chgpt_k_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=gamma_candidates if gamma_candidates is not None else np.logspace(-1, 1, 10),
        M_scorer=M_scorer,
        P_scorer=P_scorer,
        k=num_folds  # Using 5-fold CV for better stability
    )

    print(f"Best gamma found: {best_gamma:.4f}")
    return best_gamma, best_chgpt_est

def weighted_erm_extended_elbow(X, y, M, Delta_n, Lmax = 7, brute_force=False, **kwargs):
    
    candidate_chgpts, fit_scores, penalty_scores = weighted_erm_extended(
        X, y, M, L=Lmax, Delta_n=Delta_n, gamma=0, brute_force=brute_force, **kwargs
    )
    candidate_scores = np.array(fit_scores)

    # To complete: find the elbow point in the candidate_scores
        # Sort solutions by number of change points (ascending order)
    num_chgpts = [len(chgpts[chgpts != -1]) for chgpts in candidate_chgpts]
    sorted_indices = np.argsort(num_chgpts)
    
    sorted_scores = candidate_scores[sorted_indices]
    sorted_chgpts = [candidate_chgpts[i] for i in sorted_indices]
    sorted_num_chgpts = [num_chgpts[i] for i in sorted_indices]
    
    # Need at least 3 points to compute elbow
    if len(sorted_scores) < 3:
        return sorted_chgpts[0] if len(sorted_chgpts) > 0 else np.array([], dtype=int)
    
    # Compute average slopes at each point
    average_slopes = []
    
    for i in range(1, len(sorted_scores) - 1):
        # Left slope: from point i-1 to point i
        left_slope = (sorted_scores[i] - sorted_scores[i-1]) / max(1, sorted_num_chgpts[i] - sorted_num_chgpts[i-1])
        
        # Right slope: from point i to point i+1
        right_slope = (sorted_scores[i+1] - sorted_scores[i]) / max(1, sorted_num_chgpts[i+1] - sorted_num_chgpts[i])
        
        # Average of the two slopes
        avg_slope = (left_slope + right_slope) / 2.0
        average_slopes.append(abs(avg_slope))
    
    # Find elbow point: where average slope starts to decrease for the first time
    elbow_idx = 1  # Default to second point if no clear elbow is found
    
    for i in range(1, len(average_slopes)):
        if average_slopes[i] < average_slopes[i-1]:
            elbow_idx = i + 1  # +1 because average_slopes is offset by 1 from sorted_scores
            break
    
    return sorted_chgpts[elbow_idx]

def weighted_erm_extended_elbow_kneedle(X, y, M, Delta_n, L_max=7, brute_force=False, sensitivity=1.0, **kwargs):
    """
    Kneedle algorithm implementation - most widely used in practice.
    Based on "Finding a Kneedle in a Haystack" by Satopa et al.
    """
    candidate_chgpts, fit_scores, penalty_scores = weighted_erm_extended(
        X, y, M, L=L_max, Delta_n=Delta_n, gamma=0.0, brute_force=brute_force, **kwargs
    )
    candidate_scores = np.array(fit_scores)
    
    # Sort by number of change points
    num_chgpts = [len(chgpts[chgpts != -1]) for chgpts in candidate_chgpts]
    sorted_indices = np.argsort(num_chgpts)
    sorted_scores = candidate_scores[sorted_indices]
    sorted_chgpts = [candidate_chgpts[i] for i in sorted_indices]
    sorted_num_chgpts = [num_chgpts[i] for i in sorted_indices]
    
    if len(sorted_scores) < 3:
        return sorted_chgpts[0] if len(sorted_chgpts) > 0 else np.array([], dtype=int)
    
    # Normalize data to [0,1] range
    x_norm = np.array(sorted_num_chgpts, dtype=float)
    y_norm = np.array(sorted_scores, dtype=float)
    x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min()) if x_norm.max() > x_norm.min() else x_norm
    y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min()) if y_norm.max() > y_norm.min() else y_norm
    
    # Compute distances from line connecting first and last points
    distances = []
    for i in range(len(x_norm)):
        # Distance from point to line using cross product formula
        x1, y1 = x_norm[0], y_norm[0]
        x2, y2 = x_norm[-1], y_norm[-1]
        x0, y0 = x_norm[i], y_norm[i]
        
        # Line equation: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
        distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances.append(distance)
    
    # Apply sensitivity threshold - only consider points above mean + sensitivity*std
    distances = np.array(distances)
    threshold = np.mean(distances) + sensitivity * np.std(distances)
    candidates = np.where(distances >= threshold)[0]
    
    if len(candidates) == 0:
        knee_idx = np.argmax(distances)
    else:
        knee_idx = candidates[np.argmax(distances[candidates])]
    
    return sorted_chgpts[knee_idx]

