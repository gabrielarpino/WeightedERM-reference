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

def chgpt_k_fold_cv(f, X, y, gamma_candidates, M_scorer, P_scorer=None, k=5, n_jobs=4):
    """
    K-fold cross-validation following the paper's structure but adapted for regression
    with empirical risk minimization.
    
    Args:
        f (callable): A function f(X, y, gamma) that returns changepoint indices
        X (np.ndarray): Feature data (n_samples, n_features)
        y (np.ndarray): Target data (n_samples,)
        gamma_candidates (list): Hyperparameter values to test
        M_scorer (callable): Loss function M(y_pred, y_true)
        P_scorer (callable): Penalty function P(beta), optional
        k (int): Number of folds (V in the paper)
        n_jobs (int): Number of parallel jobs
    
    Returns:
        tuple: (best_gamma, best_chgpt_est)

    P_scorer implementation is not tested!
    """
    n_samples, p = X.shape
    
    # Create folds following paper's definition: F_v = {v + i·V : v + i·V ≤ n}
    # Adapted to 0-based indexing
    folds = []
    for v in range(k):
        fold_indices = []
        i = 0
        while True:
            idx = v + i * k
            if idx >= n_samples:
                break
            fold_indices.append(idx)
            i += 1
        folds.append(np.array(fold_indices))
    
    # Ensure no empty folds
    if any(len(fold) == 0 for fold in folds):
        raise ValueError(f"Cannot create {k} non-empty folds with {n_samples} samples")
    
    # JIT-compiled empirical risk function
    @partial(jax.jit, static_argnums=(0, 1))
    def empirical_risk(M_func, P_func, beta, X_data, y_data):
        y_pred = X_data @ beta
        if P_func is None:
            return jnp.sum(M_func(y_pred.flatten(), y_data.flatten()))
        return jnp.sum(M_func(y_pred.flatten(), y_data.flatten())) + jnp.sum(P_func(beta))

    def evaluate_single_gamma(gamma):
        """Evaluate one gamma value using k-fold CV"""
        cv_score = 0.0
        
        for v in range(k):
            # Get test fold F_v
            test_indices = folds[v]
            
            # Get training indices F_v^c (complement of F_v)
            train_indices = np.concatenate([folds[i] for i in range(k) if i != v])
            train_indices = np.sort(train_indices)  # Maintain temporal order
            # assert np.all(train_indices == train_indices_ordered), "Training indices must be in temporal order"
            
            # Step 1: Apply changepoint detection on training data F_v^c
            train_chgpts = f(X[train_indices], y[train_indices], gamma)
            
            # Step 2: Map changepoints back to original dataset indices
            mapped_chgpts = []
            for chgpt in train_chgpts:
                if 0 < chgpt < len(train_indices):  # Exclude boundaries
                    mapped_chgpts.append(train_indices[chgpt])
            mapped_chgpts = np.array(mapped_chgpts)
            
            # Step 3: Estimate β̂^{-F_v}_{ψ,k} for each segment using training data F_v^c
            # Add boundaries: 0 and n_samples
            all_chgpts = np.concatenate([[0], mapped_chgpts, [n_samples]])
            segment_betas = []
            
            for k_seg in range(len(all_chgpts) - 1):
                start_idx = all_chgpts[k_seg]
                end_idx = all_chgpts[k_seg + 1]
                
                # Find training indices in this segment
                segment_train_mask = (train_indices >= start_idx) & (train_indices < end_idx)
                segment_train_indices = train_indices[segment_train_mask]
                
                if len(segment_train_indices) > 0:
                    # Get training data for this segment
                    X_seg_train = X[segment_train_indices]
                    y_seg_train = y[segment_train_indices]
                    
                    # Minimize empirical risk on this training segment to get β̂
                    objective_fn = lambda beta: empirical_risk(M_scorer, P_scorer, beta, X_seg_train, y_seg_train)
                    
                    options = {'disp': False, 'maxiter': 1000}  # Options for L-BFGS-B
                    res = scipy.optimize.minimize(
                        fun=objective_fn,
                        x0=0.1 * jnp.ones(p, dtype=jnp.float64),
                        jac=jax.grad(objective_fn),
                        method='L-BFGS-B',
                        options=options
                    )
                    
                    segment_betas.append(res.x)
                else:
                    # If no training data in segment, use zero vector
                    segment_betas.append(np.zeros(p))
            
            # Step 4: Evaluate test data F_v using pre-computed changepoints and β̂ values
            fold_score = 0.0
            for i in test_indices:
                # Find which segment this test point belongs to
                segment_idx = np.searchsorted(all_chgpts[1:], i, side='right')
                segment_idx = min(segment_idx, len(segment_betas) - 1)
                
                # Get the β̂ for this segment (estimated from training data)
                beta_seg = segment_betas[segment_idx]
                
                # Compute prediction and add to empirical risk
                y_pred = X[i:i+1] @ beta_seg  # Single prediction
                assert M_scorer(y_pred, y[i:i+1]).shape == (1, )
                fold_score += float(M_scorer(y_pred, y[i:i+1])[0])  # Ensure M_scorer returns a scalar
                
                # Add penalty if provided
                # if P_scorer is not None:
                #     fold_score += float(P_scorer(beta_seg))

            # # Step 4: Evaluate test data F_v using pre-computed changepoints and β̂ values - vectorized - untested. 
            # segment_indices = np.searchsorted(all_chgpts[1:], test_indices, side='right')
            # segment_indices = np.minimum(segment_indices, len(segment_betas) - 1)

            # B_hat = np.array(segment_betas).reshape((p, len(segment_indices))) # (p, len(segment_indices))
            # assert B_hat.shape == (p, len(test_indices)), f"Expected B_hat shape {(p, len(test_indices))}, got {B_hat.shape}"

            # theta_pred = X[test_indices] @ B_hat  # (n_test, p)
            # assert theta_pred.shape == (len(test_indices), p), f"Expected theta_pred shape {(len(test_indices), p)}, got {theta_pred.shape}"

            # # Compute the loss for all test points
            # m_scores = [M_scorer(theta_pred[:, l], y[test_indices]).sum() for l in range(B_hat.shape[1])]
            # fold_score = jnp.sum(jnp.array(m_scores))  # Sum over all segments

            # # Add penalty if provided -- do not add penalty to goodness of fit score
            # # if P_scorer is not None:
            # #     fold_score += np.sum([P_scorer(beta_seg) for beta_seg in segment_betas])
            
            cv_score += fold_score
        
        return cv_score
    
    print(f"Running {k}-fold CV for {len(gamma_candidates)} gamma candidates using {n_jobs} cores...")
    
    # Parallel evaluation
    scores = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_single_gamma)(gamma) 
        for gamma in tqdm(gamma_candidates, desc="CV Progress")
    )
    
    # Find best gamma
    best_idx = np.argmin(scores)
    best_gamma = gamma_candidates[best_idx]
    
    δ = n_samples/p

    fmt = lambda g: "(" + ", ".join(f"{x:.4f}" for x in g) + ")" if isinstance(g, tuple) else f"{g:.4f}"
    print(f"CV finished for delta: {δ:.4f}. Best gamma: {fmt(best_gamma)} (score: {scores[best_idx]:.4f})\nCandidates: {[fmt(g) for g in gamma_candidates]} (Scores: {[f'{s:.4f}' for s in scores]})")
    # print(f"CV finished for delta: {δ:.4f}. Best gamma: {best_gamma:.4f} (score: {scores[best_idx]:.4f})")
    print(f"Candidate gammas: {gamma_candidates}, Scores: {scores}")
    
    # Fit final model on full dataset
    best_chgpt_est = f(X, y, best_gamma)
    
    return best_gamma, best_chgpt_est

def chgpt_two_fold_cv(f, X, y, gamma_candidates, M_scorer, P_scorer=None):
    """
    2-fold cross-validation following the paper's structure but adapted for regression.
    """
    n_samples, p = X.shape
    
    # Create folds following paper's definition (adapted to 0-based indexing)
    fold_1 = np.arange(0, n_samples, 2)  # Even indices: 0, 2, 4, ...
    fold_2 = np.arange(1, n_samples, 2)  # Odd indices: 1, 3, 5, ...
    
    if len(fold_1) == 0 or len(fold_2) == 0:
        raise ValueError("Cannot create two non-empty folds")
    
    # JIT-compiled empirical risk function
    @partial(jax.jit, static_argnums=(0, 1))
    def empirical_risk(M_func, P_func, beta, X_data, y_data):
        y_pred = X_data @ beta
        if P_func is None:
            return jnp.sum(M_func(y_pred, y_data))
        return jnp.sum(M_func(y_pred, y_data)) + jnp.sum(P_func(beta))
    
    def evaluate_single_gamma(gamma):
        cv_score = 0.0
        
        # Process both folds
        for fold_idx, (train_indices, test_indices) in enumerate([(fold_2, fold_1), (fold_1, fold_2)]):
            
            # Step 1: Get changepoints on training data
            train_chgpts = f(X[train_indices], y[train_indices], gamma)
            
            # Ensure changepoints are integers and convert to numpy array if needed
            if isinstance(train_chgpts, (list, tuple)):
                train_chgpts = np.array(train_chgpts, dtype=int)
            elif hasattr(train_chgpts, 'numpy'):  # JAX array
                train_chgpts = np.array(train_chgpts, dtype=int)
            else:
                train_chgpts = np.asarray(train_chgpts, dtype=int)
            
            # Step 2: Map to original indices
            mapped_chgpts = []
            for chgpt in train_chgpts:
                if 0 < chgpt < len(train_indices):
                    mapped_chgpts.append(train_indices[chgpt])
            mapped_chgpts = np.array(mapped_chgpts, dtype=int)
            
            # Step 3: Estimate β̂ for each segment using training data
            all_chgpts = np.concatenate([[0], mapped_chgpts, [n_samples]]).astype(int)
            segment_betas = []
            
            for k_seg in range(len(all_chgpts) - 1):
                start_idx = all_chgpts[k_seg]
                end_idx = all_chgpts[k_seg + 1]
                
                segment_train_mask = (train_indices >= start_idx) & (train_indices < end_idx)
                segment_train_indices = train_indices[segment_train_mask]
                
                if len(segment_train_indices) > 0:
                    X_seg_train = X[segment_train_indices]
                    y_seg_train = y[segment_train_indices]
                    
                    objective_fn = lambda beta: empirical_risk(M_scorer, P_scorer, beta, X_seg_train, y_seg_train)
                    
                    res = scipy.optimize.minimize(
                        fun=objective_fn,
                        x0=0.1 * jnp.ones(p, dtype=jnp.float64),
                        jac=jax.grad(objective_fn),
                        method='L-BFGS-B'
                    )
                    
                    segment_betas.append(res.x)
                else:
                    segment_betas.append(np.zeros(p))
            
            # Step 4: Evaluate on test data
            fold_score = 0.0
            for i in test_indices:
                segment_idx = np.searchsorted(all_chgpts[1:], i, side='right')
                segment_idx = min(segment_idx, len(segment_betas) - 1)
                
                beta_seg = segment_betas[segment_idx]
                y_pred = X[i:i+1] @ beta_seg
                
                # Compute M_scorer - for your case this is jnp.abs(u - v)
                m_score = M_scorer(y_pred, y[i:i+1])
                # Validate that M_scorer returns expected shape and convert to scalar
                if jnp.isscalar(m_score):
                    fold_score += float(m_score)
                elif m_score.shape == ():  # 0-dimensional array (scalar)
                    fold_score += float(m_score)
                elif m_score.shape == (1,):  # 1-element array
                    fold_score += float(m_score.item())
                else:
                    raise ValueError(f"M_scorer returned unexpected shape {m_score.shape}. "
                                   f"Expected scalar or (1,) array for single data point.")
                
                if P_scorer is not None:
                    p_score = P_scorer(beta_seg)
                    if jnp.isscalar(p_score):
                        fold_score += float(p_score)
                    elif p_score.shape == ():  # 0-dimensional array (scalar)
                        fold_score += float(p_score)
                    elif p_score.shape == (1,):  # 1-element array
                        fold_score += float(p_score.item())
                    else:
                        raise ValueError(f"P_scorer returned unexpected shape {p_score.shape}. "
                                       f"Expected scalar for single parameter vector.")
            
            cv_score += fold_score
        
        return cv_score
    
    print(f"Running 2-fold CV for {len(gamma_candidates)} gamma candidates...")
    
    scores = []
    for gamma in tqdm(gamma_candidates, desc="CV Progress"):
        scores.append(evaluate_single_gamma(gamma))
    
    # Find best gamma
    best_idx = np.argmin(scores)
    best_gamma = gamma_candidates[best_idx]
    
    print(f"CV finished. Best gamma: {best_gamma:.4f} (score: {scores[best_idx]:.4f})")
    
    # Fit final model
    best_chgpt_est = f(X, y, best_gamma)
    
    return best_gamma, best_chgpt_est

def weighted_erm(X, y, M, L, Delta_n, gamma, L_is_upper_bound=False, brute_force=False, return_gof=False, return_B_hat=False, P_func=lambda beta : 0, **kwargs):
    """
    Computes the change point estimate using the Weighted ERM method for a single L.
    (This function was provided as complete and is used as a helper).

    Delta_n is an int. 
    """
    n, p = X.shape

    if Delta_n ==1 : # No separation between change points, so use analytical binomial approximation. 
        if L_is_upper_bound:
            pi_i_l = lambda L, l, i : 1/(L - l + 1) * np.sum([scipy.special.comb(k - 1, l- 1, exact=False) * (i/n)**(l-1) * (1 - i/n)**(k-l) for k in range(l, L+1)])
        else: 
            pi_i_l = lambda L,l , i: scipy.special.comb(L-1, l-1, exact=False) * (i/n)**(l-1) * (1 - i/n)**(L-l)
        # Populate the pi L times n matrix
        pi = np.zeros((L, n))
        for l in range(L):
            for i in range(n):
                pi[l, i] = pi_i_l(L, l + 1, i + 1)
    else: 
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
    # pi = pi / np.sum(pi, axis=1, keepdims=True) # Normalize rows to sum to 1


    B_hat = np.zeros((p, L))

    # --- 1. Estimate signal coefficients B_hat ---
    for l_ in range(L):
        sample_weight = pi[l_, :] / np.sum(pi[l_, :]) # Ensure weights sum to 1
        
        @partial(jax.jit, static_argnums=(0, 1))
        def empirical_risk(M_func, P_func,beta, X_data, y_data, weights):
            return jnp.multiply(M_func((X_data @ beta).flatten(), y_data.flatten()), weights).sum() + P_func(beta)

        # def make_empirical_risk(P_func_):
        #     @partial(jax.jit, static_argnums=(0, 5))  # M_func = 0, P_func = 5
        #     def empirical_risk(M_func, beta, X_data, y_data, weights, P_func_):
        #         loss = M_func(X_data @ beta, y_data)
        #         risk = jnp.multiply(loss, weights).sum()
        #         if P_func_ is not None:
        #             risk += P_func_(beta)
        #         return risk
        #     return empirical_risk

        # empirical_risk_fn = make_empirical_risk(P_func)

        options = {'disp': False, 'maxiter': 1e3} # Useful for ensuring bounded logistic estimators 
        res = scipy.optimize.minimize(
            lambda beta: empirical_risk(M, P_func, beta, X, y, sample_weight),
            0.1*np.ones(p),
            jac=jax.grad(lambda beta: empirical_risk(M, P_func, beta, X, y, sample_weight)),
            method='L-BFGS-B',
            options=options
        )
        B_hat[:, l_] = res.x

    # --- 2. Compute Change Point Estimates using Goodness-of-Fit ---
    Theta_hat = X @ B_hat

    if brute_force and L > 1:
        opt_chgpts, opt_gof = WERM.gof.chgpt_gof_bf_mem_eff(
                M, Theta_hat, y, n, L, Delta_n, L_is_upper_bound=L_is_upper_bound,
                γ=gamma, batch_size=4096, gof_opt_output=True
            )
    elif L > 1:
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_binary_seg_mem_eff(M, Theta_hat, y, n, L, Delta_n, γ=gamma, min_window_size=30, gof_opt_output=True)
        # opt_chgpts, opt_gof = WERM.gof.chgpt_binary_segmentation(M, Theta_hat, y, n, L, min_window_size=Delta_n, gamma=gamma, gof_opt_output=True, γ=gamma)
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_dc(M, Theta_hat, y, n, L, Delta_n, γ=1.0, L_is_upper_bound=L_is_upper_bound, gof_opt_output=True)
        # opt_chgpts, opt_gof = WERM.gof.chgpt_gof_divide_conquer(M, Theta_hat, y, n, L, Delta_n, γ=gamma, L_is_upper_bound=L_is_upper_bound, gof_opt_output=True)
        opt_chgpts, opt_gof = WERM.gof.chgpt_binary_segmentation(M, Theta_hat, y, n, L, Delta_n, γ=0, L_is_upper_bound=L_is_upper_bound, gof_opt_output=True)
    else:
        opt_chgpts, opt_gof = np.array([]), np.inf  # No change points, no goodness-of-fit
        

    if return_gof:
        return (opt_chgpts, opt_gof, B_hat) if return_B_hat else (opt_chgpts, opt_gof)
    return (opt_chgpts, B_hat) if return_B_hat else opt_chgpts

def weighted_ermCV(X, y, M, L, Delta_n, gamma_candidates=None, L_is_upper_bound = True, brute_force=False, M_scorer = None, P_scorer = None, num_folds = 5, n_jobs = 4, **kwargs):
    """
    Run cross-validation to select the best gamma for WERM, and return
    the best change point configuration.
    """
    
    f = lambda X_train, y_train, gamma: weighted_erm(
        X_train, y_train, M, L=L, Delta_n=Delta_n, gamma=gamma,
        L_is_upper_bound=L_is_upper_bound, brute_force=brute_force, return_gof=False, return_B_hat=False, **kwargs
    )
    if M_scorer is None:
        M_scorer = M

    best_gamma, best_model_output = chgpt_k_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=gamma_candidates if gamma_candidates is not None else np.logspace(-1, 1, 10),
        M_scorer=M_scorer,
        P_scorer=P_scorer,
        k=num_folds,
        n_jobs=n_jobs
    )

    return best_gamma, best_model_output

def weighted_erm_extended(X, y, M, L, Delta_n, gamma, brute_force=False, P_func=lambda beta: 0, **kwargs):
    """
    Computes the optimal change point configuration by searching over L_tilde <= L
    and selecting the best based on a penalized goodness-of-fit score.
    """

    n, p = X.shape
    candidate_chgpts = []
    candidate_scores = []
    fit_scores = []
    penalty_scores = []

    # Iterate from L_tilde = 1 up to the max L provided
    for L_tilde in tqdm(range(1, L + 1), desc="Penalized Search over L"):
        # Get the unpenalized GOF score by setting gamma=0 for the search
        chgpts, fit_score = weighted_erm(
            X, y, M, L=L_tilde, Delta_n=Delta_n, gamma=0.0,
            L_is_upper_bound=False, brute_force=brute_force, return_gof=True, P_func=P_func,**kwargs
        )

        if fit_score is None: continue

        # Calculate the penalty for this configuration
        s = len(chgpts[chgpts != -1])
        assert s == (L_tilde-1) , f"Expected {L_tilde-1} change points, got {s}."
        
        # Penalty term: gamma * log(C(n-1, s))
        if s >= n:
            log_penalty_term = np.inf
        else:
            # log_penalty_term = np.log(L * scipy.special.comb(n - 1, s, exact=False) + 1e-9) # Correct
            log_penalty_term = (p * L + L) * np.log(n) # BIC, based on the number of parameters and data points

        # Print the scale of the penalty term vs. the GOF score
        if fit_score > 0:
            print(f"fit score: {fit_score:.4f}, gamma * log-penalty term: {gamma * log_penalty_term:.4f} (gamma={gamma})")
            print(f"Change points: {chgpts}, Number of change points: {s}, L_tilde: {L_tilde}, n: {n}")
        penalized_score = fit_score + gamma * log_penalty_term
        
        candidate_chgpts.append(chgpts)
        candidate_scores.append(penalized_score)
        fit_scores.append(fit_score)
        penalty_scores.append(log_penalty_term)

    # Return the change point configuration with the minimum penalized score
    if not candidate_scores:
        return np.array([], dtype=int)
    print(f"Candidate scores: {candidate_scores}")
    print(f"Fit scores: {fit_scores}")
    print(f"Penalty scores: {penalty_scores}")
    
    return candidate_chgpts, fit_scores, penalty_scores

def weighted_erm_extended_penalized(X, y, M, L, Delta_n, gamma, brute_force=False, gof_output = False, P_func=lambda beta: 0, **kwargs):
    """
    Computes the optimal change point configuration by searching over L_tilde <= L
    and selecting the best based on a penalized goodness-of-fit score.
    """

    candidate_chgpts, fit_scores, penalty_scores = weighted_erm_extended(
        X, y, M, L=L, Delta_n=Delta_n, gamma=gamma, brute_force=brute_force, P_func=P_func,**kwargs
    )
    candidate_scores = np.array(fit_scores) + gamma * np.array(penalty_scores)
    best_idx = np.argmin(candidate_scores)

    if gof_output:
        return candidate_chgpts[best_idx], candidate_scores
    return candidate_chgpts[best_idx]

def weighted_erm_extended_penalizedCV(X, y, M, L, Delta_n, gamma_candidates=None, brute_force=False, M_scorer = None, P_scorer = None, num_folds = 5, n_jobs = 4, **kwargs):
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
        k=num_folds,
        n_jobs=n_jobs
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

def weighted_erm_num_chgptCV(X, y, M, Delta_n, Lmax=7, brute_force=False, return_gof=False, P_func=lambda beta: 0, M_scorer=None, num_folds=5, n_jobs=4, **kwargs):
    """
    Cross-validation to select the best number of change points L_tilde.
    Returns the best L_tilde and the corresponding change point configuration.
    """
    f = lambda X_train, y_train, L_tilde: weighted_erm(
        X_train, y_train, M, L=L_tilde, Delta_n=Delta_n, gamma=0.0,
        brute_force=brute_force, P_func=P_func, return_gof=return_gof, **kwargs
    )

    if M_scorer is None:
        M_scorer = M

    best_L_tilde, best_chgpt_est = chgpt_k_fold_cv(
        f=f,
        X=X,
        y=y,
        gamma_candidates=np.arange(1, Lmax + 1),
        M_scorer=M_scorer,
        P_scorer=None,
        k=num_folds,
        n_jobs=n_jobs
    )

    return best_L_tilde, best_chgpt_est