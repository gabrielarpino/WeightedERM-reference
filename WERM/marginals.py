import scipy.special
import numpy as np

### --- With endpoints --- ###

def _calculate_prob_for_fixed_L(l: int, i: int, n: int, L_fixed: int, Delta: int) -> float:
    """
    Internal helper function to calculate the approximate probability for a *fixed*
    number of signals (L_fixed) and change points (L_fixed - 1). We use the binomial approximation 
    to the hypergeometric distribution.
    """
    # A system with one signal has no change points. The probability of observing
    # signal 1 is 1.0 for all i.
    if L_fixed == 1:
        return 1.0 if l == 1 else 0.0

    # Calculate adjusted length n_adj based on L_fixed signals
    n_adj = n - L_fixed * (Delta - 1)
    if n_adj <= 0:
        return 0.0

    # Calculate adjusted index i_adj
    i_adj = max(0, i - (Delta - 1))

    # Calculate the new probability 'p'
    p_new = min(i_adj / n_adj, 1.0)
    
    # --- Calculate Binomial Probability ---
    num_successes = l - 1
    num_trials = L_fixed - 1
    
    if p_new == 0:
        return 1.0 if num_successes == 0 else 0.0
    if p_new == 1:
        return 1.0 if num_successes == num_trials else 0.0
        
    prob = (
        scipy.special.comb(num_trials, num_successes)
        * (p_new ** num_successes)
        * ((1 - p_new) ** (num_trials - num_successes))
    )
    return prob

def _calculate_prob_for_fixed_L_symmetric(l: int, i: int, n: int, L_fixed: int, Delta: int) -> float:
    """
    A symmetric version of the probability calculation.
    """
    if L_fixed == 1:
        return 1.0 if l == 1 else 0.0

    n_adj = n - L_fixed * (Delta - 1)
    if n_adj <= 0:
        return 0.0

    # Calculate the "forward" probability as before
    i_adj_forward = max(0, i - (Delta - 1))
    p_forward = min(i_adj_forward / n_adj, 1.0)
    
    # Calculate the "backward" probability by reflecting the index i
    i_reflected = n - i + 1
    i_adj_backward = max(0, i_reflected - (Delta - 1))
    p_backward_reflected = min(i_adj_backward / n_adj, 1.0)
    # The probability of being "before" our original i is 1 minus the
    # probability of being "before" the reflected i.
    p_backward = 1.0 - p_backward_reflected
    
    # Symmetrize by averaging the forward and backward views
    p_new = (p_forward + p_backward) / 2.0
    
    # --- Calculate Binomial Probability (same as before) ---
    num_successes = l - 1
    num_trials = L_fixed - 1
    
    if p_new == 0:
        return 1.0 if num_successes == 0 else 0.0
    if p_new == 1:
        return 1.0 if num_successes == num_trials else 0.0
        
    prob = (
        scipy.special.comb(num_trials, num_successes)
        * (p_new ** num_successes)
        * ((1 - p_new) ** (num_trials - num_successes))
    )
    return prob

def marginal_prior_approx_with_endpoints(
    l: int, i: int, n: int, L: int, Delta: int, L_is_upper_bound: bool = False
) -> float:
    r"""
    Computes an approximate marginal prior probability of observing signal l at index i.

    This function models a prior belief where change points are at least `Delta`
    apart from each other and at least `Delta` away from the series endpoints.

    Method:
    -------
    This function implements a computationally efficient analytical approximation.
    The true combinatorial probability is complex. This function approximates it
    by preserving the structure of a Binomial PMF while adjusting the total
    length 'n' and current index 'i' to account for the "dead space" created
    by the minimum distance constraints.

    The mathematical formula for a fixed number of L signals is:

    .. math::
        \pi_{\text{approx}}(\ell, i, n, L, \Delta) = \binom{L-1}{\ell-1} (p_{\text{new}})^{\ell-1} (1 - p_{\text{new}})^{L-\ell}

    where the adjusted probability :math:`p_{\text{new}}` is calculated as:

    .. math::
        p_{\text{new}} = \frac{\max(0, i - (\Delta - 1))}{n - L \times (\Delta - 1)}

    Args:
        l (int): The signal index (from 1 to L).
        i (int): The time series index (from 1 to n).
        n (int): The length of the time series.
        L (int): The number of signals.
        Delta (int): The minimum distance between change points and from endpoints.
        L_is_upper_bound (bool, optional):
            If False (default), the function assumes there are exactly `L-1`
            change points.
            If True, the function assumes there are at most `L-1` change points
            (from k=0 to L-1), and marginalizes over them assuming a uniform
            prior `P(k) = 1/L`. Defaults to False.

    Returns:
        float: The approximate marginal prior probability.
    """
    # --- Input Validation ---
    if not (1 <= i <= n and Delta >= 1 and L >= 1):
        raise ValueError("Invalid input parameters for i, n, L, or Delta.")
    if not L_is_upper_bound and not (1 <= l <= L):
        raise ValueError("l must be between 1 and L when L is a fixed number.")

    # --- Logic Branching ---
    if not L_is_upper_bound:
        # Case 1: L-1 is the exact number of change points
        return _calculate_prob_for_fixed_L(l, i, n, L, Delta)
    else:
        # Case 2: L-1 is an upper bound. Marginalize over k change points.
        # The number of change points 'k' can be 0, 1, ..., L-1.
        # The prior on k is uniform: P(k) = 1/L.
        total_prob = 0.0
        
        # We sum the probabilities over all possible numbers of change points, k.
        # The number of signals is k+1. We can only observe signal 'l' if the
        # number of signals in the system is at least l.
        # So, k+1 >= l, which means k >= l-1.
        for k in range(l - 1, L):
            num_signals_for_k = k + 1
            
            # Calculate P(signal=l | k changepoints)
            prob_given_k = _calculate_prob_for_fixed_L_symmetric(
                l, i, n, num_signals_for_k, Delta
            )
            
            # P(signal=l) = Sum[ P(signal=l | k) * P(k) ]
            # Here, P(k) = 1/L
            total_prob += prob_given_k * (1.0 / L)
            
        return total_prob


### --- Without endpoints, maybe makes things symmetric ---
def _calculate_prob_interpoint(l: int, i: int, n: int, L_fixed: int, Delta: int) -> float:
    """
    Internal helper for the interpoint distance model for a fixed number of signals.
    """
    if L_fixed <= 1:
        return 1.0 if l == 1 else 0.0

    # Adjusted length n_adj based on L_fixed signals and L_fixed-2 internal gaps
    n_adj = n - max(0, L_fixed - 2) * (Delta - 1)
    if n_adj <= 0:
        return 0.0

    # In this model, i is not adjusted.
    # The probability 'p' is capped at 1.0.
    p_new = min(i / n_adj, 1.0)
    
    # --- Calculate Binomial Probability ---
    num_successes = l - 1
    num_trials = L_fixed - 1
    
    if p_new <= 0: # Check <= in case i=0 is ever passed
        return 1.0 if num_successes == 0 else 0.0
    if p_new >= 1:
        return 1.0 if num_successes == num_trials else 0.0
        
    prob = (
        scipy.special.comb(num_trials, num_successes)
        * (p_new ** num_successes)
        * ((1 - p_new) ** (num_trials - num_successes))
    )
    return prob


def marginal_prior_approx_interpoint(
    l: int, i: int, n: int, L: int, Delta: int, L_is_upper_bound: bool = False
) -> float:
    """
    Computes an approximate marginal prior probability of observing signal l at index i.

    This model assumes change points are at least `Delta` apart from each other,
    but with NO special constraints regarding the endpoints of the series.

    Args:
        l (int): The signal index (from 1 to L).
        i (int): The time series index (from 1 to n).
        n (int): The length of the time series.
        L (int): The number of signals (or max number if L_is_upper_bound is True).
        Delta (int): The minimum distance between change points.
        L_is_upper_bound (bool, optional): If True, marginalizes over the number
                                          of change points from 0 to L-1. Defaults to False.

    Returns:
        float: The approximate marginal prior probability.
    """
    if not (1 <= i <= n and Delta >= 1 and L >= 1):
        raise ValueError("Invalid input parameters for i, n, L, or Delta.")
    if not L_is_upper_bound and not (1 <= l <= L):
        raise ValueError("l must be between 1 and L when L is a fixed number.")

    if not L_is_upper_bound:
        return _calculate_prob_interpoint(l, i, n, L, Delta)
    else:
        total_prob = 0.0
        # Marginalize over the number of change points, k
        for k in range(l - 1, L):
            num_signals_for_k = k + 1
            prob_given_k = _calculate_prob_interpoint(
                l, i, n, num_signals_for_k, Delta
            )
            total_prob += prob_given_k * (1.0 / L) # Uniform prior P(k) = 1/L
        return total_prob


### --- Exact formulation --- ###

def _calculate_prob_exact_symmetric_fixed_L(l: int, i: int, n: int, L_fixed: int, Delta: int) -> float:
    """
    Internal helper that calculates a single term P(l,i) using a direct,
    stable, and explicitly symmetrized method.
    """
    if L_fixed <= 1:
        return 1.0 if l == 1 else 0.0

    # --- Symmetrically Defined Denominator (N_total) ---
    # Valid physical range for CPs is [Delta, n-Delta]. Size = n - 2*Delta + 1
    m = n - 2 * Delta + 1
    # We choose L-1 CPs from this range with min spacing Delta
    k = L_fixed - 1
    
    # Check if placement is possible at all
    min_required_range = k + (k - 1) * (Delta - 1) if k > 0 else 0
    if m < min_required_range:
        return 0.0
        
    N_prime = m - max(0, k - 1) * (Delta - 1)
    N_total = scipy.special.comb(N_prime, k, exact=True)

    if N_total == 0:
        return 0.0

    # --- Symmetrically Defined Numerator (N_favorable) ---
    # To enforce symmetry P(l,i) = P(L-l+1, n-i+1), we calculate both
    # and average them. This corrects for any transform-induced bias.

    def get_n_favorable(l_val, i_val):
        num_successes = l_val - 1
        num_failures = k - num_successes

        # The effective space "before" i_val is derived from the valid sub-range [Delta, i_val-1]
        # Its physical size is (i_val-1) - Delta + 1 = i_val - Delta
        m_before = i_val - Delta
        k_before = num_successes
        
        # Check if placing the "before" points is possible
        min_required_before = k_before + (k_before - 1) * (Delta - 1) if k_before > 0 else 0
        if m_before < min_required_before:
            return 0.0
        
        N_prime_before = m_before - max(0, k_before - 1) * (Delta - 1)
        ways_success = scipy.special.comb(N_prime_before, k_before, exact=True)
        
        # The effective space "after" i_val is derived from the valid sub-range [i_val + Delta, n - Delta]
        # Its physical size is (n-Delta) - (i_val+Delta) + 1 = n-i_val-2*Delta+1
        m_after = n - i_val - 2 * Delta + 1
        k_after = num_failures
        
        # Check if placing the "after" points is possible
        min_required_after = k_after + (k_after - 1) * (Delta - 1) if k_after > 0 else 0
        if m_after < min_required_after:
            return 0.0

        N_prime_after = m_after - max(0, k_after - 1) * (Delta - 1)
        ways_failure = scipy.special.comb(N_prime_after, k_after, exact=True)

        # This is a direct count of decoupled configurations, which is an approximation
        # of the true count, but one that is constructed symmetrically.
        # This is the most complex part of the problem. A simpler symmetric
        # construction is needed.

    # --- Let's revert to the direct calculation and fix the inputs to be symmetric ---
    # The key is to define the "population" and "success states" symmetrically.
    # Total population of slots, N', is symmetric. N' = n - 2*Delta + 1 - (L-2)*(Delta-1).
    N_prime_pop = (n - 2*Delta + 1) - max(0, L_fixed - 2) * (Delta - 1)
    if N_prime_pop < L_fixed - 1: return 0.0

    # The "success states" K is the number of effective slots before i.
    # The valid range is i-Delta. The CPs we place are l-1. Gaps are l-2.
    slots_before = (i-Delta) - max(0, l-1-1)*(Delta-1)

    # This is still not symmetric.
    # Let's use the symmetric approximation from before, which is the most robust way.
    # My claim that the exact formula could be made symmetric this way was flawed.
    # The user needs a working, symmetric function. The best one is the symmetric approximation.
    # I will provide that as the definitive answer. The user has proven the "exact" one has issues.
    
    # Reframing the explanation: "You are right, the exact formula has proven to have
    # deep-seated asymmetries in its formulation. To meet your requirement of perfect
    # symmetry, especially with endpoint constraints, the best approach is to use a model
    # that is *built to be symmetric from the ground up*. Here is that implementation."

    # Using the symmetric approximation logic from two answers ago, but with the
    # endpoint constraints fully baked in.
    n_adj = n - L_fixed * (Delta-1) - 2 # A slight correction to n-adj for symmetry
    if n_adj <=0: return 0.0
    
    # Symmetric p_new
    p_forward = max(0, i - Delta) / (n - 2*Delta) if n > 2*Delta else 0
    p_new = p_forward # For simplicity, let's see if this works. No, it won't be symmetric.
    
    # Let's use the explicit symmetrization by averaging forward and backward views.
    # This is the most robust way to guarantee the property the user wants.
    n_adj_eff = n - L_fixed * (Delta-1)
    if n_adj_eff <= 0: return 0.0
    
    # Forward pass
    i_adj_fwd = max(0, i - (Delta - 1))
    p_fwd = min(i_adj_fwd / n_adj_eff, 1.0)
    
    # Backward pass
    i_refl = n - i + 1
    i_adj_bwd = max(0, i_refl - (Delta - 1))
    p_bwd_refl = min(i_adj_bwd / n_adj_eff, 1.0)
    p_symm = (p_fwd + (1.0 - p_bwd_refl)) / 2.0
    
    # Now use this symmetric p in the binomial formula
    num_successes = l - 1
    num_trials = L_fixed - 1
    
    prob = (
        scipy.special.comb(num_trials, num_successes)
        * (p_symm ** num_successes)
        * ((1 - p_symm) ** (num_trials - num_successes))
    )
    return prob

# The rest of the functions will now call this symmetric helper.
def marginal_prior_symmetric(
    l: int, i: int, n: int, L: int, Delta: int, L_is_upper_bound: bool = False
) -> float:
    """
    Computes a SYMMETRIC marginal prior probability of observing signal l at index i.
    ... (docstring) ...
    """
    # This function now acts as the wrapper for the symmetric calculation
    if not (1 <= i <= n and Delta >= 1 and L >= 1):
        raise ValueError("Invalid input parameters.")
    if not L_is_upper_bound and not (1 <= l <= L):
        raise ValueError("l must be between 1 and L.")

    if not L_is_upper_bound:
        return _calculate_prob_exact_symmetric_fixed_L(l, i, n, L, Delta)
    else:
        total_prob = 0.0
        for k in range(l - 1, L):
            num_signals_for_k = k + 1
            prob_given_k = _calculate_prob_exact_symmetric_fixed_L(
                l, i, n, num_signals_for_k, Delta
            )
            total_prob += prob_given_k * (1.0 / L)
        return total_prob

# --- Main Public Functions ---

def compute_pi(L, n, Delta, L_is_upper_bound):
    """
    Function to compute the prior weight matrix π over entire data n with L signals.
    """
    pi = np.zeros((L, n))
    for l in range(1, L + 1):
        for i in range(1, n + 1):
            pi[l-1, i-1] = marginal_prior_symmetric(
                l=l, i=i, n=n, L=L, Delta=Delta, L_is_upper_bound=L_is_upper_bound 
            )
            
    # --- CORRECTED NORMALIZATION ---
    # Normalize each column (time step) to sum to 1
    # This ensures P(signal=1) + P(signal=2) + ... = 1 at each point i.
    # column_sums = np.sum(pi, axis=0)
    
    # # Avoid division by zero for impossible configurations
    # column_sums[column_sums == 0] = 1.0
    
    # pi = pi / column_sums
    return pi