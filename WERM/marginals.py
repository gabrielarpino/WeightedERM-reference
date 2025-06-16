import scipy.special
import numpy as np

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
            prob_given_k = _calculate_prob_for_fixed_L(
                l, i, n, num_signals_for_k, Delta
            )
            
            # P(signal=l) = Sum[ P(signal=l | k) * P(k) ]
            # Here, P(k) = 1/L
            total_prob += prob_given_k * (1.0 / L)
            
        return total_prob

def compute_pi(L, n, Delta, L_is_upper_bound):
    """Function to compute the prior weight matrix π over entire data n with L signals."""
    pi = np.zeros((L, n))
    for l in range(1, L + 1):
        for i in range(1, n + 1):
            pi[l-1, i-1] = marginal_prior_approx_with_endpoints(
                l=l, i=i, n=n, L=L, Delta=Delta, L_is_upper_bound=L_is_upper_bound
            )
    # Normalize π for stability
    pi = pi / np.sum(pi, axis=1, keepdims=True)
    return pi