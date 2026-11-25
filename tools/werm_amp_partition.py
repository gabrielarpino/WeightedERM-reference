"""Exact shared partition prior and search for the WERM/AMP comparison.

A boundary ``c`` is the first observation of the new segment (zero-based).
All segments are nonempty; only *interior* segments have minimum size ``gap``.
This is the canonical WERM search support.  Using the same support for AMP is
a deliberate matched-support adaptation: the original AMP enumerator also
constrains endpoint segments and uses strict spacing inequalities.

Unknown-count priors are uniform over K=1,...,L, then uniform over admissible
partitions given K.  Fixed-count priors condition on K=L.  Marginals and MAP
search are exact on the integer observation grid, without enumeration or
interpolation.
"""

from itertools import combinations
from math import comb, log

import numpy as np


def _dimensions(n, L, gap):
    values = (n, L, gap)
    if any(isinstance(v, (bool, np.bool_)) or not np.isscalar(v)
           or not np.isfinite(v) or int(v) != v or v < 1 for v in values):
        raise ValueError('n, L and gap must be positive integers')
    return tuple(int(v) for v in values)


def _partition_count(n, K, gap):
    if K == 1:
        return 1
    # Subtract gap-1 from each successive inter-boundary displacement.
    available = n - 1 - (K - 2) * (gap - 1)
    return comb(available, K - 1) if available >= K - 1 else 0


def log_partition_counts(n, L, gap):
    """Return log(number of partitions) for K=1,...,L; -inf if infeasible.

    For K>=2 the count is binomial(n-1-(K-2)(gap-1), K-1).
    """
    n, L, gap = _dimensions(n, L, gap)
    counts = [_partition_count(n, K, gap) for K in range(1, L + 1)]
    return np.asarray([log(c) if c else -np.inf for c in counts])


def _binomial_array(upper, lower):
    """Small-order binomial coefficients, including zero for invalid uppers."""
    upper = np.asarray(upper, dtype=float)
    answer = np.ones_like(upper)
    for j in range(1, lower + 1):
        answer *= (upper - j + 1) / j
    return np.where(upper >= lower, answer, 0.)


def _conditional_marginals(n, K, gap):
    count = _partition_count(n, K, gap)
    if not count:
        raise ValueError('Infeasible segment count and minimum spacing')
    if K == 1:
        return np.ones((1, n))
    m = K - 1
    # cdf[r-1,i] = P(c_r <= i), with boundaries indexed r=1,...,m.
    cdf = np.zeros((m, n))
    locations = np.arange(1, n)
    for r in range(1, m + 1):
        before = _binomial_array(locations - 1 - (r - 1) * (gap - 1), r - 1)
        after = _binomial_array(n - locations - 1 - (m - r) * (gap - 1), m - r)
        mass = before * after / count
        if not np.isclose(mass.sum(), 1., atol=2e-12, rtol=2e-12):
            raise ArithmeticError('Partition-boundary probabilities do not sum to one')
        cdf[r - 1, 1:] = np.cumsum(mass)
        cdf[r - 1, -1] = 1.  # Every boundary is <= n-1, exactly.
    phi = np.vstack((1. - cdf[:1], cdf[:-1] - cdf[1:], cdf[-1:]))
    if np.min(phi) < -1e-11:
        raise ArithmeticError('Negative partition marginal beyond roundoff')
    phi = np.maximum(phi, 0.)
    phi /= phi.sum(axis=0, keepdims=True)
    return phi


def prior_marginals(n, L, gap, fixed_count=False):
    """Return exact L-by-n label probabilities under the shared prior.

    Every requested count must be feasible.  With ``fixed_count=True`` use
    exactly L segments; otherwise average the K-conditional marginals with
    probability 1/L.  Labels present for K segments are 0,...,K-1.
    """
    n, L, gap = _dimensions(n, L, gap)
    if not _partition_count(n, L, gap):
        raise ValueError('Infeasible maximum segment count and minimum spacing')
    if fixed_count:
        return _conditional_marginals(n, L, gap)
    phi = np.zeros((L, n))
    for K in range(1, L + 1):
        phi[:K] += _conditional_marginals(n, K, gap) / L
    return phi


def partition_minimize(losses, gap):
    """Exact O(n*K) additive-loss minimizer with exactly K nonempty segments.

    ``losses[i,k]`` is the loss of assigning observation i to segment k.
    Prefix minima remove the otherwise quadratic search over previous cuts.
    Ties use the first prefix minimizer and the earliest final cut.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2 or not np.isfinite(losses).all():
        raise ValueError('losses must be a finite two-dimensional matrix')
    n, K, gap = _dimensions(*losses.shape, gap)
    if not _partition_count(n, K, gap):
        raise ValueError('Infeasible segment count and minimum spacing')
    if K == 1:
        return np.empty(0, dtype=int)
    prefix = np.vstack((np.zeros(K), np.cumsum(losses, axis=0)))
    previous = np.full(n, np.inf)
    previous[1:] = prefix[1:n, 0]
    back = np.full((K, n), -1, dtype=int)
    for segments in range(2, K):
        candidates = previous - prefix[:n, segments - 1]
        running = np.minimum.accumulate(candidates)
        improves = candidates < np.r_[np.inf, running[:-1]]
        first_argmin = np.maximum.accumulate(np.where(improves, np.arange(n), -1))
        current = np.full(n, np.inf)
        ends = np.arange(1 + (segments - 1) * gap, n)
        current[ends] = prefix[ends, segments - 1] + running[ends - gap]
        back[segments, ends] = first_argmin[ends - gap]
        previous = current
    ends = np.arange(1, n)
    objective = previous[ends] + prefix[n, K - 1] - prefix[ends, K - 1]
    winner = int(np.argmin(objective))
    if not np.isfinite(objective[winner]):
        raise ValueError('Infeasible segment count and minimum spacing')
    last = int(ends[winner])
    boundaries = [last]
    for segments in range(K - 1, 1, -1):
        last = int(back[segments, last])
        boundaries.append(last)
    return np.asarray(boundaries[::-1], dtype=int)


def map_partition(log_likelihood, gap, fixed_count=False):
    """MAP partition under the shared count/location prior.

    ``log_likelihood`` has shape (n,L), with independent-observation log
    likelihoods conditional on each segment label.  Candidate objectives
    are negative log likelihood + log(N_K) + log(L) for unknown K, and
    negative log likelihood + log(N_L) for fixed K=L.  The returned
    ``candidate_objectives`` has L entries; unsearched counts are +inf.
    """
    ll = np.asarray(log_likelihood, dtype=float)
    if ll.ndim != 2 or not np.isfinite(ll).all():
        raise ValueError('log_likelihood must be a finite two-dimensional matrix')
    n, L, gap = _dimensions(*ll.shape, gap)
    logs = log_partition_counts(n, L, gap)
    if not np.isfinite(logs[-1]):
        raise ValueError('Infeasible maximum segment count and minimum spacing')
    candidates = (L,) if fixed_count else range(1, L + 1)
    objectives = np.full(L, np.inf)
    fits = {}
    for K in candidates:
        cp = partition_minimize(-ll[:, :K], gap)
        labels = np.searchsorted(cp, np.arange(n), side='right')
        negative_log_likelihood = -float(ll[np.arange(n), labels].sum())
        log_prior = -float(logs[K - 1]) - (0. if fixed_count else log(L))
        objectives[K - 1] = negative_log_likelihood - log_prior
        fits[K] = dict(cp=cp, count=K - 1, segments=K,
                       negative_log_likelihood=negative_log_likelihood,
                       log_prior=log_prior, objective=float(objectives[K - 1]))
    winner = int(np.argmin(objectives)) + 1
    return dict(fits[winner], candidate_objectives=objectives)


def amp_signal_label_subsets(L, K):
    """Signal-label support used by the source AMP posterior search.

    A no-change configuration may use any one of the L working signals.
    For K>=2, the first signal is 0 and subsequent signals form an increasing
    (K-1)-subset of 1,...,L-1. Thus an unused working signal may be skipped;
    reaching signal 3 does not require introducing a segment for signal 2.

    This is the support of upstream ``eta_to_psi_jax_combinations_mapped``.
    Its prior-marginal helper does not consistently cover the K=1,2 cases;
    the AMP-specific functions below use this same support for both steps.
    """
    _, L, _ = _dimensions(1, L, 1)
    _, K, _ = _dimensions(1, K, 1)
    if K > L:
        raise ValueError('K cannot exceed the number of working signals L')
    if K == 1:
        return [(label,) for label in range(L)]
    return [(0, *tail) for tail in combinations(range(1, L), K - 1)]


def amp_prior_marginals(n, L, gap, fixed_count=False):
    """Exact AMP marginals for a consistent count/location/label prior.

    Unknown K is uniform on 1,...,L. Conditional on K, admissible boundary
    sets and the label tuples in ``amp_signal_label_subsets`` are each uniform
    and independent. The no-change mass is consequently shared across all
    working signals, rather than assigned entirely to the first signal.
    Fixed K=L is identical to the ordinary ordered-label prior.
    """
    n, L, gap = _dimensions(n, L, gap)
    if fixed_count:
        return prior_marginals(n, L, gap, fixed_count=True)
    if not _partition_count(n, L, gap):
        raise ValueError('Infeasible maximum segment count and minimum spacing')
    phi = np.zeros((L, n))
    for K in range(1, L + 1):
        conditional = _conditional_marginals(n, K, gap)
        label_sets = amp_signal_label_subsets(L, K)
        for labels in label_sets:
            phi[list(labels)] += conditional / (L * len(label_sets))
    phi /= phi.sum(axis=0, keepdims=True)
    return phi


def amp_map_partition(log_likelihood, gap, fixed_count=False):
    """Exact joint MAP using the same AMP prior as ``amp_prior_marginals``.

    Unknown-count objectives include log(L)+log(N_K)+log(number of label
    tuples), so allowing additional label tuples does not change the total
    prior mass assigned to a candidate count. No true count, true labels or
    true boundaries are inputs. With K=L this reduces to ``map_partition``.
    """
    ll = np.asarray(log_likelihood, dtype=float)
    if ll.ndim != 2 or not np.isfinite(ll).all():
        raise ValueError('log_likelihood must be a finite two-dimensional matrix')
    n, L, gap = _dimensions(*ll.shape, gap)
    if fixed_count:
        result = map_partition(ll, gap, fixed_count=True)
        return dict(result, selected_signal_labels=tuple(range(L)),
                    configuration_support='source_MAP_signal_labels')
    logs = log_partition_counts(n, L, gap)
    if not np.isfinite(logs[-1]):
        raise ValueError('Infeasible maximum segment count and minimum spacing')
    objectives = np.full(L, np.inf)
    winners = {}
    for K in range(1, L + 1):
        label_sets = amp_signal_label_subsets(L, K)
        log_prior = -log(L) - float(logs[K - 1]) - log(len(label_sets))
        for labels in label_sets:
            cp = partition_minimize(-ll[:, list(labels)], gap)
            assigned = np.asarray(labels)[np.searchsorted(cp, np.arange(n), side='right')]
            negative_log_likelihood = -float(ll[np.arange(n), assigned].sum())
            objective = negative_log_likelihood - log_prior
            if objective < objectives[K - 1]:
                objectives[K - 1] = objective
                winners[K] = dict(cp=cp, count=K - 1, segments=K,
                    selected_signal_labels=labels,
                    negative_log_likelihood=negative_log_likelihood,
                    log_prior=log_prior, objective=objective)
    winner = int(np.argmin(objectives)) + 1
    return dict(winners[winner], candidate_objectives=objectives,
                configuration_support='source_MAP_signal_labels')
