"""Exact fixed-count partition helper used by the upper-bound experiment."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')


import numpy as np


def squared_partition(theta, y, gap):
    """Exact fixed-count DP, with the same feasible set/ties as chgpt_gof_dp.

    Additive segment costs allow a prefix minimum instead of the quadratic
    predecessor loop. End segments have no minimum length beyond one row;
    adjacent internal boundaries must be at least gap apart.
    """
    theta, y = np.asarray(theta), np.asarray(y)
    n, count = theta.shape
    if y.shape != (n,) or count < 1 or n < count or gap < 1:
        raise ValueError('Invalid partition dimensions')
    losses = (theta - y[:, None]) ** 2
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite partition loss')
    prefix = np.vstack((np.zeros(count), np.cumsum(losses, axis=0)))
    if count == 1:
        return np.empty(0, dtype=int)
    previous = np.full(n, np.inf)
    previous[1:] = prefix[1:n, 0]
    back = np.full((count, n), -1, dtype=int)
    for segments in range(2, count):
        candidates = previous - prefix[:n, segments - 1]
        candidates[0] = np.inf
        running = np.minimum.accumulate(candidates)
        improves = candidates < np.r_[np.inf, running[:-1]]
        first_argmin = np.maximum.accumulate(np.where(improves, np.arange(n), -1))
        current = np.full(n, np.inf)
        ends = np.arange(gap + 1, n)
        current[ends] = prefix[ends, segments - 1] + running[ends - gap]
        back[segments, ends] = first_argmin[ends - gap]
        previous = current
    final = previous[1:] + prefix[n, count - 1] - prefix[1:n, count - 1]
    last = int(np.argmin(final)) + 1
    if not np.isfinite(final[last - 1]):
        raise ValueError('Infeasible fixed count')
    boundaries = [last]
    for segments in range(count - 1, 1, -1):
        last = int(back[segments, last])
        boundaries.append(last)
    return np.array(boundaries[::-1], dtype=int)


