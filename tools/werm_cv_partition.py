"""Linear-time-per-column dynamic program for additive partition losses."""
import numpy as np


def additive_partition(losses, gap, end_min=1):
    """Minimize sum_i losses[i, assigned_segment] over ordered partitions.

    There are exactly ``losses.shape[1]`` nonempty segments. Interior segments
    have at least ``gap`` rows; first and last segments have at least
    ``end_min`` rows. With end_min=1 this is the canonical repository feasible
    set. With gap=end_min=m every segment has at least m rows. The one-segment
    case requires n >= end_min, rather than counting its endpoints twice.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2:
        raise ValueError('losses must be a two-dimensional matrix')
    n, count = losses.shape
    if (count < 1 or gap < 1 or end_min < 1 or int(gap) != gap
            or int(end_min) != end_min or not np.isfinite(losses).all()):
        raise ValueError('Invalid loss matrix or segment constraints')
    gap, end_min = int(gap), int(end_min)
    if count == 1:
        if n < end_min:
            raise ValueError('Infeasible minimum segment size')
        return np.empty(0, dtype=int)
    if n < 2*end_min+(count-2)*gap:
        raise ValueError('Infeasible fixed count and minimum segment sizes')
    prefix = np.vstack((np.zeros(count), np.cumsum(losses,axis=0)))
    previous = np.full(n,np.inf)
    previous[end_min:] = prefix[end_min:n,0]
    back = np.full((count,n),-1,dtype=int)
    for segments in range(2,count):
        candidates = previous-prefix[:n,segments-1]
        running = np.minimum.accumulate(candidates)
        improves = candidates < np.r_[np.inf,running[:-1]]
        first_argmin = np.maximum.accumulate(np.where(improves,np.arange(n),-1))
        current = np.full(n,np.inf)
        ends = np.arange(end_min+(segments-1)*gap,n)
        current[ends] = prefix[ends,segments-1]+running[ends-gap]
        back[segments,ends] = first_argmin[ends-gap]
        previous = current
    ends = np.arange(end_min,n-end_min+1)
    objective = previous[ends]+prefix[n,count-1]-prefix[ends,count-1]
    winner = int(np.argmin(objective))
    if not np.isfinite(objective[winner]):
        raise ValueError('Infeasible fixed count and minimum segment sizes')
    last = int(ends[winner])
    boundaries = [last]
    for segments in range(count-1,1,-1):
        last = int(back[segments,last])
        boundaries.append(last)
    return np.asarray(boundaries[::-1],dtype=int)
