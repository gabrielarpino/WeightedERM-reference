import numpy as np
import jax.numpy as jnp
import scipy as sp

def hausdorff(x1, x2):
  """ Calculate the Hausdorff distance between two sets of points using scipy."""
  x1 = np.array(x1).reshape(-1, 1)
  x2 = np.array(x2).reshape(-1, 1)
  return max(sp.spatial.distance.directed_hausdorff(x1, x2)[0], sp.spatial.distance.directed_hausdorff(x2, x1)[0])

def hausdorff_jax(x1, x2):
    """ Calculate the Hausdorff distance between two sets of points using jax. NOT TESTED. """
    x1 = jnp.array(x1).reshape(-1, 1)
    x2 = jnp.array(x2).reshape(-1, 1)
    return jnp.maximum(jnp.max(jnp.abs(x1 - x2)), jnp.max(jnp.abs(x2 - x1)))

def closest_point_distance(candidate_set, true_point): 
    """ Find the distance from the closest point in candidate_set to true_point."""
    candidate_set = np.array(candidate_set).reshape(-1, 1)
    true_point = np.array(true_point).reshape(-1, 1)
    distances = np.abs(candidate_set - true_point)
    return np.min(distances)