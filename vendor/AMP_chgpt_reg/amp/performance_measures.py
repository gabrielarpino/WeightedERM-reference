__all__ =[
    "norm_sq_corr",
    "MSE",
    "MSE_normalized",
    "hausdorff"
]

import numpy as np
import scipy as sp

def norm_sq_corr(x1, x2):
  if np.all(x1 == 0) or np.all(x2 == 0): return 0
  return np.square(np.dot(x1, x2)) / (np.square(np.linalg.norm(x1)) * np.square(np.linalg.norm(x2)))

def MSE(beta, beta_hat):
  return np.mean(np.square(beta - beta_hat))

def MSE_normalized(beta, beta_hat):
  return np.linalg.norm(beta - beta_hat, ord = 'fro')**2 / np.linalg.norm(beta)**2

def hausdorff(x1, x2):
  """ Calculate the Hausdorff distance between two sets of points using scipy."""
  x1 = np.array(x1).reshape(-1, 1)
  x2 = np.array(x2).reshape(-1, 1)
  return max(sp.spatial.distance.directed_hausdorff(x1, x2)[0], sp.spatial.distance.directed_hausdorff(x2, x1)[0])