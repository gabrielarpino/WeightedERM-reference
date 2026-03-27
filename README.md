# WeightedERM Reference

This repository contains the reference implementation and main experiment notebooks for WeightedERM (WERM), developed for high-dimensional change point detection. The repository is notebook-centered: the main experiments and applications live in Jupyter notebooks, while the underlying implementation lives in the `WERM/` source directory.

## Where To Start

For the main experimental workflows, start with:

- `M1 Comparison Experiments.ipynb`
- `M2 Comparison Experiments.ipynb`
- `M3 Comparison Experiments.ipynb`

These are the main comparison notebooks and the best entry point for the core uses of WeightedERM.

## Repository Structure

- `WERM/`: source code implementing WeightedERM and related utilities
- `FRED-MD data/`: data files used by the FRED-MD real-data notebook
- `*.ipynb`: experiment and analysis notebooks
- `setup.py`: local package metadata for installing `WERM`
- `LICENCE.txt`: license information

## Notebook Guide

### Main comparison notebooks

- `M1 Comparison Experiments.ipynb`  
  Main experiment notebook for the linear model with sparse signals, i.e. sparse regression vector coefficients.

- `M2 Comparison Experiments.ipynb`  
  Main experiment notebook for the linear model with differentially sparse signals.

- `M3 Comparison Experiments.ipynb`  
  Main experiment notebook for the logistic model with sparse signals, including experiments with `l1` penalization.

### Real-data notebooks

- `FRED-MD Real Data Experiment.ipynb`  
  Real-data analysis on the FRED-MD macroeconomic dataset.

- `Myocardial Infarction Experiment.ipynb`  
  Real-data analysis for the myocardial infarction application.

### Supporting analysis notebooks

- `SE match estimation.ipynb`  
  State-evolution and estimation experiments supporting the method and its analysis.

- `SE match estimation - plotting.ipynb`  
  Plotting companion notebook for the SE matching experiments.

- `SE match posterior.ipynb`  
  Posterior-related experiments and visualizations.

- `Visualize Marginals.ipynb`  
  Utility notebook for inspecting and visualizing marginal prior weights.

## Source Code Guide

The main implementation lives in `WERM/`. The most important files are:

- `WERM/estimators.py`  
  Core WeightedERM estimation routines, including changepoint estimation helpers.

- `WERM/gof.py`  
  Goodness-of-fit search routines used to estimate changepoint locations.

- `WERM/marginals.py`  
  Utilities for changepoint priors and marginal weight matrices.

- `WERM/model.py`  
  Basic model definitions and helper routines used throughout the notebooks.

Additional supporting files include:

- `WERM/comparison.py`
- `WERM/posterior.py`
- `WERM/config.py`

## Installation

This repository is primarily a reference research codebase rather than a polished package, so the safest way to run it is to create a fresh environment, install the required dependencies, and then install the local `WERM` module.

The code and results in this repository were developed and are expected to work with Python `3.10.10`.

### Minimal requirements

The following packages are enough to run the quick-start example below:

- `numpy`
- `scipy`
- `matplotlib`
- `jax`
- `jaxlib`
- `jaxopt`
- `scikit-learn`
- `tqdm`
- `joblib`
- `torch`

### Additional notebook dependencies

Some notebooks require extra packages:

- `notebook` or `jupyterlab` to open notebooks
- `rpy2` and a working R installation for `FRED-MD Real Data Experiment.ipynb`
- `pandas` and `ucimlrepo` for `Myocardial Infarction Experiment.ipynb`
- `KDEpy` and `kde_diffusion` for parts of `M1 Comparison Experiments.ipynb` and `M2 Comparison Experiments.ipynb`
- `seaborn` for parts of `M3 Comparison Experiments.ipynb`

### Example installation

```bash
git clone <repo-url>
cd Weighted-ERM

python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

If you want to run all remaining notebooks, also install:

```bash
pip install notebook pandas ucimlrepo seaborn KDEpy rpy2
```

and install `kde_diffusion` separately if needed for the M1/M2 notebooks. This was tested in Python version `3.10.10`. 

## Minimal Quick Start

The example below gives a small end-to-end WeightedERM workflow in the style of the linear-model experiments. It:

1. generates a Gaussian design,
2. generates sparse regression vectors,
3. creates responses with two true change points,
4. constructs the simplest uniform marginal weights,
5. runs WeightedERM for a fixed number of change points,
6. returns the estimated change point locations.

```python
import numpy as np
import jax.numpy as jnp

from WERM.marginals import compute_pi
from WERM.estimators import weighted_erm

rng = np.random.default_rng(0)

n = 120
p = 60
num_chgpts = 2
L = num_chgpts + 1

X = rng.normal(size=(n, p)) / np.sqrt(n)

sparsity = 0.15
beta = np.zeros((p, L))
for ell in range(L):
    support = rng.choice(p, size=int(sparsity * p), replace=False)
    beta[support, ell] = rng.normal(scale=1.0, size=len(support))

true_chgpts = np.array([40, 85])
segments = np.zeros(n, dtype=int)
segments[true_chgpts[0]:true_chgpts[1]] = 1
segments[true_chgpts[1]:] = 2

noise_std = 0.5
y = np.array([
    X[i] @ beta[:, segments[i]] + rng.normal(scale=noise_std)
    for i in range(n)
])

M = lambda u, v: jnp.power(u - v, 2)

pi = compute_pi(L=L, n=n, Delta=1, L_is_upper_bound=False)
print("Weight matrix shape:", pi.shape)

estimated_chgpts = weighted_erm(
    X=X,
    y=y,
    M=M,
    L=L,
    Delta_n=1,
    gamma=0.0,
    L_is_upper_bound=False,
    brute_force=False,
    return_gof=False,
    return_B_hat=False,
)

print("True change points:", true_chgpts)
print("Estimated change points:", estimated_chgpts)
```

Notes:

- `compute_pi(...)` shows the simple uniform marginal weights for this setup.
- `weighted_erm(...)` uses the same simple closed-form weight construction internally when `Delta_n=1`.
- Observations are assumed to be ordered in time/index order.

## Notes

- This is the reference/original research codebase. The notebooks are the primary interface for understanding and reproducing the workflows.
- In changepoint settings, observations are assumed to be ordered in time or index order.

## License

See `LICENCE.txt`.
