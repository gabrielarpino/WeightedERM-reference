# WeightedERM: revised-paper experiments

Notebooks contain the paper figures, experiment code, and saved results. Open them directly to view the figures.

| Notebook | Paper results |
|---|---|
| [01 Sparse linear](01_sparse_linear.ipynb) | Figure 1 |
| [02 Logistic theory](02_logistic_theory.ipynb) | Figure 2 |
| [03 Linear theory](03_linear_theory.ipynb) | Figure 3 |
| [04 Posterior](04_posterior.ipynb) | Figure 4 |
| [05 Sparse, heavy-tailed](05_sparse_heavy_tailed.ipynb) | Figures 5–6 |
| [06 Sparse differences](06_sparse_differences.ipynb) | Figures 7–8 |
| [07 Sparse logistic](07_sparse_logistic.ipynb) | Figure 9 |
| [08 Myocardial infarction](08_myocardial.ipynb) | Figure 10 and sensitivity table |
| [09 FRED-MD](09_fred_md.ipynb) | FRED-MD results |
| [10 Dense Gaussian](10_dense_gaussian.ipynb) | Figure 11 |
| [11 Alternating prior](11_alternating_prior.ipynb) | Figures 12–13 |
| [12 Count selection](12_count_selection.ipynb) | Figure 14 |
| [13 Upper bound](13_upper_bound.ipynb) | Figure 15 |
| [14 Location prior](14_location_prior.ipynb) | Figure 16 |
| [15 AMP](15_amp.ipynb) | Figures 17–18 |

Use Python 3.10. From this directory:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

PDF display requires Poppler (`pdftoppm`). Plot regeneration also requires a LaTeX installation with `pdflatex`, PGF, `type1cm`, `cm-super`, and `dvipng`.

Each notebook starts with two switches:

- `REGENERATE_FIGURES = True`: plot the saved numerical results and display the regenerated figures.
- `RUN_EXPERIMENTS = True`: recompute the experiment, then its figures. Large simulations can take substantial time. Runtime plots reflect the machine used for the saved runs.

Full simulation reruns additionally require R (tested with 4.3.1), its development tools, and the competitors:

```sh
Rscript install_competitors.R
python -m pip install -r requirements-full.txt
```

Use the R installation associated with the Python environment (`R_HOME` if needed). AMP and the repaired GLMBSA implementation are bundled with their licenses. Frozen FRED-MD inputs and the myocardial data are included. Original setting indices are retained for reproducible random seeds; only settings used by the paper are run.

Unpenalized WERM uses 10-fold CV; coefficient-penalized WERM uses 2 folds. The FRED-MD notebook shows the saved 10-fold unpenalized scores selecting zero changes. Figure 2 uses $\kappa=1$.
