# Lightweight pre-change baseline

These are deliberately reduced checks. `SMOKE` is not a reproduction, and no
duplicated notebook block was selected as canonical. The original development
tree at `/scratch/ga442/Weighted-ERM` is used strictly read-only when an artifact
was not ported to this cleaner reference repository.

| Workflow | Source notebook/script | Check run | Result | Notes/blocker |
| --- | --- | --- | --- | --- |
| M1 | `M1 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic sparse-linear call (`n=18, p=4, L=2, seed=101`) | SMOKE | Full notebook and R competitor paths were not run. |
| M2 | `M2 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic differential-sparsity call (`n=20, p=5, L=3, seed=202`) | SMOKE | Full notebook and R competitor paths were not run. |
| M3 | `M3 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic penalized logistic call (`n=18, p=4, L=2, seed=303`) | SMOKE | Full notebook and custom R logistic competitor were not run. |
| Myocardial | `Myocardial Infarction Experiment.ipynb`, archive-writing cell execution 54 | Loaded the historical result and checked required keys, posterior/configuration shapes, and finite posterior values | SMOKE | Read-only archive has `posterior=(63,)`, `eta=(63, 1)`, seed 42, and `chgpt_est=[956]`. This is archived output, not a fresh reproduction; the preceding historical cell used `Parallel(n_jobs=45)` and was not rerun. |
| FRED-MD | `FRED-MD Real Data Experiment.ipynb` | Loaded the committed local pickle and checked X/y dimensions | SMOKE | Fresh R preprocessing and `inferchange` were not run; local shape is `X=(773, 114), y=(773,)`. Historical `fredmd.csv` and `fredmd_data.RData` were located but not copied or rerun. |

## Historical reference tree

The check looks first in this repository, then in the read-only historical root
`/scratch/ga442/Weighted-ERM`. Set `WERM_HISTORICAL_ROOT` to an equivalent copy
on another machine. No check writes to that tree, and the myocardial pickle is
not copied into this repository. Without the historical tree (or a future local
copy of that result), Myocardial remains `BLOCKED` rather than being regenerated
or silently reinterpreted.

The three comparison notebooks contain duplicated historical experiment blocks
with different grids and worker counts, and the historical tree contains many
archived M1/M2/M3 result files without one unambiguous mapping to these smoke
cases. This baseline does not silently choose one. The intended Python version
is exactly 3.10.10 in `setup.py`;
this server supplied 3.10.20, so the requirements lock targets Python 3.10 and
the patch-version difference is recorded in the ignored run log.
