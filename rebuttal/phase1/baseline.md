# Lightweight pre-change baseline

These are deliberately reduced checks. `SMOKE` is not a reproduction, and no
duplicated notebook block was selected as canonical.

| Workflow | Source notebook/script | Check run | Result | Notes/blocker |
| --- | --- | --- | --- | --- |
| M1 | `M1 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic sparse-linear call (`n=18, p=4, L=2, seed=101`) | SMOKE | Full notebook and R competitor paths were not run. |
| M2 | `M2 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic differential-sparsity call (`n=20, p=5, L=3, seed=202`) | SMOKE | Full notebook and R competitor paths were not run. |
| M3 | `M3 Comparison Experiments.ipynb`; `WERM.estimators.weighted_erm` | Import plus one deterministic penalized logistic call (`n=18, p=4, L=2, seed=303`) | SMOKE | Full notebook and custom R logistic competitor were not run. |
| Myocardial | `Myocardial Infarction Experiment.ipynb` | Checked for the expected local posterior result | BLOCKED | UCI data are fetched remotely and `results/myocardial_infarction_posterior_exact_one_chgpt.pkl` is absent. |
| FRED-MD | `FRED-MD Real Data Experiment.ipynb` | Loaded the committed local pickle and checked X/y dimensions | SMOKE | Fresh R preprocessing and `inferchange` were not run; archived local shape is `X=(773, 114), y=(773,)`. |

The three comparison notebooks contain duplicated historical experiment blocks
with different grids and worker counts. This baseline does not silently choose
one of them. The intended Python version is exactly 3.10.10 in `setup.py`;
this server supplied 3.10.20, so the requirements lock targets Python 3.10 and
the patch-version difference is recorded in the ignored run log.
