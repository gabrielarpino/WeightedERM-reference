# Completed myocardial exploration

## Updated preference (supersedes the recommendation below)

The preferred setup is now **no intercept**. When the auxiliary fit is logistic,
its penalty must match the posterior-input fit in **both family and strength**;
small constants are preferred. Least-squares auxiliary fits remain an alternative.
The earlier intercept/mixed-penalty result below is retained as a historical
sensitivity result, not the current recommendation. The exact-match no-intercept
comparison is recorded separately in `exact_matching_no_intercept/` beneath the
original heuristic exploration results directory. It retains all tested strengths,
iteration counts, fitting diagnostics, and posterior outcomes, not only favorable ones.

## Historical exploratory setting (intercept, mismatched penalties)

Among the finite settings tested, requiring P(1 change)>0.5 and conditional mass above age 55>0.5
in both numerical realizations, then minimizing mean total variation to the archived conditional
curve, selected:

- Centered, whitened features; unpenalized fitting intercepts; minimum segment length Delta=1.
- Count prior P(0)=P(1)=1/2; location uniform over the 61 proper distinct-age splits.
- Auxiliary logistic fit with L1 strength 1e-4, 30 damped rho updates from I, damping 0.25.
- Subsequent logistic fit with L2 strength 1.5e-6, intercept unpenalized.
- Penalties use weighted mean loss, with L2=lambda*sum(beta^2), not half this quantity.
- Intercept-aware observation model with logistic offsets calibrated to the global observed
  prevalence 0.23167848699763594; no true change location enters that calibration.

The selected rho is approximately [[0.28627969, 0.08243948], [0.08243948, 0.02404166]].
Its nonzero moment-equation residual is 0.11350346. This is deliberately a finite-iteration
heuristic, not an exact moment-equation solution or the setting with the smallest moment residual.

With 512 Sobol draws/seed 42 and an independent 1024 draws/seed 123, both with at most 100
state-solver evaluations, P(1) lies between 0.717686 and 0.747208. The conditional location mode
is age 81 in both, and the mass strictly above 55 exceeds 0.9999999999999 in both.
Mean TV distance to the archived conditional curve is 0.290673. It is more concentrated than
the archived curve, not identical. The count prior was never adjusted to obtain these results.

This setting was chosen after inspecting outcomes and must be described as exploratory if used
in the paper. Independent simulation draws check numerical sensitivity, not independent-data
validation. The small-penalty fits still use approximately unpenalized state equations.

## Where to view and reproduce

- `Myocardial Infarction Experiment - heuristic exploration.ipynb`: complete centered/no-intercept
  grid, all tables, posterior atlas, numerical refinement, denser penalty grid, independent-draw checks.
- `Myocardial Infarction Experiment - intercept exploration.ipynb`: coherent intercept extension,
  separately labeled uncentered/old-equation diagnostic, independent checks, final penalty resolution,
  and the selected curves and recipe in the last cell.
- `results/myocardial_intercept_exploration_f53f59063391/selected_exploratory_setup.json`: exact recipe.
- `results/myocardial_exploration_4026d0e775d2/`: first-stage checkpoints and complete tables.
- `results/myocardial_intercept_exploration_f53f59063391/`: extension checkpoints and complete tables.

Experiment code is visible in the notebooks; rerunning reuses matching checkpoints. The original
reference notebook/checkpoint is preserved. The checkpoint SHA256 remains
2531ba1839f2f747cee6299162429b44c8fed5b3c2c4c0ac640adc23709cd456.

## Important conclusions beyond the chosen curve

The archived high count probability depended on a uniform-per-configuration prior, not half–half
count odds. Under the archived likelihood and two-boundary mixture null, half–half odds give
P(1)=0.334942; the all-signal-0 null gives 0.267894. Conditional location shape and count evidence
are distinct quantities.

Centering does not replace a logistic intercept. With an unpenalized intercept and fixed feature
scaling, centering is an exact affine reparameterization: checked prediction differences are below
4e-14 on the actual data. A coherent intercept model substantially improves the LS auxiliary
moment fit (relative residual about 0.00661 versus about 0.225 without the intercept), though the
curve-selected recipe uses a different auxiliary fit and a larger residual.

The conditional prior already assigns 35/61=0.57377049 mass above age 55. A diffuse curve just
exceeding 0.5 beyond 55 is therefore not by itself strong localization evidence. The selected
intercept-aware curve is substantially more concentrated in the older ages.

Some coarse zero-intercept posteriors changed radically upon numerical refinement despite small
sampled state-equation residuals. These cases are retained, not hidden. The uncentered diagnostic
retaining zero-mean equations is not interchangeable with the model-aware intercept extension.

The combined numerical/regression suite passed 32 tests, including independent proximal roots,
sklearn ridge scaling, finite-iteration return behavior, centering invariance, independent
Bernoulli/Gaussian state simulation, and preservation of the reference analysis.
