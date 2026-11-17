# Myocardial heuristic exploration: interpretation and audit trail

The exploratory notebook is `Myocardial Infarction Experiment - heuristic exploration.ipynb`.
It leaves the archived figure and the strict mathematical audit intact. The code is visible in
the notebook; `tools/myocardial_exploration.py` is its identical, checkpointable execution source.

## What is varied

- Minimum segment length: 1, 85, or 169 observations, enforced at both ends on the distinct-age grid.
- Auxiliary fitting loss: least squares, logistic plus L2 (1e-8 or 1e-6), or logistic plus L1
  (1e-6 or 1e-4). Logistic response generation is retained even for squared auxiliary loss.
- Subsequent posterior fitting loss: logistic, with each of the four penalty settings independently.
- Number of damped covariance updates: 1, 3, 10, 30; initialize at the identity, damping 0.25,
  eigenvalue clipping to [1e-4, 100]. Also evaluate the direct constrained LS moment approximation.
- Numerical state accuracy: coarse QMC 128 / at most 40 solver evaluations; declared representative
  cases are rerun at 128 / 10 and 512 / 100. Solver evaluations are not covariance updates.

There are 3 x (5 x 4 + 1) x 4 = **252 primary coarse comparisons**. The deterministic refinement
set is every auxiliary choice at 3 and 30 updates with Delta=1, every LS update budget there,
and the direct LS approximation at each Delta: 15 covariance setups x 2 accuracies x 4 posterior
fits = 120 refined comparisons. A denser posterior-penalty sweep reuses seven refined Delta=1
covariance setups: the five LS estimates and moderate-ridge auxiliary estimates at 3/30 updates.
Thirteen L2 strengths from 1e-9 to 1e-3 and nine L1 strengths from 1e-7 to 1e-3, logarithmically
spaced, add 154 comparisons with 160-node likelihood quadrature.
These form a bounded sensitivity exploration, not an exhaustive search of continuous settings.

## What is not tuned

Prior count probabilities remain 1/2 each. The no-change state is represented once. Locations are
uniform conditional on a change, within the admissible grid. The weights are obtained by marginalizing
that prior, not by passing a desired change location. The raw data, preprocessing, random seed, and
posterior observation loss are fixed. No likelihood temperature is introduced. Every result is retained,
including large moment residuals and probabilities below 1/2.

The primary likelihood is the audited joint density p(V,Y | configuration). A separate diagnostic
removes the Gaussian p(V | configuration) term to quantify the effect of conditional-only scoring.
That diagnostic is not presented as the same posterior, nor does it restore the old covariance formula.

## Finite iteration is not convergence

The legacy procedure returned finite iterates, often averaged, without a sufficiently small residual
in all the coupled equations. It could therefore return a plausible-looking matrix even when no
positive-definite matrix matched the empirical moments. The new finite alternating scheme similarly
returns approximate estimates deliberately, but records the residual and its numerical regularization.
It is original-style, not a bit-for-bit recreation of the old frozen-diagonal algorithm.

For the strict LS audit, the moment system uniquely required a negative-definite transformed signal
covariance. Allowing nonzero residuals does not remove that fact; it makes approximation possible.
For this centered, no-intercept logistic model, the model's marginal response mean is 1/2 whereas the
observed rate is about 0.232. Gaussian-design and observation-model assumptions are not established
for these clinical features. Approximate posterior count probabilities are thus not calibration guarantees.

## Small penalties

The weighted mean loss is normalized by the sum of weights. L2 adds lambda times the squared coefficient
norm (no factor 1/2); L1 adds lambda times the absolute coefficient sum. Numerical magnitudes alone do
not establish that these are negligible, especially with the design divided by sqrt(n). Fits record KKT
residuals. Ridge scaling is checked independently against sklearn. The state equations and empirical
adjustment remain the unpenalized ones: using them with small-penalty fits is explicitly a heuristic,
not an unproved claim that the existing asymptotic theorem already covers penalized fitting.

## Outcome-based choices

Conditional-curve similarity is measured by total variation relative to the archived conditional mass
function on the full union of supports, assigning zero to excluded ages. A second TV measure restricts
and renormalizes the reference on the common support. Reference mass retained is also reported,
so that a restrictive Delta cannot appear to match simply by hiding excluded mass. P(1)>1/2
is reported separately. Ranking by either outcome is exploratory selection on the same dataset and must
be disclosed if such a setting is used in a paper. A desirable probability is not a numerical correctness
test. Numerical refinement can reveal a desirable coarse result to be unstable.
