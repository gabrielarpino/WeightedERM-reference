# Myocardial signal-strength audit

## Specification and scope

Source: `../WERM-paper/JRSSB-revision-rebuttal/JRSSB_revision.tex`, paragraph
“Estimating the signal strength matrix” (lines 1052–1060), equations
`b-fp-implicit`, `mu-fp-implicit`, `cross-cov-fp-implicit`, `b-hat-eqn`,
and `theta-deb-fp-eqn`. The myocardial section explicitly chooses **squared
fitting loss for estimating Gamma, then logistic fitting loss for the posterior**.
The observation model remains logistic in BOTH stages. Squared fitting loss
does not imply linear/Gaussian responses.

The legacy API, notebooks, manuscript and saved reference remain unchanged.
`WERM/signal_strength.py` is additive. The new executed notebook starts from
raw UCI data and freshly fits coefficients, adjusted scores and rho; it loads
no archived fitted quantities. A posterior safety gate prevents silently
substituting an invalid rho or falling back to the archived estimate.

## Notation and coupled equations

Gamma = rho = Cov(Z), Lambda multiplies Z, and K = Cov(G), where G is
independent Gaussian noise and V = Z Lambda + G. Therefore

```
Cov(V)   = Lambda.T @ rho @ Lambda + K
Cov(Z,V) = rho @ Lambda.
```

Gamma is the predictor covariance, not the coefficient covariance: no extra
factor of delta belongs here. The empirical target is the uncentered second
moment `S = Theta_adjusted.T @ Theta_adjusted / n`, not a centered covariance.
For L=2 the independent unknowns number 12: b (2), Lambda (4), K (3), Gamma
(3). There are 9 fixed-point equations plus 3 symmetric moment equations.
The manuscript's larger count should be checked for double-counting symmetric entries.

## Exact squared-loss reduction with logistic observations

Let `M(t,y)=(t-y)^2`, `w_il=phi[l,i]`, and `s=psi_i`. Its prox is
`t=(v+2*b_l*w_il*y)/(1+2*b_l*w_il)`. Define

```
a_il = 2*b_l*w_il/(1+2*b_l*w_il).
```

The b equation becomes `mean_i a_il=1/delta`, independent of rho, Lambda,
K and y. Solve each positive scalar root by bracketing. The exact empirical
adjustment is `Theta_adjusted_il=Theta_hat_il+2*b_l*w_il*(Theta_hat_il-y_i)`.

Write `h_s=E sigmoid'(sqrt(rho_ss)*N(0,1))`. Gaussian integration by parts
gives `E[Z Y | psi=s]=rho[:,s]*h_s`. Zero-intercept symmetry gives
`E[Y]=E[Y^2]=1/2`. The second fixed-point equation consequently gives

```
T_sl      = mean_i(a_il * 1{psi_i=s}) / mean_i(a_il)
Lambda_sl = h_s*T_sl.
```

With `A=Lambda.T@rho@Lambda` and `v_il=h_s*(rho@Lambda)[s,l]`, independence
of G from (Z,Y) gives

```
E[(V_l-Y)(V_m-Y)] = A_lm + K_lm - v_il - v_im + 1/2
K_lm = delta*mean_i[a_il*a_im*(A_lm-v_il-v_im+1/2)]
       / (1-delta*mean_i[a_il*a_im]).
```

Gaussian G and Bernoulli noise are integrated analytically; only smooth 1D
h integrals remain. Gaussian quadrature is checked at 80 and 160 nodes.
Broad Gaussian signals use adaptive integration on the logistic length scale.
Independent Monte Carlo samples of Z, G and actual Bernoulli Y verify the
original score and covariance equations, not just our rearranged formula.

## Linear feasibility certificate and inverse

Set `H=diag(h)` and `C=H@rho@H`. C must be PSD. Then `A=T.T@C@T`,
`v_il=(C@T)[s,l]`, and `D_lm=delta*mean(a_l*a_m)` imply

```
S_lm = [(T.T@C@T)_lm
        - delta*mean_i(a_il*a_im*(v_il+v_im)) + D_lm/2] / (1-D_lm).
```

This is affine linear in C's three independent entries. Construct its
3-by-3 operator from the symmetric basis matrices. If full rank, it uniquely
identifies C. A negative eigenvalue of this unique C certifies that **no PSD
rho solves the empirical equations**; it is not an optimizer/initialization
failure. Rank deficiency instead indicates nonidentification and does not
by itself establish infeasibility.

For feasible C, recover r=rho_ss by scalar inversion `C_ss=r*h(r)^2`, then
`rho_st=C_st/(h_s*h_t)`. The scalar map increases from 0 to 1/(2*pi): Stein's
identity gives `sqrt(r)*h(r)=E[N*sigmoid(sqrt(r)*N)]`, whose derivative in
sqrt(r) is strictly positive. Thus `0<C_ss<1/(2*pi)` is also necessary.
Positive diagonal congruence preserves PSD. The linear inverse and scalar
roots provide a deterministic alternative to stochastic alternating updates.
When infeasible, a multistart constrained approximation is saved only as a
diagnostic, explicitly NOT as an accepted exact root.

## Legacy-code findings

1. `estimate_rho_iterative` freezes approximately solved Lambda/K while updating
   rho, changes Monte Carlo keys at each iteration, and returns after three
   steps without checking all equations at its final rho. An inner optimizer
   stopping successfully is not a joint-system convergence certificate.
2. Separately fitted, frozen diagonals do not implement the full joint system.
   They can make the remaining moment equations incompatible.
3. Both fixed-point and empirical-b routines average all iterates, including
   transients, after a fixed iteration count. They do not return certified roots.
   `success_status` actually contains a residual norm. Archived residuals
   are approximately 0.015–0.064, despite requested tolerance 1e-4.
4. Independent tanh correlation entries do not ensure PSD for L>2. Even for
   L=2, adding epsilon to off-diagonal scaling and resetting the exact diagonal
   can technically lose PSD extremely close to unit correlation.
5. `general_fp_rho` is not a usable joint solver: it references undefined M
   and rho and has no rho coordinates in its optimization vector.
6. The fixed-point routine generates `V=Z@nu+G`, making its nu = Lambda.
   The logistic posterior computes `nu.T@inv(rho)@nu+K`, which would instead
   require nu = Cov(Z,V) = rho@Lambda. Passing the actual fixed-point output
   mixes incompatible parameterizations.
7. Correct Gaussian conditioning is `Sigma=Lambda.T@rho@Lambda+K`,
   `C_ZV=rho@Lambda`, `mu=C_ZV@solve(Sigma,V_i)`, and
   `R=rho-C_ZV@solve(Sigma,C_ZV.T)`. The old conditional covariance's
   transposes also do not match a cross-covariance interpretation. The new
   standalone likelihood uses these formulas and passes independent scalar
   quadrature checks with both zero and nonzero Lambda.
8. The old logistic likelihood omits log p(V). The manuscript specifies the
   joint likelihood p(V,Y). Since Lambda/K depend on candidate configuration,
   p(V) is generally candidate-dependent and cannot be canceled. The new
   likelihood includes it and integrates the logistic link by Gaussian quadrature.
9. The original configuration prior is uniform per candidate, not per count.
   The fresh notebook uses one null (-1), mass 1/2, and 61 proper age-grid
   splits totaling 1/2, deriving phi from exactly that prior. Uniform over age
   splits is not uniform over all n-1 observation-index splits; this deliberate
   finite-grid choice is stated rather than claiming they coincide.
10. Row scaling of unpenalized weights is harmless: multiplying row l by c
    preserves its coefficient fit, divides b_l by c, and preserves b_l*phi_li.
    This invariance is tested. Nonunit column sums alone are therefore not
    proof of an error. However switching from exactly-one to at-most-one
    weights changes their SHAPE, as does fitting and adjusting with different shapes.

## Fresh-data findings

The auxiliary LS pilot split is index 1127 (age 67). Its empirical adjusted
second moment is approximately

```
S = [[0.02653459, 0.02760159],
     [0.02760159, 0.03349759]].
```

The affine operator has rank 3 and condition number 52.18. Its unique solution is

```
C = [[-0.03126414,  0.02198920],
     [ 0.02198920, -0.03446053]],
eigenvalues(C) = [-0.05490954, -0.01081513].
```

The linear reconstruction error is about 2e-17. No valid rho solves these
particular empirical equations under the stated observation model and pilot
configuration. This does not disprove asymptotic theory: finite-sample or
model mismatch can make empirical moments infeasible. It also does not
establish infeasibility for every different configuration or model.

The finding survives independent refits with unrestricted observation-index pilot
splits: with age-grid half-count weights the effective eigenvalues are approximately
(-0.05871,-0.01073); with all-index half-count weights (-0.03736,-0.00936);
and with legacy exactly-one-change weights (-0.03211,-0.00892). Thus it is not
merely an artifact of the corrected prior or of restricting the pilot to age starts.

Observed response prevalence is 0.231678, while centered zero-intercept
Gaussian logistic signals imply mean 1/2 for every rho. Age-sorted clinical
covariates are also not verified Gaussian-design samples. These are model
adequacy concerns, not numerical tolerances. An intercept is a plausible
extension, not a proven cure, and must enter fitting, fixed points and likelihood
consistently. It must not be introduced silently to recover a preferred curve.

## Downstream status and scientific choice

There is a second independent obstruction. For BOTH positive-weight logistic
designs, a linear program finds a nonzero direction d satisfying
`(2*y_i-1)*(X_i@d) >= 0` and the sum of these signed margins >= 1.
Rechecked witness minima are about -4.1e-14 and -1.8e-14 (floating-point
roundoff); the sums are 1. This is a numerical quasi-separation certificate:
the unpenalized logistic likelihood has no finite maximizer. Tests distinguish
complete separation, quasi-separation, overlap and zero-weight observations.
Sklearn finite iterates have coefficient norms about 2290 and 4616 and tiny
gradients, but small gradients do not prove a finite MLE exists. FISTA also
failed its KKT target after 30000 iterations. Adding an intercept alone does
not remove an existing separating direction; addressing separation may require
regularization or a scientifically justified feature/model change.

Least squares remains the auxiliary rho-estimation loss. The final WERM fit
and posterior are still intended to use logistic loss. No valid new rho has
passed the feasibility gate, so no new posterior is presented as validated.
The paper's tolerance 0.08 is large relative to norm(S), approximately 0.058;
loosening that threshold does not resolve negative-definite C.

Next choices: extend the observation model (for example, an estimated intercept)
and address separation (for example, regularization, whose theory differs),
or explicitly accept a constrained approximate moment fit and finite-iteration
logistic estimator for a diagnostic posterior, with residuals and instability reported.
Neither should be silently substituted for a solution of the paper's equations.
The manuscript's old numeric claims have not been edited and should be revisited
once the model/estimator decision is resolved.
