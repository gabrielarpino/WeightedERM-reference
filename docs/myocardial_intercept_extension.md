# Intercept/centering extension

This extension is in `Myocardial Infarction Experiment - intercept exploration.ipynb`.
It supplements, rather than replaces, the centered/no-intercept sensitivity grid.

## Centering is not an intercept

For a fixed feature transformation, X_raw = X_centered + 1 m^T.
The predictions X_centered beta + alpha equal X_raw beta + (alpha - m^T beta).
An unpenalized intercept therefore makes centering an exact reparameterization, including with
L1 or L2 penalties on the slopes alone. On these data, the maximum prediction difference from
this transformation is below 4e-14. Without an intercept the feasible prediction vectors change,
and centered logistic regression fixes the prediction at the mean feature vector to 1/2.
Centering the feature columns does not remove an imbalanced response rate.

## Explicit observation-model extension

Use Y | Z_s ~ Bernoulli(sigmoid(mu_s + Z_s)), with Z centered Gaussian of covariance rho.
For each proposed diagonal rho_ss, calibrate mu_s to the same global observed response rate r:
E sigmoid(mu_s + sqrt(rho_ss) N) = r. The rate is estimated once from all observations;
it is not determined by a pilot change location or the desired age-55 outcome. This specifies
an empirical-Bayes model extension. It is not the old zero-intercept model under another name.

Let h_s = E sigmoid'(mu_s + Z_s). Gaussian integration by parts still gives
Cov(Z_j,Y) = rho_js h_s. For squared auxiliary loss with a fitted intercept, the population
fitted intercept is r; the centered Bernoulli residual has second moment r(1-r).
Consequently the reduced LS state retains Lambda = diag(h) T but replaces the old 1/2 noise
term by r(1-r). The covariance equation is

K_lm = delta mean[a_l a_m (A_lm - v_l - v_m + r(1-r))] /
       (1 - delta mean[a_l a_m]),

where A = Lambda^T rho Lambda, v_l = h_s (rho Lambda)_sl, and
a_l = 2 b_l phi_l / (1 + 2 b_l phi_l). Independent Bernoulli/Gaussian simulation tests verify
the zero score, cross moment, and covariance equations.

For logistic fitting, the state solve has two additional fitted intercept unknowns alpha_l
and two additional equations E W_l = 0. The proximal loss uses alpha_l + t, whereas the latent
response uses mu_s + Z_s. Those two intercepts are not conflated. Center empirical adjusted
scores by the fitted total offset. In a centered design, that offset is the fitted intercept;
its score equation ensures that the mean adjustment is zero up to fitting precision.

Evaluate the joint density using Cov(V)=Lambda^T rho Lambda+K and Cov(Z,V)=rho Lambda,
with mu_s included in the conditional Bernoulli probability. Prior count odds remain 1:1,
and the conditional age-grid prior is unchanged. The aspect ratio uses 110 slopes; the fixed
number of unpenalized intercepts is not treated as Gaussian random-design coordinates.

## Scope and limitations

The intercept branch crosses 13 signal-strength estimates with four logistic posterior fits:
52 comparisons. Auxiliary estimates use LS, logistic L2 1e-6, or logistic L1 1e-4 at 1/3/10/30
damped updates, plus a direct LS moment approximation. The posterior fit strengths remain L2
1e-8/1e-6 and L1 1e-6/1e-4. Fitting intercepts are never penalized.

As before, using small-penalty fits with unpenalized state equations is approximate. Intercept
estimation and prevalence calibration are plug-in steps, not integrated posterior uncertainty.
The Gaussian-design assumptions are not proved for these clinical features. Allowing residuals
does not establish calibration of a posterior count probability.

The uncentered/no-intercept diagnostic adds 20 comparisons and intentionally retains the old
zero-mean equations after changing the fit. Its violated mean assumption is labeled explicitly;
it should not be selected over the coherent branch merely for meeting the desired outcome.

The direct LS intercept-aware fit has moment residual about 0.00035346, or 0.00661 relative to
the empirical moment norm (0.0534893). This compares with approximately 0.013 and 0.225 relative
error in the original centered/no-intercept fresh audit. The smaller residual is a moment-fit
result, not a posterior-validation result. All returned matrices, finite iteration counts,
state residuals, and posterior outcomes remain available in uniquely named checkpoints.
