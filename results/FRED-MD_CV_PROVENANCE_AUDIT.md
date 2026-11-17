# FRED-MD CV provenance audit

Audit date: 2026-08-30

The package-level `weighted_erm_joint_cv` hard-segment refit contract changed
at 2026-08-28 18:40:53 Europe/London.  Before that change, hard segment fits
used `M_scorer`; afterward they use the base fitting loss `M`, with `M_scorer`
used only on held-out predictions.

## Invalid for CV-based conclusions

The following artifacts predate the correction.  Their CV scores, CV-selected
penalty constants, and CV-selected segment counts must not be used:

- `FRED-MD_elbow-six-methods_penalty-constant-CV.pkl`
- `FRED-MD_elbow-six-methods_penalty-constant-CV_delta40.pkl`
- Penalized/CV-selected portions of
  `FRED-MD_elbow-six-methods_Huber-penalty-constant-CV_delta40.pkl`
- `FRED-MD_refined-penalty-delta-specification-search*`
- CV-selected portions of `FRED-MD_unpenalized-and-Cge3-Lq-Huber-delta-search*`
- `FRED-MD_realistic-C-search_Huber-LS*`
- CV-selected portions of `FRED-MD_unpenalized-fold-sensitivity-CV-vs-elbow*`
- CV-selected portions of
  `FRED-MD_unpenalized-high-fold-sensitivity-CV-vs-elbow*`
- `FRED-MD_unrolled-unpenalized-CV-score-curves*`
- `FRED-MD_no-outlier-truncation-CV-comparison*`
- `FRED-MD_unpenalized-high-fold-20-50-100-CV*`
- CV tables in `FRED-MD_real-data-exploration/`
- `FRED-MD_full-joint-cv-search/`
- `FRED-MD_unpenalized-huber-epsilon-joint-cv/`
- `FRED-MD_huber-l1-absolute-delta1-cv/`

Cached CV output in the older experiment notebooks should likewise be treated
as stale unless it is explicitly backed by one of the post-correction artifact
directories below.

## Unaffected portions of old artifacts

Direct full-data fits at a fixed `(L, C)` do not invoke the CV hard-refit code.
Unpenalized full-data elbow curves are therefore unaffected.  In particular,
the original unpenalized Huber epsilon=1.35 elbow result was reproduced fresh:
two changes at 1982-01 and 2008-09 on the 1960-01--2024-05 data.

Fixed-fit dates remain computationally valid, but a fixed fit is not a valid
selected result when its `L` or `C` was chosen using a pre-correction CV score.

## Post-correction artifacts

These runs started after the correction and use the intended loss contract:

- `FRED-MD_period-intercept-CV/`
- `FRED-MD_full-current-unpenalized-CV/`
- `FRED-MD_high-delta-unpenalized-CV-vs-elbow/`
- `FRED-MD_original-Huber-L1-local-delta-elbow/`
- `FRED-MD_original-Huber-elbow-reproduction/`

`FRED-MD_full-current-penalized-CV/` also used corrected code, but was canceled
and is incomplete; its checkpoint must not be presented as a completed grid.

## Separate numerical reproducibility issue

The corrected period/intercept run did produce the Huber-1.35, L1, no-intercept,
two-fold, n/30 result with `C=0.30`, three changes, and mean absolute CV score
0.5354979447.  A later corrected rerun assigned a materially worse score to the
same candidate.  This is not old-CV contamination: it is a separate numerical
stability problem in L-BFGS-B optimization of nonsmooth L1 objectives.  That
candidate must be labelled unstable until it survives a deterministic solver
or repeated-run stability check.
