# WER-7 exact-DP production promotion

Status: **PASS — ready for manual review, not yet merged**

- Human approval recorded on Linear WER-7 before implementation.
- Accepted WER-6 base: `d3e7a01db767c6d9032a0db0bf8f41f43a213249`.
- Rerun: `WERM_PHASE1_WORKERS=1 bash scripts/rebuttal/check_phase1.sh`.
- Result: 52 tests passed; 23/23 DP/brute-force cases passed; maximum
  objective difference `0`; DP was worse than greedy in 0 cases.

## Production routing

`weighted_erm` and `HVI` now call one `_exact_partition_search` adapter.
The CV, extended, penalized, elbow, posterior, and experiment paths already
delegate to those two estimators, so no notebook or scientific driver was
rewritten. Brute force and binary segmentation remain directly available in
`WERM.gof` as audit oracles.

The legacy `brute_force` argument remains accepted, emits a `FutureWarning`,
and still uses exact DP. It cannot silently switch a production estimator back
to an oracle. The default scalar `gamma` is normalized to the verified legacy
K-only penalty. A custom finite K-only table or callable is supported through
`partition_penalty` when `gamma=0`; providing both is rejected as ambiguous.
A callable that cannot be invoked as `penalty(K)` is rejected explicitly.

## Observable behavior changes

- Fixed-K changepoints are exact under the same feasible set. With nonzero
  `gamma`, the reported objective now includes the verified legacy K-only
  constant; locations cannot change solely because of that constant.
- Upper-bound searches can select K using the exact raw objective plus the
  K-only penalty, rather than the previous unpenalized greedy route.
- K=1 now returns its finite raw objective instead of the historical `inf`
  placeholder, which is required for honest upper-bound K selection.

These are partition-search changes only. Coefficient fitting, coefficient
regularization `P_func`, model fitting, preprocessing, and notebook contents
are unchanged.

## Narrow smoke comparison

| Workflow | Accepted oracle baseline | Promoted production smoke | Assessment |
| --- | --- | --- | --- |
| M1 | `[8]` | `[8]` | unchanged reduced smoke |
| M2 | `[11, 13]` | `[11, 13]` | unchanged reduced smoke |
| M3 | `[7]` | `[7]` | unchanged reduced smoke |
| Myocardial | archived result `[956]` | archived structural check only | not freshly rerun |
| FRED-MD | committed pickle loaded | first 24 rows, 6 columns, K=2 gives `[14]` | new exact-DP subset smoke, not a reproduction |

Every smoke ran sequentially with one effective worker and numerical-library
thread caps set to one. The historical `/scratch/ga442/Weighted-ERM` tree was
read only and was not modified. Full command output is in
`results/test_output.txt`.
