# Provenance

`myglm3` is a local, fidelity-repaired derivative of [`gabrielarpino/myglm-2`](https://github.com/gabrielarpino/myglm-2), cloned at upstream commit `273dae2ba39f2033e53a3e6766acfabeec761030`.

The automatic `bsa_chgpt` routine restores the no-split candidate from the authors' preserved implementation at `c4d934f`. It retains the current direct-`lambda` `glmnet` fitting interface, which replaces the original path fit plus coefficient interpolation without changing the candidate objective.
