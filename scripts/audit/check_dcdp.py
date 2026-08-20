import numpy as np

from WERM import DCDP_utils as D


def make(n=600, p=200, alpha=0.5, kappa=1.0, noise=np.sqrt(0.1), r=0.4, seed=11):
    rng = np.random.default_rng(seed)
    delta = n / p
    Sigma = r ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Xs = rng.normal(size=(n, p)) @ np.linalg.cholesky(Sigma).T
    mask = rng.random((p, 3)) < alpha
    B = rng.normal(0, kappa * np.sqrt(delta), (p, 3)) * mask / np.sqrt(n)
    Th = Xs @ B
    eps = rng.standard_t(4, size=n) * noise
    cps = (n * np.array([0.35, 0.65])).astype(int)
    C = np.zeros(n, int)
    C[cps[0]:cps[1]] = 1
    C[cps[1]:] = 2
    y = Th[np.arange(n), C] + eps
    return Xs, y / np.std(y), cps


Xs, y, cps = make()
est = D.dcdp_cv_grid_linear(
    40,
    [0.1],
    [1],
    smooth=2,
    buffer=2,
    step_refine=1,
    buffer_refine=2,
    lam_refine=0.1,
)
grid = np.floor(np.arange(1, 41) * (600 / 41)).astype(int)
print("true cps:", cps)
for tag, Y in (("1-D y (buggy)", y), ("2-D y (fixed)", y.reshape(-1, 1))):
    for gam in [25, 50, 100, 200]:
        cp, _ = est.dp_grid((Y, Xs), grid, 0.1, gam)
        print(f"  {tag:16s} gamma={gam:4d}: K={len(cp):3d} cp={cp[:6]}")
