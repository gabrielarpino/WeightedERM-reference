import numpy as np

from WERM.proximal import fit_weighted_l1


def test_fista_acceleration_converges_on_underdetermined_low_lambda_huber_fit():
    """Regression for the small-segment refits used by M1 joint CV."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(103, 200))
    beta = np.zeros(200)
    beta[:20] = rng.normal(size=20)
    y = X @ beta + 0.5 * rng.standard_t(4, size=X.shape[0])

    result = fit_weighted_l1(
        X,
        y,
        np.ones(X.shape[0]),
        loss="huber",
        l1=0.0069,
        tol=1e-6,
        max_iter=2000,
    )

    assert result.converged
    assert result.kkt_violation <= 1e-6
