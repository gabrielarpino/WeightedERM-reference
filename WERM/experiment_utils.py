import hashlib

import numpy as np

from WERM.performance_measures import hausdorff


def trial_rng(experiment, setting_idx, trial_idx):
    """Return a stable NumPy generator keyed by experiment task identity."""
    key = f"{experiment}|{int(setting_idx)}|{int(trial_idx)}".encode()
    seed = int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) % (2**32)
    return np.random.default_rng(seed)


def data_hash(X, y):
    """Hash the exact contiguous design/response bytes for one trial."""
    X_bytes = np.ascontiguousarray(X).tobytes()
    y_bytes = np.ascontiguousarray(y).tobytes()
    return hashlib.blake2b(X_bytes + y_bytes, digest_size=16).hexdigest()


def weighted_erm_joint_cv(
    X,
    y,
    M,
    M_scorer,
    Lmax,
    delta_fraction,
    l1_constants=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    num_folds=5,
    n_jobs=1,
    standardize_y=False,
):
    """Jointly select WERM segment count and a dimensionless L1 constant.

    This is an experiment-layer specialization of ``chgpt_k_fold_cv``. For a
    candidate ``(num_segments, C)``, the coefficient penalty is
    ``C * sqrt(2 log(p) / n_train) * ||beta||_1``. The scale is recomputed for
    every training fold and for the final full-data refit, and the same penalty
    is used when refitting segment coefficients for validation predictions.
    Validation scores contain prediction loss only. WERM's separate partition
    penalty stays at zero.

    If ``standardize_y`` is true, each fold divides both its training and held-
    out responses by the training-response standard deviation. The final refit
    analogously uses the full-data standard deviation. A zero or nonfinite
    standard deviation is replaced by one.
    """
    import jax
    import jax.numpy as jnp
    import scipy.optimize
    from joblib import Parallel, delayed

    import WERM.estimators

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim == 0 or y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have matching sample dimensions")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    if int(Lmax) < 1:
        raise ValueError("Lmax must be at least one segment")
    if not 0 < float(delta_fraction) < 1:
        raise ValueError("delta_fraction must lie strictly between zero and one")
    if int(num_folds) < 2:
        raise ValueError("num_folds must be at least two")
    if int(num_folds) > X.shape[0]:
        raise ValueError(
            f"Cannot create {int(num_folds)} non-empty folds with {X.shape[0]} samples"
        )

    constants = tuple(float(C) for C in l1_constants)
    if not constants or any(not np.isfinite(C) or C <= 0 for C in constants):
        raise ValueError("l1_constants must contain finite positive values")

    n_samples = X.shape[0]
    p = X.shape[1]
    candidates = [
        (num_segments, C)
        for num_segments in range(1, int(Lmax) + 1)
        for C in constants
    ]

    folds = [
        np.arange(v, n_samples, int(num_folds), dtype=int)
        for v in range(int(num_folds))
    ]

    def response_scale(values):
        if not standardize_y:
            return 1.0
        scale = float(np.std(values))
        return scale if np.isfinite(scale) and scale > 0.0 else 1.0

    def l1_scale(C, n_train):
        return C * np.sqrt(2.0 * np.log(p) / n_train)

    def fit_candidate(X_train, y_train, candidate):
        num_segments, C = candidate
        n_train = len(y_train)
        l1_lambda = l1_scale(C, n_train)
        P_func = lambda beta: l1_lambda * jnp.sum(jnp.abs(beta))
        return WERM.estimators.weighted_erm(
            X=X_train,
            y=y_train,
            M=M,
            L=int(num_segments),
            Delta_n=max(1, int(float(delta_fraction) * n_train)),
            gamma=0.0,
            L_is_upper_bound=False,
            return_B_hat=False,
            brute_force=False,  # production route is exact chgpt_gof_dp
            P_func=P_func,
        )

    if M_scorer is None:
        M_scorer = M

    def fit_segment_coefficients(
        train_indices, mapped_chgpts, y_train_scaled, l1_lambda
    ):
        all_chgpts = np.concatenate(
            ([0], np.asarray(mapped_chgpts, dtype=int), [n_samples])
        )
        segment_betas = []

        for start_idx, end_idx in zip(all_chgpts[:-1], all_chgpts[1:]):
            segment_train_indices = train_indices[
                (train_indices >= start_idx) & (train_indices < end_idx)
            ]
            if segment_train_indices.size == 0:
                segment_betas.append(np.zeros(p))
                continue

            X_segment = X[segment_train_indices]
            # y_train_scaled is ordered exactly as train_indices.
            segment_positions = np.searchsorted(train_indices, segment_train_indices)
            y_segment = y_train_scaled[segment_positions]

            def objective(beta):
                prediction = X_segment @ beta
                prediction_loss = jnp.mean(
                    M_scorer(prediction.flatten(), jnp.asarray(y_segment).flatten())
                )
                return prediction_loss + l1_lambda * jnp.sum(jnp.abs(beta))

            result = scipy.optimize.minimize(
                fun=objective,
                x0=0.1 * np.ones(p),
                jac=jax.grad(objective),
                method="L-BFGS-B",
                options={"disp": False, "maxiter": 1000},
            )
            segment_betas.append(np.asarray(result.x))

        return all_chgpts, segment_betas

    def evaluate_candidate(candidate):
        _, C = candidate
        cv_score = 0.0

        for fold_number, test_indices in enumerate(folds):
            train_indices = np.sort(
                np.concatenate(
                    [fold for i, fold in enumerate(folds) if i != fold_number]
                )
            )
            scale = response_scale(y[train_indices])
            y_train_scaled = y[train_indices] / scale
            train_chgpts = np.asarray(
                fit_candidate(X[train_indices], y_train_scaled, candidate), dtype=int
            ).reshape(-1)

            # Match chgpt_k_fold_cv: a training-position boundary maps to the
            # original index of the first training observation after it.
            mapped_chgpts = np.asarray(
                [
                    train_indices[chgpt]
                    for chgpt in train_chgpts
                    if 0 < chgpt < len(train_indices)
                ],
                dtype=int,
            )
            l1_lambda = l1_scale(C, len(train_indices))
            all_chgpts, segment_betas = fit_segment_coefficients(
                train_indices, mapped_chgpts, y_train_scaled, l1_lambda
            )

            y_test_scaled = y[test_indices] / scale
            for test_position, original_index in enumerate(test_indices):
                segment_idx = np.searchsorted(
                    all_chgpts[1:], original_index, side="right"
                )
                segment_idx = min(segment_idx, len(segment_betas) - 1)
                prediction = (
                    X[original_index:original_index + 1]
                    @ segment_betas[segment_idx]
                )
                loss = np.asarray(
                    M_scorer(
                        prediction,
                        y_test_scaled[test_position:test_position + 1],
                    )
                )
                if loss.shape != (1,):
                    raise ValueError(
                        "M_scorer must return shape (1,) for one validation observation"
                    )
                # Deliberately exclude the coefficient penalty from validation.
                cv_score += float(loss[0])

        return cv_score

    scores = Parallel(n_jobs=int(n_jobs))(
        delayed(evaluate_candidate)(candidate) for candidate in candidates
    )
    best_pair = candidates[int(np.argmin(scores))]

    full_scale = response_scale(y)
    cp = fit_candidate(X, y / full_scale, best_pair)
    best_segments, best_C = best_pair
    return (int(best_segments), float(best_C)), np.asarray(cp, dtype=int).reshape(-1)

def evaluate_changepoints(cp, true_chgpts, n):
    """Apply the common comparison metric and three allowed result statuses."""
    if cp is None:
        return {
            "raw_cp": None,
            "status": "not_applicable",
            "haus": np.nan,
            "size": np.nan,
        }

    raw_cp = np.asarray(cp, dtype=int).reshape(-1)
    if raw_cp.size == 0:
        return {
            "raw_cp": raw_cp,
            "status": "empty_estimate",
            "haus": 1.0,
            "size": 0,
        }

    return {
        "raw_cp": raw_cp,
        "status": "ok",
        "haus": float(hausdorff(raw_cp, true_chgpts)) / int(n),
        "size": int(raw_cp.size),
    }


def summarize_setting(haus, sizes, statuses):
    """Summarize one method/setting with empty estimates penalized in haus."""
    haus = np.asarray(haus, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    statuses = np.asarray(statuses, dtype=object)
    if not set(statuses.tolist()) <= {"ok", "empty_estimate", "not_applicable"}:
        raise ValueError("unexpected experiment status")
    return {
        "mean_haus": float(np.nanmean(haus)),
        "median_haus": float(np.nanmedian(haus)),
        "empty_estimate_rate": float(np.mean(statuses == "empty_estimate")),
        "not_applicable_rate": float(np.mean(statuses == "not_applicable")),
        "mean_size": float(np.nanmean(sizes)),
    }



def glm_bsa3_cv(
    X,
    y,
    kmax=2,
    delta=0.1,
    c1_candidates=(0.045, 0.09, 0.18, 0.36, 0.72),
    num_folds=5,
):
    """Select myglm3's c1 by interlaced held-out prediction loss.

    c1 is the only tuned quantity: kmax remains the automatic binary-
    segmentation pass budget and delta remains the fixed minimum-segment
    fraction. Every fold runs the unchanged myglm3 routine on training rows,
    maps its compressed-time boundaries back to the original timeline, refits
    segment coefficients with myglm3's exact glmnet penalty rule, and scores
    unpenalized logistic loss on held-out rows. The selected c1 is then refit
    once on the full data.
    """
    import WERM.comparison as comparison

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y must have matching sample dimensions")
    if int(kmax) < 1:
        raise ValueError("kmax must be positive")
    if not 0 < float(delta) < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if int(num_folds) < 2 or int(num_folds) > len(y):
        raise ValueError("num_folds must be between two and the sample size")

    candidates = tuple(float(c1) for c1 in c1_candidates)
    if not candidates or any(
        not np.isfinite(c1) or c1 <= 0 for c1 in candidates
    ):
        raise ValueError("c1_candidates must contain finite positive values")

    n = len(y)
    folds = [
        np.arange(v, n, int(num_folds), dtype=int)
        for v in range(int(num_folds))
    ]

    def clean_cp(cp, sample_size):
        if cp is None:
            return np.array([], dtype=int)
        cp = np.asarray(cp, dtype=int).reshape(-1)
        return np.unique(cp[(cp > 1) & (cp < int(sample_size))])

    def candidate_score(c1):
        score = 0.0
        try:
            for fold_number, test_indices in enumerate(folds):
                train_indices = np.sort(
                    np.concatenate(
                        [fold for i, fold in enumerate(folds) if i != fold_number]
                    )
                )
                compressed_cp = clean_cp(
                    comparison.glm_bsa3(
                        X[train_indices],
                        y[train_indices].reshape(-1, 1),
                        kmax=int(kmax),
                        c1=c1,
                        m1=None,
                        delta=float(delta),
                        exact=False,
                        quiet=True,
                    ),
                    len(train_indices),
                )
                mapped_cp = np.asarray(
                    [train_indices[cp] for cp in compressed_cp], dtype=int
                )
                boundaries = np.concatenate(([0], mapped_cp, [n]))

                assigned = np.zeros(test_indices.size, dtype=bool)
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    segment_train = train_indices[
                        (train_indices >= start) & (train_indices < end)
                    ]
                    segment_test_mask = (
                        (test_indices >= start) & (test_indices < end)
                    )
                    if segment_train.size == 0:
                        raise ValueError("GLMBSA-CV produced an empty training segment")
                    beta = comparison.glm_bsa3_segment_beta(
                        X[segment_train], y[segment_train], c1
                    )
                    if np.any(segment_test_mask):
                        eta = X[test_indices[segment_test_mask]] @ beta
                        y_test = y[test_indices[segment_test_mask]]
                        score += float(
                            np.sum(np.logaddexp(0.0, eta) - y_test * eta)
                        )
                        assigned[segment_test_mask] = True
                if not np.all(assigned):
                    raise ValueError("GLMBSA-CV left held-out observations unscored")
        except Exception as exc:
            print(f"GLMBSA-CV c1={c1:g} failed: {exc}")
            return np.inf
        return score

    scores = np.asarray([candidate_score(c1) for c1 in candidates], dtype=float)
    if not np.any(np.isfinite(scores)):
        raise RuntimeError("all GLMBSA-CV candidates failed")
    best_index = int(np.argmin(scores))
    best_c1 = candidates[best_index]
    full_cp = clean_cp(
        comparison.glm_bsa3(
            X,
            y.reshape(-1, 1),
            kmax=int(kmax),
            c1=best_c1,
            m1=None,
            delta=float(delta),
            exact=False,
            quiet=True,
        ),
        n,
    )
    return best_c1, full_cp, scores
