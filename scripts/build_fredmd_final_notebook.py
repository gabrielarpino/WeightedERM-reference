"""Build the concise final FRED-MD analysis notebook."""

from textwrap import dedent

import nbformat as nbf


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        r"""
        # FRED-MD Final

        Huber loss (1.35), two-fold full joint CV, $\Delta=n/20$, no intercept, and at most four change points. CV scores held-out predictions with absolute loss; segmentation and segment refits use Huber loss.
        """
    ),
    code(
        r"""
        import os
        import pickle
        from contextlib import redirect_stderr, redirect_stdout
        from pathlib import Path

        import jax.numpy as jnp
        import numpy as np
        import pandas as pd

        import WERM.estimators
        from WERM.experiment_utils import weighted_erm_joint_cv

        N_JOBS = min(36, os.cpu_count() or 1)
        SEGMENTS = (1, 2, 3, 4, 5)
        CONSTANTS = (0.03, 0.30, 3.00)
        DELTA_FRACTION = 1 / 20
        NUM_FOLDS = 2

        def huber(prediction, truth, epsilon=1.35):
            residual = prediction - truth
            absolute = jnp.abs(residual)
            return jnp.where(
                absolute <= epsilon,
                0.5 * residual**2,
                epsilon * (absolute - 0.5 * epsilon),
            )

        def absolute(prediction, truth):
            return jnp.abs(prediction - truth)

        def quietly(function):
            with open(os.devnull, "w") as sink:
                with redirect_stdout(sink), redirect_stderr(sink):
                    return function()

        with (Path("FRED-MD data") / "fredmd_data.pkl").open("rb") as handle:
            fred = pickle.load(handle)

        X = np.asarray(fred["X"], dtype=float)
        y = np.asarray(fred["y"], dtype=float).reshape(-1)
        dates = np.asarray(fred["date"]).astype(str)
        assert X.shape == (773, 114)
        assert dates[[0, -1]].tolist() == ["1960-01", "2024-05"]
        """
    ),
    code(
        r"""
        # Huber + L1: exhaustive joint CV over penalty constant and segments
        selected, change_points, diagnostics = quietly(lambda: weighted_erm_joint_cv(
            X=X,
            y=y,
            M=huber,
            M_scorer=absolute,
            Lmax=5,
            delta_fraction=DELTA_FRACTION,
            l1_constants=CONSTANTS,
            num_folds=NUM_FOLDS,
            n_jobs=N_JOBS,
            standardize_y=False,
            selection_strategy="exhaustive",
            return_diagnostics=True,
            penalty_norm="l1",
            segment_counts=SEGMENTS,
        ))
        change_points = np.asarray(change_points, dtype=int)
        pd.DataFrame([{
            "method": "Huber + L1, full joint CV",
            "C": selected[1],
            "num_change_points": len(change_points),
            "change_point_dates": "; ".join(dates[change_points]),
        }])
        """
    ),
    markdown("## Sensitivity analysis"),
    code(
        r"""
        # Unpenalized Huber: full CV
        _, cv_change_points = quietly(lambda: weighted_erm_joint_cv(
            X=X,
            y=y,
            M=huber,
            M_scorer=absolute,
            Lmax=5,
            delta_fraction=DELTA_FRACTION,
            num_folds=NUM_FOLDS,
            n_jobs=N_JOBS,
            standardize_y=False,
            selection_strategy="successive_halving",
            penalty_norm="none",
            segment_counts=SEGMENTS,
        ))
        cv_change_points = np.asarray(cv_change_points, dtype=int)

        # Unpenalized Huber: original slope-based elbow
        minimum_length = max(int(np.ceil(len(y) / 50)), int(DELTA_FRACTION * len(y)), 1)
        curve = []
        for segments in SEGMENTS:
            cp, gof = quietly(lambda: WERM.estimators.weighted_erm(
                X=X,
                y=y,
                M=huber,
                L=segments,
                Delta_n=minimum_length,
                gamma=0.0,
                brute_force=False,
                return_gof=True,
                P_func=lambda beta: jnp.array(0.0),
            ))
            curve.append((np.asarray(cp, dtype=int), float(gof)))

        gofs = np.asarray([gof for _, gof in curve])
        average_slopes = np.asarray([
            ((gofs[i] - gofs[i - 1]) + (gofs[i + 1] - gofs[i])) / 2
            for i in range(1, len(gofs) - 1)
        ])
        elbow_segments = len(SEGMENTS)
        for j in range(1, len(average_slopes)):
            if abs(average_slopes[j]) < abs(average_slopes[j - 1]):
                elbow_segments = j + 2
                break
        elbow_change_points = curve[elbow_segments - 1][0]

        pd.DataFrame([
            {
                "method": "Unpenalized Huber, full CV",
                "num_change_points": len(cv_change_points),
                "change_point_dates": "; ".join(dates[cv_change_points]),
            },
            {
                "method": "Unpenalized Huber, original elbow",
                "num_change_points": len(elbow_change_points),
                "change_point_dates": "; ".join(dates[elbow_change_points]),
            },
        ])
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)

with open("FRED MD Final.ipynb", "w", encoding="utf-8") as handle:
    nbf.write(notebook, handle)
