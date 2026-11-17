"""Build the exhaustive FRED-MD joint-CV search notebook."""

from textwrap import dedent

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        r"""
        # FRED-MD full joint-CV search

        This notebook uses the package-level `weighted_erm_joint_cv` after its loss
        semantics were corrected. For every configuration below, WERM segmentation
        and hard segment refits use the base loss M. Absolute loss is used only for
        held-out CV scoring.

        Every surface is exhaustive over both the coefficient-penalty constant and
        0--4 candidate change points. Loss, penalty norm, fold count, and minimum
        spacing are treated as sensitivity configurations; they are declared before
        looking at results. All outcomes are retained, including configurations that
        select zero, one, or four change points.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import hashlib
        import itertools
        import json
        import os
        import pickle
        import platform
        import sys
        import time

        import jax
        import jax.numpy as jnp
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from WERM.experiment_utils import weighted_erm_joint_cv

        ROOT = Path.cwd()
        DATA_PATH = ROOT / "FRED-MD data" / "fredmd_data.pkl"
        RESULT_DIR = ROOT / "results" / "FRED-MD_full-joint-cv-search"
        FIGURE_DIR = ROOT / "figures" / "FRED-MD_full-joint-cv-search"
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT = RESULT_DIR / "checkpoint.json"
        SUMMARY_PATH = RESULT_DIR / "summary.json"
        N_JOBS = min(36, os.cpu_count() or 1)

        print({
            "python": sys.version.split()[0], "platform": platform.platform(),
            "jax": jax.__version__, "n_jobs": N_JOBS,
        })
        """
    ),
    code(
        r"""
        with DATA_PATH.open("rb") as handle:
            fred = pickle.load(handle)

        X = np.asarray(fred["X"], dtype=float)
        y = np.asarray(fred["y"], dtype=float).reshape(-1)
        dates = np.asarray(fred["date"]).astype(str).reshape(-1)
        n, p = X.shape
        assert X.shape == (773, 114) and y.shape == (773,)

        DATA_HASH = hashlib.blake2b(
            np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes(),
            digest_size=16,
        ).hexdigest()
        REFERENCE_DATES = {
            "McScan": ["1974-02", "2010-05", "2019-05"],
            "DPDU_different_window": ["2008-06", "2020-01"],
        }
        REFERENCE_INDICES = {
            name: [int(np.flatnonzero(dates == value)[0]) for value in values]
            for name, values in REFERENCE_DATES.items()
        }

        def squared_loss(prediction, truth):
            return jnp.square(prediction - truth)

        def absolute_loss(prediction, truth):
            return jnp.abs(prediction - truth)

        def make_huber(epsilon):
            epsilon = float(epsilon)
            def huber_loss(prediction, truth):
                residual = prediction - truth
                absolute = jnp.abs(residual)
                return jnp.where(
                    absolute <= epsilon,
                    0.5 * jnp.square(residual),
                    epsilon * (absolute - 0.5 * epsilon),
                )
            return huber_loss

        LOSSES = {
            "least-squares": squared_loss,
            "Huber-eps0.5": make_huber(0.5),
            "Huber-eps0.75": make_huber(0.75),
            "Huber-eps1": make_huber(1.0),
            "Huber-eps1.35": make_huber(1.35),
            "Huber-eps2": make_huber(2.0),
        }
        """
    ),
    md(
        r"""
        ## Predeclared exhaustive grid

        The constant grid spans weak through strong regularization and is shared by
        L1 and L2 for a transparent comparison. Each surface contains 30 candidates:
        five segment counts times six constants. Two-fold CV is evaluated across a
        broad spacing grid; three- and five-fold CV are evaluated at the central
        spacing choice to check fold sensitivity. Huber epsilon 1.35 is part of that
        primary grid. Epsilon values 0.5, 0.75, 1.0, and 2.0 are predeclared
        sensitivities at two folds and the two central spacing choices. The response
        standard deviation is about 0.88, so these thresholds span strongly robust
        through nearly least-squares behavior on the actual fitted scale.
        """
    ),
    code(
        r"""
        CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
        SEGMENT_COUNTS = (1, 2, 3, 4, 5)  # 0--4 change points
        SEARCH_CONFIGS = []
        for loss_name, penalty_norm in itertools.product(
            ("least-squares", "Huber-eps1.35"), ("l1", "l2")
        ):
            for delta_fraction in (1/50, 1/40, 1/30, 1/25, 1/20):
                SEARCH_CONFIGS.append({
                    "loss": loss_name, "penalty_norm": penalty_norm,
                    "num_folds": 2, "delta_fraction": delta_fraction,
                })
            for num_folds in (3, 5):
                SEARCH_CONFIGS.append({
                    "loss": loss_name, "penalty_norm": penalty_norm,
                    "num_folds": num_folds, "delta_fraction": 1/40,
                })

        for loss_name, penalty_norm, delta_fraction in itertools.product(
            ("Huber-eps0.5", "Huber-eps0.75", "Huber-eps1", "Huber-eps2"),
            ("l1", "l2"), (1/40, 1/30),
        ):
            SEARCH_CONFIGS.append({
                "loss": loss_name, "penalty_norm": penalty_norm,
                "num_folds": 2, "delta_fraction": delta_fraction,
            })

        assert len(SEARCH_CONFIGS) == 44
        pd.DataFrame(SEARCH_CONFIGS)
        """
    ),
    code(
        r"""
        def config_key(config):
            return "|".join([
                config["loss"], config["penalty_norm"],
                str(config["num_folds"]), f'{config["delta_fraction"]:.12g}',
            ])

        def alignment(cp):
            cp = tuple(int(value) for value in cp)
            candidates = []
            if len(cp) == 3:
                target = REFERENCE_INDICES["McScan"]
                distances = [abs(a - b) for a, b in zip(cp, target)]
                candidates.append(("McScan", distances))
            if len(cp) == 2:
                target = REFERENCE_INDICES["DPDU_different_window"]
                candidates.append(("DPDU_different_window", [abs(a - b) for a, b in zip(cp, target)]))
                for subset in itertools.combinations(REFERENCE_INDICES["McScan"], 2):
                    candidates.append(("McScan_best_2_of_3", [abs(a - b) for a, b in zip(cp, subset)]))
            if not candidates:
                return {"reference": None, "distances_months": [], "mean_months": None, "max_months": None}
            reference, distances = min(candidates, key=lambda item: (np.mean(item[1]), max(item[1])))
            return {
                "reference": reference,
                "distances_months": [int(value) for value in distances],
                "mean_months": float(np.mean(distances)),
                "max_months": int(max(distances)),
            }

        if CHECKPOINT.exists():
            with CHECKPOINT.open() as handle:
                records = json.load(handle)
        else:
            records = []
        completed = {record["config_key"] for record in records}
        print(f"Resuming with {len(records)} of {len(SEARCH_CONFIGS)} configurations complete")
        """
    ),
    code(
        r"""
        for setting_index, config in enumerate(SEARCH_CONFIGS, start=1):
            key = config_key(config)
            if key in completed:
                continue
            print(f'[{setting_index}/{len(SEARCH_CONFIGS)}] {key}', flush=True)
            started = time.perf_counter()
            selected, cp, diagnostics = weighted_erm_joint_cv(
                X=X, y=y, M=LOSSES[config["loss"]], M_scorer=absolute_loss,
                Lmax=max(SEGMENT_COUNTS), delta_fraction=config["delta_fraction"],
                l1_constants=CONSTANTS, num_folds=config["num_folds"],
                n_jobs=N_JOBS, standardize_y=False,
                selection_strategy="exhaustive", return_diagnostics=True,
                penalty_norm=config["penalty_norm"], segment_counts=SEGMENT_COUNTS,
            )
            candidates = [tuple(value) for value in diagnostics["candidates"]]
            scores = np.asarray(diagnostics["scores"], dtype=float)
            chosen_index = candidates.index(tuple(selected))
            baseline = min(
                score for candidate, score in zip(candidates, scores)
                if candidate[0] == 1 and np.isfinite(score)
            )
            selected_score = float(scores[chosen_index])
            cp = np.asarray(cp, dtype=int).reshape(-1)
            record = {
                **config, "config_key": key,
                "constants": list(CONSTANTS), "segment_counts": list(SEGMENT_COUNTS),
                "score_loss": "absolute_loss", "refit_loss": config["loss"],
                "selected_segments": int(selected[0]),
                "selected_change_points": int(selected[0] - 1),
                "selected_constant": float(selected[1]),
                "estimated_change_points": cp.tolist(),
                "estimated_dates": dates[cp].tolist(),
                "selected_cv_mean": selected_score / n,
                "best_zero_cp_cv_mean": float(baseline / n),
                "relative_improvement_vs_zero_pct": float((baseline - selected_score) / baseline * 100),
                "alignment": alignment(cp),
                "candidate_scores": [
                    {"segments": int(candidate[0]), "change_points": int(candidate[0] - 1),
                     "constant": float(candidate[1]), "cv_total": float(score), "cv_mean": float(score / n)}
                    for candidate, score in zip(candidates, scores)
                ],
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            completed.add(key)
            with CHECKPOINT.open("w") as handle:
                json.dump(records, handle, indent=2)
            print({k: record[k] for k in (
                "selected_change_points", "selected_constant", "estimated_dates",
                "relative_improvement_vs_zero_pct", "elapsed_seconds",
            )}, flush=True)

        assert len(records) == len(SEARCH_CONFIGS)
        """
    ),
    code(
        r"""
        result_table = pd.DataFrame([{
            "loss": record["loss"], "penalty": record["penalty_norm"],
            "folds": record["num_folds"], "delta": record["delta_fraction"],
            "selected CPs": record["selected_change_points"],
            "C": record["selected_constant"], "dates": ";".join(record["estimated_dates"]),
            "CV improvement vs zero (%)": record["relative_improvement_vs_zero_pct"],
            "reference": record["alignment"]["reference"],
            "mean reference distance (months)": record["alignment"]["mean_months"],
            "max reference distance (months)": record["alignment"]["max_months"],
        } for record in records])

        target_table = result_table[result_table["selected CPs"].isin([2, 3])].copy()
        target_table = target_table.sort_values(
            ["mean reference distance (months)", "CV improvement vs zero (%)"],
            ascending=[True, False], na_position="last",
        )
        target_table.head(20)
        """
    ),
    code(
        r"""
        summary = {
            "notebook": "FRED-MD Full CV Search.ipynb",
            "data_hash": DATA_HASH,
            "loss_contract": {
                "segmentation": "base loss M",
                "hard_segment_refits": "base loss M",
                "heldout_scoring_only": "absolute_loss",
            },
            "grid": {
                "constants": list(CONSTANTS), "segment_counts": list(SEGMENT_COUNTS),
                "configuration_count": len(SEARCH_CONFIGS),
                "selection_strategy": "exhaustive",
            },
            "reference_dates": REFERENCE_DATES,
            "records": records,
            "target_records": [
                record for record in records if record["selected_change_points"] in (2, 3)
            ],
        }
        with SUMMARY_PATH.open("w") as handle:
            json.dump(summary, handle, indent=2)
        result_table.to_csv(RESULT_DIR / "all_configurations.csv", index=False)
        target_table.to_csv(RESULT_DIR / "two_or_three_change_points.csv", index=False)

        counts = result_table.groupby(["loss", "penalty", "selected CPs"]).size().unstack(fill_value=0)
        display(counts)
        print(f"Found {len(target_table)} configurations selecting 2 or 3 change points")
        display(target_table.head(20))
        """
    ),
    code(
        r"""
        if len(target_table):
            best_row = target_table.iloc[0]
            best_record = next(
                record for record in records
                if record["loss"] == best_row["loss"]
                and record["penalty_norm"] == best_row["penalty"]
                and record["num_folds"] == int(best_row["folds"])
                and np.isclose(record["delta_fraction"], best_row["delta"])
            )
            print("BEST PREDECLARED REFERENCE-ALIGNED 2/3-CP CONFIGURATION")
            print(json.dumps({key: best_record[key] for key in (
                "loss", "penalty_norm", "num_folds", "delta_fraction",
                "selected_change_points", "selected_constant",
                "estimated_change_points", "estimated_dates",
                "selected_cv_mean", "best_zero_cp_cv_mean",
                "relative_improvement_vs_zero_pct", "alignment",
            )}, indent=2))

            figure, axis = plt.subplots(figsize=(13, 5))
            axis.plot(pd.to_datetime(dates), y, color="0.4", linewidth=1.0, label="INDPRO response")
            for index, value in enumerate(best_record["estimated_change_points"]):
                axis.axvline(pd.Timestamp(str(dates[value])), color="#D55E00", linewidth=2.3,
                             label="WERM full CV" if index == 0 else None)
            for name, color, style in (("McScan", "#0072B2", "--"), ("DPDU_different_window", "#009E73", ":")):
                for index, value in enumerate(REFERENCE_INDICES[name]):
                    axis.axvline(pd.Timestamp(str(dates[value])), color=color, linestyle=style, alpha=0.75,
                                 label=name if index == 0 else None)
            axis.set_title("Best predeclared 2/3-change-point WERM full-CV result")
            axis.grid(alpha=0.2)
            axis.legend(ncol=3)
            figure.tight_layout()
            figure.savefig(FIGURE_DIR / "best_reference_aligned_result.png", dpi=200, bbox_inches="tight")
            plt.show()
        else:
            print("No predeclared configuration selected exactly 2 or 3 change points.")
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "weighted-erm", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)

with open("FRED-MD Full CV Search.ipynb", "w", encoding="utf-8") as handle:
    nbf.write(notebook, handle)
