"""Build the executed FRED-MD period/intercept WERM-CV report notebook."""

from textwrap import dedent

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        r"""
        # FRED-MD period and intercept CV exploration

        This notebook reports the completed 352-surface sensitivity experiment for
        WERM on FRED-MD. It compares the DPDU window, the requested 1970--2024
        McScan-style window, the latest official vintage, and the original full
        1960--2024 sample. Every dataset is run with and without a constant column.

        The corrected loss contract is fixed throughout:

        - the base loss (least squares or Huber) is used for segmentation;
        - the same base loss is used for every hard within-segment coefficient refit;
        - absolute loss is used **only** on held-out observations for CV scoring;
        - when a constant column is present, it is excluded from L1/L2 penalties;
        - `Lmax=5` means candidates of 1--5 segments, hence 0--4 change points.

        The full numerical output is checkpointed under
        `results/FRED-MD_period-intercept-CV/`. Re-running the expensive search is
        optional; this report reads those completed artifacts.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import json

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        ROOT = Path.cwd()
        RESULT_DIR = ROOT / "results" / "FRED-MD_period-intercept-CV"
        with (RESULT_DIR / "summary.json").open() as handle:
            summary = json.load(handle)

        all_results = pd.read_csv(RESULT_DIR / "all_selections.csv")
        nonzero = pd.read_csv(RESULT_DIR / "nonzero_selections.csv")
        target = pd.read_csv(RESULT_DIR / "two_or_three_change_points.csv")
        joint_epsilon = pd.read_csv(
            RESULT_DIR / "joint_unpenalized_huber_epsilon_selections.csv"
        )
        HIGH_DELTA_DIR = (
            ROOT / "results" / "FRED-MD_high-delta-unpenalized-CV-vs-elbow"
        )
        high_delta = pd.read_csv(HIGH_DELTA_DIR / "all_selections.csv")
        high_delta_joint = pd.read_csv(
            HIGH_DELTA_DIR / "joint_huber_epsilon_selections.csv"
        )
        LOCAL_DELTA_DIR = (
            ROOT / "results" / "FRED-MD_original-Huber-L1-local-delta-elbow"
        )
        local_delta = pd.read_csv(LOCAL_DELTA_DIR / "selections.csv")

        assert len(summary["records"]) == 352
        assert len(all_results) == 352
        print({
            "completed_surfaces": len(all_results),
            "nonzero_selections": len(nonzero),
            "two_or_three_cp_selections": len(target),
            "max_parallel_jobs": summary["grid"]["n_jobs"],
        })
        """
    ),
    md(
        r"""
        ## Data and reference-protocol audit

        The historical pickle is the bundled McScan-prepared design: FRED
        transformation codes, per-series standardization, 99.9% truncation, and six
        principal components plus the idiosyncratic components. The Python
        reproduction in `scripts/fredmd_preprocessing.py` was checked against it to
        machine precision (up to arbitrary PCA signs).

        The latest-vintage design uses the same pipeline on the official July 2026
        FRED-MD download. Its last usable predictor month is May 2026 because the
        design predicts next-month industrial production. Exact comparisons across
        vintages are not pure endpoint comparisons: revisions and the balanced
        feature set also change (114 historical predictors versus 92 latest-vintage
        predictors).

        There is also an important naming distinction: the requested “McScan period”
        below is 1970-01--2024-01. The bundled McScan paper's final real-data analysis
        actually uses 1960-01--2024-05, included here as the audit dataset.
        """
    ),
    code(
        r"""
        dataset_rows = []
        for name, values in summary["datasets"].items():
            dataset_rows.append({"dataset": name, **values})
        datasets = pd.DataFrame(dataset_rows)
        display(datasets[["dataset", "start", "end", "n", "p"]])
        display(pd.Series(summary["official_vintage"], name="latest-vintage metadata"))
        """
    ),
    md(
        r"""
        ## Search grid

        For each dataset/intercept pair, the search covers two and five interlaced
        folds, minimum segment fractions `n/50` and `n/30`, unpenalized least
        squares, unpenalized Huber with epsilon in
        `{0.5, 0.75, 1, 1.35, 2, 3}`, and L1/L2-penalized least squares or
        Huber (`epsilon=1.35`) with constants
        `{0.15, 0.3, 0.5, 0.75, 1, 2}`. Coefficient constants and the segment count
        are selected jointly for penalized fits. An additional summary selects
        epsilon and segment count jointly for unpenalized Huber.
        """
    ),
    code(
        r"""
        display(pd.Series(summary["loss_contract"], name="CV contract"))
        display(pd.Series({
            "coefficient constants": summary["grid"]["constants"],
            "segment counts": summary["grid"]["segment_counts"],
            "Huber epsilons": summary["grid"]["unpenalized_huber_epsilons"],
            "sensitivities": summary["grid"]["sensitivities"],
        }, name="values"))
        """
    ),
    md(
        r"""
        ## Overall selection behavior

        The full grid contains many zero selections, especially with five folds.
        This makes fold sensitivity part of the scientific result rather than a
        nuisance to hide. The table and plot below expose the complete count
        distribution by dataset and intercept choice.
        """
    ),
    code(
        r"""
        count_table = (
            all_results.groupby(["dataset", "intercept", "selected_change_points"])
            .size().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
        )
        count_table.columns = [f"{value} CP" for value in count_table.columns]
        display(count_table)

        count_table.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="viridis")
        plt.ylabel("number of CV surfaces")
        plt.xlabel("dataset and intercept")
        plt.title("Selected WERM change-point counts across the complete grid")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ## Every 2- or 3-change-point candidate

        These are all 22 target-count selections, without filtering on location.
        `cv_mae` is the held-out absolute-error score. The improvement column compares
        the selected result with the best zero-change-point candidate on the same
        loss/penalty/fold/delta surface. A positive but small margin should be treated
        as fragile.
        """
    ),
    code(
        r"""
        target_display = target.copy()
        target_display["delta"] = target_display["delta_fraction"].map(
            lambda value: "n/50" if np.isclose(value, 1/50) else "n/30"
        )
        target_display["cv_mae"] = target_display["cv_mae"].round(6)
        target_display["zero_cp_cv_mae"] = target_display["zero_cp_cv_mae"].round(6)
        target_display["improvement_vs_zero_pct"] = target_display[
            "improvement_vs_zero_pct"
        ].round(3)
        display(target_display[[
            "dataset", "intercept", "loss", "penalty", "folds", "delta",
            "selected_constant", "selected_change_points", "estimated_dates",
            "cv_mae", "zero_cp_cv_mae", "improvement_vs_zero_pct",
        ]].reset_index(drop=True))
        """
    ),
    md(
        r"""
        ## Best defensible candidates

        The most useful result is in the latest-vintage 1970--2026 sample. With a
        properly unpenalized intercept, two-fold LS-L1 selects two changes under both
        spacing choices with a 2.59% margin. Huber-L1 also selects two changes at
        `n/50`, and its later date (2010-11) is closer to McScan's 2010-05 estimate.
        Without an intercept, Huber-L1 is exactly stable across both spacings and its
        later date is 2009-06, closer to the DPDU financial-crisis change.

        None of those current-vintage candidates survives the switch to five folds;
        five-fold CV chooses zero. Thus the honest conclusion is that WERM-CV can
        select two or three changes, but the evidence is sensitive to the validation
        split rather than uniformly stable.
        """
    ),
    code(
        r"""
        preferred_mask = (
            target["dataset"].eq("Our_latest_1970-01_to_2026-05")
            & (
                (target["loss"].eq("least-squares") & target["penalty"].eq("l1"))
                | (target["loss"].eq("Huber-eps1.35") & target["penalty"].eq("l1"))
            )
        )
        preferred = target.loc[preferred_mask].sort_values(
            ["intercept", "loss", "delta_fraction"]
        )
        display(preferred.reset_index(drop=True))
        """
    ),
    md(
        r"""
        ## Joint unpenalized Huber epsilon and count

        Sweeping epsilon is sensible only if it is treated as another tuning
        parameter and selected inside the same CV comparison. On the latest and
        McScan-style long samples, joint unpenalized Huber always selects zero. On the
        shorter DPDU window it selects four changes with two folds and zero with five
        folds. Individual fixed-epsilon surfaces can select two or three changes, but
        they are not the joint epsilon/count winners and should not be reported as if
        they were.
        """
    ),
    code(
        r"""
        joint_display = joint_epsilon.copy()
        joint_display["delta"] = joint_display["delta_fraction"].map(
            lambda value: "n/50" if np.isclose(value, 1/50) else "n/30"
        )
        display(joint_display[[
            "dataset", "intercept", "folds", "delta", "selected_epsilon",
            "selected_change_points", "estimated_dates", "cv_mae",
            "improvement_vs_epsilon_specific_zero_pct",
        ]].reset_index(drop=True))
        """
    ),
    md(
        r"""
        ## Comparison with published dates and practical recommendation

        Reference dates used for interpretation:

        - McScan final analysis: 1974-02, 2010-05, 2019-05.
        - DPDU 2000--2022 analysis: 2008-06 and 2020-01.

        Recommended primary WERM sensitivity: latest official vintage, leading
        unpenalized intercept, Huber loss with epsilon 1.35, L1 constant selected from
        the stated grid, two interlaced folds, `Delta=n/50`, and 0--4 candidate change
        points. It selects two changes at **1996-03 and 2010-11**, with a 2.42% CV-MAE
        improvement over zero changes. The 2010 date is comparable to McScan's
        2010-05 finding; the earlier date is not a close published match.

        As a count-stability sensitivity, latest-vintage LS-L1 with an intercept
        selects two changes for both `n/50` and `n/30` (1996-03 and approximately
        2015), with a 2.59% margin. Huber-L1 without an intercept also selects two for
        both spacings (1997-05 and 2009-06), with a 2.21% margin. These alternatives
        support the count but show location sensitivity.

        Avoid choosing epsilon, folds, or spacing solely because a desired count
        appears. Pre-specify the primary configuration and disclose the five-fold zero
        result. A larger independent forecast period or repeated blocked/interlaced
        splits would be the strongest next robustness check.
        """
    ),
    code(
        r"""
        # Compact machine-checkable recommendation record.
        recommendation = target.loc[
            target["dataset"].eq("Our_latest_1970-01_to_2026-05")
            & target["intercept"].eq(True)
            & target["loss"].eq("Huber-eps1.35")
            & target["penalty"].eq("l1")
            & target["folds"].eq(2)
            & np.isclose(target["delta_fraction"], 1/50)
        ].iloc[0]
        display(recommendation.to_frame("recommended WERM-CV setting"))
        """
    ),
    md(
        r"""
        ## Higher minimum-spacing comparison: CV versus the existing elbow

        A second completed experiment applies unpenalized least squares and all six
        Huber constants to the three requested period variants with
        `Delta` in `{n/20, n/10, n/5}`. It reproduces the original notebook's
        fold-independent GOF elbow rule exactly.

        Out of 252 surfaces, CV selects a nonzero count in 42. Every one is on the
        DPDU sample with two folds; all five-fold and both long-sample selections are
        zero. The elbow is nonzero for those 42 DPDU cases, but there are **no exact
        count matches**: CV chooses three or four changes while the elbow chooses two.
        `n/5` is the closest internal agreement because every two-fold DPDU CV fit
        chooses three changes versus the elbow's two, with the smallest location
        Hausdorff discrepancies around 29 months.

        This is not a strong published-date match. On the DPDU sample, `n/5` is about
        55 months, so a January 2020 change cannot be selected: only about 36 months
        remain before December 2022. Thus the apparently cleaner `n/5` agreement is
        partly imposed by an overly restrictive endpoint constraint.
        """
    ),
    code(
        r"""
        high_delta_overview = high_delta.groupby("dataset").agg(
            surfaces=("key", "size"),
            cv_nonzero=("cv_selected_change_points", lambda values: (values > 0).sum()),
            both_nonzero=("both_nonzero", "sum"),
            exact_nonzero_count_matches=(
                "count_match",
                lambda values: (
                    values & high_delta.loc[values.index, "both_nonzero"]
                ).sum(),
            ),
        )
        display(high_delta_overview)

        dpdu_nonzero = high_delta.loc[
            high_delta["dataset"].eq("DPDU_2000-01_to_2022-12")
            & high_delta["both_nonzero"]
        ]
        pattern = dpdu_nonzero.groupby(
            ["folds", "delta_fraction", "intercept"]
        ).agg(
            settings=("key", "size"),
            cv_counts=("cv_selected_change_points", lambda x: sorted(set(x))),
            elbow_counts=("elbow_selected_change_points", lambda x: sorted(set(x))),
            best_cv_gain_pct=("cv_improvement_vs_zero_pct", "max"),
            minimum_hausdorff_months=("location_hausdorff_months", "min"),
        )
        display(pattern)
        """
    ),
    md(
        r"""
        The table below shows the jointly selected Huber epsilon at each high-Delta
        setting. Joint Huber selection is nonzero only for two-fold DPDU fits. These
        are useful sensitivity results, but none can be described as CV/elbow count
        agreement.
        """
    ),
    code(
        r"""
        display(high_delta_joint.reset_index(drop=True))
        """
    ),
    md(
        r"""
        ## Original-sample Huber-L1 local spacing rerun

        The saved `n/30`, Huber 1.35, L1, no-intercept, two-fold result selected
        three changes with `C=.30`. A focused rerun at `n/30`, `n/28`, `n/25`,
        `n/22`, and `n/20` shows that this CV winner is not numerically stable.
        The `n/30` rerun selects zero, while only `n/28` selects three changes—and
        its advantage over zero is just 0.055%.

        The full-data `L=4,C=.30` solution itself still returns the original
        1972-11, 2009-06, and 2011-07 dates at `n/30`; what changes is the held-out
        score assigned to that candidate. The established conditional-C elbow and a
        fixed-C=.30 elbow both select one change at `n/30`, so neither supports a
        three-change count.
        """
    ),
    code(
        r"""
        display(local_delta.reset_index(drop=True))
        """
    ),
    md(
        r"""
        ## Reproduction

        The expensive search is checkpointed after each surface and caps parallelism
        at 36 jobs:

        ```bash
        PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
          MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          .venv-phase1/bin/python scripts/run_fredmd_period_intercept_cv.py
        ```

        Existing checkpoint records are skipped. Delete or relocate the result
        directory only if a completely fresh run is intentionally required.
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

output = "FRED-MD Period and Intercept CV.ipynb"
nbf.write(notebook, output)
print(output)
