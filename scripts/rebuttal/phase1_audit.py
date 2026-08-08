"""Generate the compact Phase 1 exact-DP audit outputs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import os
import re
import time
from pathlib import Path

# Bound inner numerical-library threads before importing NumPy/JAX.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from WERM.gof import (
    chgpt_binary_segmentation,
    chgpt_gof_bf_mem_eff,
    chgpt_gof_dp,
    legacy_gamma_k_penalty,
    manuscript_log_binomial_k_penalty,
    table_k_penalty,
    zero_k_penalty,
)


TOLERANCE = 1e-9
CASE_COLUMNS = [
    "case_id", "source", "n", "Lmax", "K_mode", "Delta", "penalty_id",
    "dp_K", "brute_K", "greedy_K", "dp_changepoints",
    "brute_changepoints", "greedy_changepoints", "dp_raw", "brute_raw",
    "greedy_raw", "dp_total", "brute_total", "greedy_total",
    "dp_vs_brute_abs_diff", "greedy_minus_dp", "status", "note",
]
PALETTE = {
    "dp": "#0072B2",
    "brute": "#D55E00",
    "greedy": "#009E73",
    "penalty": "#CC79A7",
    "total": "#E69F00",
}


def squared_loss(prediction, outcome):
    return jnp.square(prediction - outcome)


def effective_workers(requested):
    if requested < 1:
        raise ValueError("requested workers must be positive")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(requested, 32, available), available


def raw_objective(Z, y, changepoints):
    boundaries = [0, *np.asarray(changepoints, dtype=int).tolist(), len(y)]
    return float(sum(
        np.square(
            Z[boundaries[column] : boundaries[column + 1], column]
            - y[boundaries[column] : boundaries[column + 1]]
        ).sum()
        for column in range(len(boundaries) - 1)
    ))


def feasible_partitions(n, K, Delta):
    if K == 1:
        yield np.array([], dtype=int)
        return
    for candidate in itertools.combinations(range(1, n), K - 1):
        if all(right - left >= Delta for left, right in zip(candidate, candidate[1:])):
            yield np.asarray(candidate, dtype=int)


def exhaustive_fixed_K(Z, y, K, Delta):
    scored = [(raw_objective(Z, y, cp), cp) for cp in feasible_partitions(len(y), K, Delta)]
    if not scored:
        raise ValueError("infeasible exhaustive case")
    best = min(score for score, _ in scored)
    optima = [cp for score, cp in scored if np.isclose(score, best, atol=1e-12, rtol=0)]
    return best, optima


def exhaustive_selection(Z, y, L, Delta, penalty, upper_bound):
    K_values = range(1, L + 1) if upper_bound else (L,)
    candidates = []
    for K in K_values:
        try:
            raw, optima = exhaustive_fixed_K(Z, y, K, Delta)
        except ValueError:
            continue
        candidates.append((raw + penalty(K), K, raw, optima))
    if not candidates:
        raise ValueError("infeasible exhaustive selection")
    best_total = min(candidate[0] for candidate in candidates)
    total_optima = [
        candidate for candidate in candidates
        if np.isclose(candidate[0], best_total, atol=1e-12, rtol=0)
    ]
    selected = min(total_optima, key=lambda candidate: candidate[1])
    unique = len(total_optima) == 1 and len(selected[3]) == 1
    return selected[1], selected[2], selected[0], selected[3][0], unique


def greedy_fixed_K(Z, y, K, Delta):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        changepoints = chgpt_binary_segmentation(
            squared_loss, Z[:, :K], y, len(y), K, Delta, γ=0.0,
        )
    changepoints = np.asarray(changepoints, dtype=int)
    if len(changepoints) != K - 1:
        return None
    if np.any(np.diff(changepoints) < Delta):
        return None
    return changepoints


def greedy_selection(Z, y, L, Delta, penalty, upper_bound):
    K_values = range(1, L + 1) if upper_bound else (L,)
    candidates = []
    for K in K_values:
        changepoints = greedy_fixed_K(Z, y, K, Delta)
        if changepoints is None:
            continue
        raw = raw_objective(Z, y, changepoints)
        candidates.append((raw + penalty(K), K, raw, changepoints))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))


def deterministic_case(boundaries, n):
    K = len(boundaries) + 1
    Z = np.full((n, K), 3.0)
    y = np.zeros(n)
    endpoints = [0, *boundaries, n]
    for column in range(K):
        Z[endpoints[column] : endpoints[column + 1], column] = 0.0
    return Z, y


def penalty_for(penalty_id, n, L, Delta):
    if penalty_id == "zero":
        return zero_k_penalty
    if penalty_id == "manuscript":
        return manuscript_log_binomial_k_penalty(n, 0.2)
    if penalty_id == "legacy":
        return legacy_gamma_k_penalty(n, L, Delta, 0.15)
    if penalty_id == "table":
        return table_k_penalty({K: 0.12 * (K - 1) ** 2 for K in range(1, L + 1)})
    raise ValueError(f"unknown penalty: {penalty_id}")


def validation_cases():
    cases = []
    for case_id, boundaries, n, Delta in (
        ("hand_m1", (4,), 9, 1),
        ("hand_m2", (3, 8), 12, 2),
        ("hand_m3", (2, 6, 10), 12, 3),
    ):
        Z, y = deterministic_case(boundaries, n)
        cases.append({
            "case_id": case_id, "source": "hand", "Z": Z, "y": y,
            "L": len(boundaries) + 1, "Delta": Delta,
            "upper_bound": False, "penalty_id": "zero",
        })

    penalty_ids = ("zero", "manuscript", "legacy", "table")
    for seed in range(20):
        rng = np.random.default_rng(2000 + seed)
        L = 1 + seed % 4
        n = 8 + seed % 4
        Delta = (1, 2, 3)[seed % 3]
        upper_bound = seed >= 12
        if upper_bound:
            L = 4
        cases.append({
            "case_id": f"random_{seed:02d}", "source": "seeded-random",
            "Z": rng.normal(size=(n, L)), "y": rng.normal(size=n),
            "L": L, "Delta": Delta, "upper_bound": upper_bound,
            "penalty_id": penalty_ids[seed % len(penalty_ids)],
        })
    return cases


def format_changepoints(changepoints):
    if changepoints is None:
        return ""
    return ";".join(str(value) for value in np.asarray(changepoints, dtype=int))


def run_validation():
    rows = []
    example_payload = {}
    for case in validation_cases():
        Z, y = case["Z"], case["y"]
        n, L, Delta = len(y), case["L"], case["Delta"]
        penalty = penalty_for(case["penalty_id"], n, L, Delta)
        result = chgpt_gof_dp(
            squared_loss, Z, y, n, L, Delta,
            L_is_upper_bound=case["upper_bound"], penalty=penalty,
        )
        brute_K, _, _, brute_cp, unique = exhaustive_selection(
            Z, y, L, Delta, penalty, case["upper_bound"],
        )

        dp_raw = raw_objective(Z, y, result["changepoints"])
        brute_raw = raw_objective(Z, y, brute_cp)
        dp_total = dp_raw + penalty(result["K"])
        brute_total = brute_raw + penalty(brute_K)
        difference = abs(dp_total - brute_total)

        greedy = greedy_selection(Z, y, L, Delta, penalty, case["upper_bound"])
        if greedy is None:
            greedy_total = greedy_K = greedy_raw = greedy_cp = None
            gap = None
            greedy_note = "greedy infeasible/skipped"
        else:
            greedy_total, greedy_K, greedy_raw, greedy_cp = greedy
            gap = greedy_total - dp_total
            greedy_note = ""

        failures = []
        if difference > TOLERANCE:
            failures.append("objective mismatch")
        if unique and (result["K"] != brute_K or not np.array_equal(result["changepoints"], brute_cp)):
            failures.append("unique optimum mismatch")
        if gap is not None and gap < -TOLERANCE:
            failures.append("DP worse than greedy")
        note = "; ".join(failures or ([greedy_note] if greedy_note else ["unique optimum" if unique else "tied optimum"]))
        rows.append({
            "case_id": case["case_id"], "source": case["source"], "n": n,
            "Lmax": L, "K_mode": "upper-bound" if case["upper_bound"] else "fixed",
            "Delta": Delta, "penalty_id": case["penalty_id"],
            "dp_K": result["K"], "brute_K": brute_K, "greedy_K": greedy_K,
            "dp_changepoints": format_changepoints(result["changepoints"]),
            "brute_changepoints": format_changepoints(brute_cp),
            "greedy_changepoints": format_changepoints(greedy_cp),
            "dp_raw": dp_raw, "brute_raw": brute_raw, "greedy_raw": greedy_raw,
            "dp_total": dp_total, "brute_total": brute_total,
            "greedy_total": greedy_total, "dp_vs_brute_abs_diff": difference,
            "greedy_minus_dp": gap, "status": "FAIL" if failures else "PASS",
            "note": note,
        })
        if case["case_id"].startswith("hand_"):
            example_payload[case["case_id"]] = (case, result["changepoints"], brute_cp, greedy_cp)
    return rows, example_payload


def timed(function, repeats=2):
    function()  # short warm-up
    measured = []
    for repeat in range(repeats):
        start = time.perf_counter()
        function()
        measured.append((repeat + 1, time.perf_counter() - start))
    return measured


def live_brute(Z, y, K, Delta):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return chgpt_gof_bf_mem_eff(
            squared_loss, Z, y, len(y), K, Delta,
            γ=0.0, gof_opt_output=True,
        )


def run_benchmark():
    rows = []

    def record(scenario, n, K, method, function):
        for repeat, seconds in timed(function):
            rows.append({
                "scenario": scenario, "n": n, "K": K, "Lmax": K,
                "Delta": 1, "method": method, "repeat": repeat,
                "seconds": seconds,
            })

    for n in (16, 24, 32, 40):
        K = 3
        rng = np.random.default_rng(3000 + n)
        Z, y = rng.normal(size=(n, K)), rng.normal(size=n)
        record("n-scaling", n, K, "DP", lambda Z=Z, y=y: chgpt_gof_dp(
            squared_loss, Z, y, len(y), K, 1,
        ))
        record("n-scaling", n, K, "brute force", lambda Z=Z, y=y: live_brute(Z, y, K, 1))
        record("n-scaling", n, K, "greedy", lambda Z=Z, y=y: greedy_fixed_K(Z, y, K, 1))

    n = 28
    rng = np.random.default_rng(4040)
    Z_full, y = rng.normal(size=(n, 5)), rng.normal(size=n)
    for K in (2, 3, 4, 5):
        Z = Z_full[:, :K]
        record("K-scaling", n, K, "DP", lambda Z=Z, K=K: chgpt_gof_dp(
            squared_loss, Z, y, n, K, 1,
        ))
        record("K-scaling", n, K, "brute force", lambda Z=Z, K=K: live_brute(Z, y, K, 1))
        record("K-scaling", n, K, "greedy", lambda Z=Z, K=K: greedy_fixed_K(Z, y, K, 1))
    return rows


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, figures_dir, stem):
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(case_rows, timing_rows, examples, figures_dir):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    dp = np.asarray([row["dp_total"] for row in case_rows])
    brute = np.asarray([row["brute_total"] for row in case_rows])
    maximum = np.max(np.abs(dp - brute))
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.scatter(brute, dp, color=PALETTE["dp"], s=28, alpha=0.85)
    limits = [min(dp.min(), brute.min()), max(dp.max(), brute.max())]
    ax.plot(limits, limits, color="0.25", linewidth=1, linestyle="--")
    ax.set(xlabel="Brute-force total objective", ylabel="DP total objective")
    ax.text(0.03, 0.95, f"max |difference| = {maximum:.2e}", transform=ax.transAxes, va="top")
    save_figure(fig, figures_dir, "01_dp_vs_bruteforce")

    comparable = [row for row in case_rows if row["greedy_minus_dp"] is not None]
    gaps = np.asarray([row["greedy_minus_dp"] for row in comparable])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(np.arange(len(gaps)), gaps, color=PALETTE["greedy"], s=25)
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1)
    ax.set(xlabel="Validation case", ylabel=r"Objective$_{greedy}$ - Objective$_{DP}$")
    save_figure(fig, figures_dir, "02_greedy_gap")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, scenario, x_key, x_label in (
        (axes[0], "n-scaling", "n", r"$n$ ($K=3$)"),
        (axes[1], "K-scaling", "K", r"$K$ ($n=28$)"),
    ):
        subset = [row for row in timing_rows if row["scenario"] == scenario]
        for method, color in (("DP", PALETTE["dp"]), ("brute force", PALETTE["brute"]), ("greedy", PALETTE["greedy"])):
            method_rows = [row for row in subset if row["method"] == method]
            x_values = sorted({row[x_key] for row in method_rows})
            medians = [np.median([row["seconds"] for row in method_rows if row[x_key] == value]) for value in x_values]
            ax.plot(x_values, medians, marker="o", label=method, color=color)
        ax.set(xlabel=x_label, ylabel="Measured seconds")
        ax.set_yscale("log")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_runtime")

    rng = np.random.default_rng(5151)
    n, L, Delta = 18, 5, 2
    Z, y = rng.normal(size=(n, L)), rng.normal(size=n)
    penalty = manuscript_log_binomial_k_penalty(n, 0.35)
    result = chgpt_gof_dp(
        squared_loss, Z, y, n, L, Delta,
        L_is_upper_bound=True, penalty=penalty,
    )
    K_values = np.arange(1, L + 1)
    raw = np.asarray([result["raw_by_K"][K] for K in K_values])
    penalties = np.asarray([result["penalty_by_K"][K] for K in K_values])
    total = np.asarray([result["total_by_K"][K] for K in K_values])
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(K_values, raw, marker="o", color=PALETTE["dp"], label="raw best")
    ax.plot(K_values, penalties, marker="s", color=PALETTE["penalty"], label=r"$p(K)$")
    ax.plot(K_values, total, marker="^", color=PALETTE["total"], label="raw + penalty")
    ax.axvline(result["K"], color="0.35", linestyle="--", linewidth=1, label=f"selected K={result['K']}")
    ax.set(xlabel="Number of segments K", ylabel="Objective component", xticks=K_values)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, figures_dir, "04_penalty_selection")

    figure_examples = (
        ("hand_m1", "M1-like", (4,), (0.0, 2.0)),
        ("hand_m2", "M2-like", (3, 8), (0.0, 2.0, -1.2)),
        ("hand_m3", "M3-like", (2, 6, 10), (0.2, 0.85, 0.25, 0.7)),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.8), sharex=False)
    legend_handles = {}
    for ax, (case_id, title, true_cp, levels) in zip(axes, figure_examples):
        case, dp_cp, brute_cp, greedy_cp = examples[case_id]
        n = len(case["y"])
        endpoints = (0, *true_cp, n)
        signal = np.empty(n)
        for index, level in enumerate(levels):
            signal[endpoints[index] : endpoints[index + 1]] = level
        signal += 0.05 * np.sin(np.arange(n))
        line = ax.plot(np.arange(n), signal, color="0.25", marker=".", linewidth=1, label="example signal")[0]
        legend_handles["example signal"] = line
        for position in true_cp:
            handle = ax.axvline(position, color="0.15", linestyle=":", linewidth=1.3, label="true")
            legend_handles["true"] = handle
        for label, changepoints, color, linestyle in (
            ("DP", dp_cp, PALETTE["dp"], "-"),
            ("brute force", brute_cp, PALETTE["brute"], "--"),
            ("greedy", greedy_cp, PALETTE["greedy"], "-."),
        ):
            if changepoints is None:
                continue
            for position in changepoints:
                handle = ax.axvline(position, color=color, linestyle=linestyle, linewidth=1.1, label=label)
                legend_handles[label] = handle
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("signal")
    axes[-1].set_xlabel("Observation index")
    labels = ["example signal", "true", "DP", "brute force", "greedy"]
    fig.legend([legend_handles[label] for label in labels], labels, frameon=False, ncol=5, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, figures_dir, "05_example_partitions")


def pytest_count(test_log):
    if not test_log.exists():
        return "unknown"
    matches = re.findall(r"(\d+) passed", test_log.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else "unknown"


def write_readme(root, rows, timing_rows, test_count):
    passed = sum(row["status"] == "PASS" for row in rows)
    maximum = max(row["dp_vs_brute_abs_diff"] for row in rows)
    worse = sum(
        row["greedy_minus_dp"] is not None and row["greedy_minus_dp"] < -TOLERANCE
        for row in rows
    )
    overall = "PASS" if passed == len(rows) and worse == 0 else "FAIL"
    grouped = {}
    for row in timing_rows:
        key = (row["scenario"], row["n"], row["K"], row["method"])
        grouped.setdefault(key, []).append(row["seconds"])
    speedups = []
    for scenario, n, K, method in grouped:
        if method != "DP":
            continue
        dp_median = np.median(grouped[(scenario, n, K, "DP")])
        brute_median = np.median(grouped[(scenario, n, K, "brute force")])
        speedups.append(brute_median / dp_median)
    faster_count = sum(speedup > 1 for speedup in speedups)
    content = f"""# Phase 1 exact-DP audit

**Overall status: {overall}**

- DP/brute-force cases passed: **{passed} / {len(rows)}**
- Maximum objective difference: **{maximum:.3e}**
- Cases where DP was worse than greedy: **{worse}**
- Focused tests: **{test_count} passed**
- DP faster than brute force: **{faster_count} / {len(speedups)}** matched configurations ({min(speedups):.1f}x--{max(speedups):.1f}x)
- Exact rerun: `bash scripts/rebuttal/check_phase1.sh`

![DP versus brute force](figures/01_dp_vs_bruteforce.png)
![Greedy objective gap](figures/02_greedy_gap.png)
![Measured runtime](figures/03_runtime.png)
![K-only penalty selection](figures/04_penalty_selection.png)
![Example partitions](figures/05_example_partitions.png)

## How to inspect this PR in five minutes

1. Confirm all DP/brute-force rows pass in [cases.csv](results/cases.csv).
2. Confirm there are no negative greedy gaps beyond `1e-9`.
3. Look for points off the parity line and verify the marked K minimizes the penalized curve.
4. Inspect the three partition panels for indexing mistakes.
5. Scan the diff to confirm the production estimator dispatch is unchanged.

Measured repeats are in [timings.csv](results/timings.csv); command output is in
[test_output.txt](results/test_output.txt). The pre-change workflow record is in
[baseline.md](baseline.md).

## Historical reference tree

Artifacts not ported into this cleaner repository are checked read-only under
`/scratch/ga442/Weighted-ERM` (override with `WERM_HISTORICAL_ROOT`). The
Myocardial status is an archived-result `SMOKE`, based on structural validation
of the historical posterior pickle; it is not a fresh reproduction. Historical
FRED CSV/RData inputs were located but not rerun. No audit command writes to or
modifies the historical tree, and no historical result pickle is copied into
this repository. If that source tree is unavailable, Myocardial is `BLOCKED`.

The DP candidate matches the live brute-force convention: `Delta` constrains
adjacent internal changepoints but not endpoints, and K uses the first K columns
of Z. The requested log-binomial penalty uses K-1 changepoints. No manuscript
source is committed in this repository, so the formula was verified against the
issue specification; the legacy helper separately matches the live gamma formula.
The production/default search path is unchanged.
"""
    (root / "README.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("rebuttal/phase1"))
    parser.add_argument("--requested-workers", type=int, default=1)
    args = parser.parse_args()
    workers, available = effective_workers(args.requested_workers)
    print(f"EFFECTIVE WORKERS: {workers} (requested={args.requested_workers}, affinity={available}, cap=32)")
    print("AUDIT MODE: quick, sequential, single-process timing")

    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cases, examples = run_validation()
    timings = run_benchmark()
    write_csv(results_dir / "cases.csv", cases, CASE_COLUMNS)
    timing_columns = ["scenario", "n", "K", "Lmax", "Delta", "method", "repeat", "seconds"]
    write_csv(results_dir / "timings.csv", timings, timing_columns)
    make_plots(cases, timings, examples, figures_dir)
    write_readme(args.output_root, cases, timings, pytest_count(results_dir / "test_output.txt"))

    passed = sum(row["status"] == "PASS" for row in cases)
    maximum = max(row["dp_vs_brute_abs_diff"] for row in cases)
    worse = sum(row["greedy_minus_dp"] is not None and row["greedy_minus_dp"] < -TOLERANCE for row in cases)
    print(f"DP/BRUTE CASES PASSED: {passed}/{len(cases)}")
    print(f"MAXIMUM OBJECTIVE DIFFERENCE: {maximum:.3e}")
    print(f"DP-WORSE-THAN-GREEDY CASES: {worse}")
    return 0 if passed == len(cases) and worse == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
