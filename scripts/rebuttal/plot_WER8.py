#!/usr/bin/env python3
"""Aggregate WER-8 row-level artifacts and make paper-style figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wer8_common import CONFIG, M2_DELTAS, M3_KAPPAS, REPO_ROOT, REPETITIONS


METHOD_STYLE = {
    "McScan": ("McScan", "#EE7733", "o", "-"),
    "charcoal": ("charcoal", "#EE3377", "^", "-."),
    "WERM-unp": ("WERM", "#009988", "<", "--"),
    "GLMBSA": ("GLMBSA", "#EE3377", "^", ":"),
    "WERM-CV": ("WERM-CV", "#CC3311", "v", "-"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.5,
        "savefig.dpi": 300,
    }
)


def load_rows(artifact_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    records: list[dict] = []
    for path in sorted((artifact_root / "raw").glob("*/*/*.json")):
        record = json.loads(path.read_text())
        records.append(record)
        base = {
            "workflow": record["workflow"],
            "parameter_index": record["parameter_index"],
            "parameter_name": record["parameter_name"],
            "parameter_value": record["parameter_value"],
            "repetition": record["repetition"],
            "seed": record["seed"],
            "commit": record["commit"],
            "task_status": record["task_status"],
            "record_path": str(path.resolve()),
            "task_log": record["task_log"],
        }
        for method in record["methods"]:
            rows.append(
                {
                    **base,
                    "method": method["method"],
                    "status": method["status"],
                    "selected_K": method["selected_K"],
                    "selected_changepoint_count": method["selected_changepoint_count"],
                    "empty_estimate": method["empty_estimate"],
                    "hausdorff_over_n": method["hausdorff_over_n"],
                    "runtime_seconds": method["runtime_seconds"],
                    "changepoints": json.dumps(method["changepoints"]),
                    "metric_note": method["metric_note"],
                    "failure": method["failure"],
                }
            )
    return pd.DataFrame(rows), records


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    if rows.empty:
        return pd.DataFrame()
    keys = ["workflow", "parameter_index", "parameter_name", "parameter_value", "method"]
    for values, group in rows.groupby(keys, sort=True, dropna=False):
        successful = group[group.status == "SUCCESS"]
        defined = successful[successful.hausdorff_over_n.notna()]
        output.append(
            dict(
                zip(keys, values),
                accounted_repetitions=int(group.repetition.nunique()),
                success_count=int((group.status == "SUCCESS").sum()),
                blocked_count=int(group.status.str.startswith("BLOCKED").sum()),
                failed_count=int(group.status.isin(["FAILED", "NOT_RUN_DATA_FAILURE"]).sum()),
                historical_skip_count=int((group.status == "NOT_RUN_HISTORICAL_RULE").sum()),
                empty_estimate_count=int((successful.empty_estimate == True).sum()),  # noqa: E712
                defined_hausdorff_count=len(defined),
                hausdorff_mean=defined.hausdorff_over_n.mean(),
                hausdorff_q25=defined.hausdorff_over_n.quantile(0.25),
                hausdorff_q75=defined.hausdorff_over_n.quantile(0.75),
                changepoint_count_mean=successful.selected_changepoint_count.mean(),
                changepoint_count_q25=successful.selected_changepoint_count.quantile(0.25),
                changepoint_count_q75=successful.selected_changepoint_count.quantile(0.75),
                runtime_mean_seconds=successful.runtime_seconds.mean(),
                runtime_q25_seconds=successful.runtime_seconds.quantile(0.25),
                runtime_q75_seconds=successful.runtime_seconds.quantile(0.75),
            )
        )
    return pd.DataFrame(output)


def _errorbar(ax, summary: pd.DataFrame, method: str, x_column: str, metric: str):
    if metric == "runtime":
        mean_column, q25_column, q75_column = (
            "runtime_mean_seconds",
            "runtime_q25_seconds",
            "runtime_q75_seconds",
        )
    else:
        mean_column, q25_column, q75_column = (
            f"{metric}_mean",
            f"{metric}_q25",
            f"{metric}_q75",
        )
    data = summary[(summary.method == method) & summary[mean_column].notna()].sort_values("parameter_index")
    if data.empty:
        return False
    label, color, marker, linestyle = METHOD_STYLE[method]
    mean = data[mean_column].to_numpy(float)
    q25 = data[q25_column].to_numpy(float)
    q75 = data[q75_column].to_numpy(float)
    ax.errorbar(
        # Match the notebooks: asymmetric bar lengths are absolute distances
        # from the mean to the 25th and 75th percentiles.
        data[x_column], mean, yerr=np.vstack([np.abs(mean - q25), np.abs(q75 - mean)]),
        label=label, color=color, marker=marker, ls=linestyle,
        capsize=4, capthick=2, markersize=7, linewidth=3,
        markeredgewidth=1.5, markeredgecolor="white",
    )
    return True


def make_figure(workflow: str, summary: pd.DataFrame, figures_dir: Path) -> None:
    workflow_summary = summary[summary.workflow == workflow]
    methods = CONFIG[workflow]["methods"]
    parameter = CONFIG[workflow]["parameter_name"]
    x_label = r"$\delta$" if workflow == "m2" else r"$\kappa$"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    panels = (
        ("hausdorff", "Hausdorff Distance / $n$", "(a) Changepoint Location Error"),
        ("changepoint_count", "Number of Changepoints", "(b) Changepoint Set Size"),
        ("runtime", "Runtime (seconds)", "(c) Method Runtime"),
    )
    plotted = []
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for method in methods:
            plotted_now = _errorbar(ax, workflow_summary, method, "parameter_value", metric)
            if plotted_now and method not in plotted:
                plotted.append(method)
        if metric == "changepoint_count":
            ax.axhline(2, color="black", linewidth=2.2, alpha=0.8, label="True Size")
            ax.set_ylim(0, 6)
        elif metric == "hausdorff":
            ax.set_ylim(0, 0.5)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel(x_label, fontsize=18)
        if workflow == "m2":
            ax.set_xticks(M2_DELTAS)
        else:
            ax.set_xticks(M3_KAPPAS)
            ax.set_xticklabels([f"{value:.2f}" for value in M3_KAPPAS], rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=17)
        ax.set_title(title, fontsize=18, pad=15, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=12, width=1.3, length=5)
        if workflow == "m3":
            ax.tick_params(axis="x", labelsize=9)

    unavailable = [method for method in methods if method not in plotted]
    if unavailable:
        fig.text(
            0.99, 0.01,
            "Not plotted (no successful rows): " + ", ".join(unavailable),
            ha="right", va="bottom", fontsize=10,
        )
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=14, frameon=True)
    fig.suptitle(f"{workflow.upper()}: 100-repetition exact-DP rerun", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.94], w_pad=4.0)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / f"{workflow.upper()}_100_repetitions"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_run_log(artifact_root: Path, results_dir: Path, rows: pd.DataFrame, records: list[dict]) -> None:
    run_files = sorted((artifact_root / "runs").glob("*.json"))
    lines = [
        "WER-8 compact run log",
        f"artifact_root: {artifact_root.resolve()}",
        f"raw_records: {len(records)}",
        f"method_rows: {len(rows)}",
        "",
    ]
    for path in run_files:
        run = json.loads(path.read_text())
        lines.extend(
            [
                f"run: {path.name}",
                f"  workflow: {run['workflow']}",
                f"  commit: {run['commit']}",
                f"  command: {run['command']}",
                f"  started_utc: {run['started_utc']}",
                f"  ended_utc: {run['ended_utc']}",
                f"  effective_workers: {run['settings']['effective_workers']}",
                f"  inner_threads: {run['settings']['inner_threads']}",
                f"  seed_policy: {run['settings']['seed_policy']}",
                f"  task_count_started: {run['task_count_started']}",
                f"  exit_code: {run['exit_code']}",
                f"  competitor_dependency: {run['competitor_dependency']}",
                "",
            ]
        )
    if not rows.empty:
        lines.append("method status totals:")
        for (workflow, method, status), count in rows.groupby(["workflow", "method", "status"]).size().items():
            lines.append(f"  {workflow}/{method}/{status}: {count}")
    (results_dir / "run_log.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "rebuttal" / "WER-8-m2-m3-100" / "artifacts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "rebuttal" / "WER-8-m2-m3-100",
    )
    args = parser.parse_args()
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows, records = load_rows(args.artifact_root)
    if rows.empty:
        raise SystemExit(f"no raw records found below {args.artifact_root / 'raw'}")
    summary = summarize(rows)
    rows.to_csv(results_dir / "accounting.csv", index=False)
    for workflow in ("m2", "m3"):
        summary[summary.workflow == workflow].to_csv(results_dir / f"{workflow}_summary.csv", index=False)
        make_figure(workflow, summary, figures_dir)
    write_run_log(args.artifact_root, results_dir, rows, records)

    expected = {"m2": len(M2_DELTAS) * REPETITIONS, "m3": len(M3_KAPPAS) * REPETITIONS}
    for workflow, count in expected.items():
        actual = len({(r["parameter_index"], r["repetition"]) for r in records if r["workflow"] == workflow})
        print(f"{workflow.upper()} ACCOUNTED: {actual}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
