"""Compact response-letter figures from the completed adjusted-WERM study.

No simulations or fits are run.  The saved results used by
M2_adjusted_comparison.ipynb remain byte-for-byte unchanged.  Run from the
repository root with ``python -m tools.export_m2_adjusted_response``.
"""
from pathlib import Path
import hashlib
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from tools import consistent_plotting as style


OUTPUT = Path("figures/M2_adjusted_response")
SOURCES = {
    "unknown": Path("results/M2_adjusted_comparison/unknown_030_055_27ab19f1cb75/unknown.pkl"),
    "known": Path("results/M2_adjusted_comparison/ecd9d5c2529d/known.pkl"),
}
TRUTH = {"unknown": [.30, .55], "known": [.25, .45, .70]}
DELTAS = [1.5, 3.5, 7.5, 11.5]
METHODS = ["WERM", "WERM adjusted"]
LABELS = [r"\textbf{WERM}", r"\textbf{WERM adjusted}"]
COLORS = ["black", "#4477AA"]
PLOT_STYLE = {
    **style.CANONICAL_STYLE,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(payload, method, metric):
    return [[r["methods"][method][metric] for r in payload["records"]
             if r["setting"] == i and r["methods"][method]["error"] is None]
            for i in range(len(DELTAS))]


def save(fig, name, audit):
    """Use fixed canvases so all three panels align at the same printed scale."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    for ax in fig.axes:
        bounds = ax.get_tightbbox(renderer)
        assert bounds.x0 >= frame.x0 - 1 and bounds.y0 >= frame.y0 - 1, name
        assert bounds.x1 <= frame.x1 + 1 and bounds.y1 <= frame.y1 + 1, name
    for txt in fig.texts:
        bounds = txt.get_window_extent(renderer)
        assert bounds.x0 >= frame.x0 - 1 and bounds.y0 >= frame.y0 - 1, name
        assert bounds.x1 <= frame.x1 + 1 and bounds.y1 <= frame.y1 + 1, name
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{name}.{extension}", bbox_inches=None)
    audit["figures"].append({
        "name": name, "size_inches": fig.get_size_inches().tolist(),
        "legends": sum(ax.get_legend() is not None for ax in fig.axes),
    })
    plt.close(fig)


def line_panels(mode, payload, audit):
    for metric, suffix, ylabel in [
        ("haus", "haus_line", r"Hausdorff dist. / $n$"),
        ("size", "count_line", "No. change points"),
    ]:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.subplots_adjust(left=.19, right=.98, bottom=.18, top=.96)
            for j, method in enumerate(METHODS):
                style.error_series(ax, DELTAS, groups(payload, method, metric),
                                   LABELS[j], COLORS[j], ("D", "o")[j],
                                   ("-", "--")[j], scale=1)
            ax.set(xlabel=r"$\delta=n/p$", ylabel=ylabel,
                   xticks=DELTAS, xticklabels=[str(d) for d in DELTAS])
            ax.set_xlim(.9, 12.1)
            if metric == "size":
                ax.axhline(len(TRUTH[mode]), color="gray", alpha=.8,
                           lw=2.5, zorder=0)
                ax.set(ylim=(-.12, 3.55), yticks=[0, 1, 2, 3])
                ax.legend(loc="upper right" if mode == "unknown" else "lower right",
                          fontsize=16, framealpha=.4, handlelength=1.6,
                          borderpad=.3, labelspacing=.3)
            save(fig, f"{mode}_{suffix}", audit)


def histogram_panel(mode, payload, audit):
    rows = [0, 2]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True)
        fig.subplots_adjust(left=.14, right=.90, bottom=.20, top=.81,
                            wspace=.12, hspace=.14)
        peak = 0.
        counts = []
        for row, setting in enumerate(rows):
            counts.append([])
            for col, method in enumerate(METHODS):
                ax = axes[row, col]
                values = np.concatenate([
                    r["methods"][method]["cp"] / r["n"]
                    for r in payload["records"]
                    if r["setting"] == setting and r["methods"][method]["error"] is None
                ])
                counts[-1].append(len(values))
                assert len(values) > 0 and np.all((values > 0) & (values < 1))
                # Same 12 data-range bins and five-fold bandwidth selection as M2.
                hist, edges, _ = ax.hist(values, bins=12, density=True,
                                        color=COLORS[col], alpha=.8,
                                        edgecolor="white", linewidth=1)
                assert np.isclose(np.sum(hist * np.diff(edges)), 1)
                cv = GridSearchCV(KernelDensity(kernel="gaussian"),
                                  {"bandwidth": np.logspace(-2, 0, 50)},
                                  cv=5, n_jobs=1)
                cv.fit(values[:, None])
                grid = np.linspace(0, 1, 200)
                density = np.exp(cv.best_estimator_.score_samples(grid[:, None]))
                ax.plot(grid, density, color="#555555", lw=1, alpha=.8, zorder=2)
                ax.fill_between(grid, density, color="#999999", alpha=.2, zorder=1)
                peak = max(peak, float(max(hist)), float(max(density)))
                for point in TRUTH[mode]:
                    ax.axvline(point, color="black", ls="--", lw=2, alpha=.9, zorder=10)
                ax.set(xlim=(0, 1), xticks=TRUTH[mode],
                       xticklabels=[f"${format(point, 'g').lstrip('0')}$"
                                    for point in TRUTH[mode]])
                ax.tick_params(width=1, length=4, labelsize=16, pad=3)
                for spine in ("left", "bottom"):
                    ax.spines[spine].set_linewidth(1)
                if col == 1:
                    ax.text(1.06, .5, rf"$\delta={DELTAS[setting]:g}$",
                            transform=ax.transAxes, rotation=90,
                            va="center", ha="left", fontsize=18)
        # Retain the canonical scale unless a peak would otherwise be clipped.
        limit = max(18, 2 * np.ceil(peak / 2))
        for ax in axes.ravel():
            ax.set(ylim=(0, limit), yticks=([0, 8, 16] if limit == 18 else [0, limit/2, limit]))
        axes[0, 0].set_title(r"\textbf{WERM}" + "\n" + r"\phantom{adjusted}",
                             fontsize=18, pad=9, linespacing=.95)
        axes[0, 1].set_title(r"\textbf{WERM}" + "\n" + r"\textbf{adjusted}",
                             fontsize=18, pad=9, linespacing=.95)
        fig.text(.52, .035, "Fractional location", ha="center", va="bottom", fontsize=18)
        fig.text(.025, .50, "Density", va="center", rotation="vertical", fontsize=18)
        audit["histograms"][mode] = {
            "deltas": [DELTAS[i] for i in rows], "pooled_change_points": counts,
            "maximum_density": peak, "density_limit": float(limit),
            "bins": 12, "kde_bandwidth_cv_folds": 5,
        }
        save(fig, f"{mode}_histograms_delta1p5_7p5", audit)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = {str(path): digest(path) for path in SOURCES.values()}
    audit = {"sources_sha256": initial, "figures": [], "histograms": {},
             "error_bars": "Absolute distances of the mean to the first and third quartiles; not confidence intervals.",
             "hausdorff": "Conditional on a nonempty estimated change-point set, as in the source notebook.",
             "known_count": "Exactly three change points in every trial; both curves coincide."}
    for mode, path in SOURCES.items():
        payload = pickle.loads(path.read_bytes())
        assert payload["spec"]["p"] == 100 and payload["spec"]["trials"] == 100
        assert list(payload["spec"]["deltas"]) == DELTAS
        assert list(payload["spec"]["fractions"][mode]) == TRUTH[mode]
        assert len(payload["records"]) == 400
        for i in range(4):
            records = [r for r in payload["records"] if r["setting"] == i]
            assert len(records) == 100
            for r in records:
                assert r["n"] == int(100 * DELTAS[i])
                for method in METHODS:
                    fit = r["methods"][method]
                    assert fit["error"] is None
                    assert fit["size"] == len(fit["cp"])
                    if mode == "known":
                        assert fit["size"] == 3
        line_panels(mode, payload, audit)
        histogram_panel(mode, payload, audit)
    assert initial == {str(path): digest(path) for path in SOURCES.values()}
    audit["export_script_sha256"] = digest(Path(__file__))
    (OUTPUT / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    # A raster preview only; the response uses the six vector PDF panels.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, mode in enumerate(SOURCES):
        for col, suffix in enumerate(("haus_line", "count_line", "histograms_delta1p5_7p5")):
            axes[row, col].imshow(plt.imread(OUTPUT / f"{mode}_{suffix}.png"))
            axes[row, col].axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig(OUTPUT / "preview.png", dpi=150)
    plt.close(fig)
    print(f"Exported six PDF/PNG panels to {OUTPUT}; saved results unchanged.")


if __name__ == "__main__":
    main()
