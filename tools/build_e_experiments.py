"""Generate the E-PRIOR notebook; preserve the original E-L notebook cells."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

COMMON = r'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
from time import perf_counter
import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from scipy.special import comb
from WERM.estimators import weighted_erm
from WERM.gof import chgpt_gof_dp

plt.switch_backend("module://matplotlib_inline.backend_inline")
plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.labelsize": 16,
    "axes.titlesize": 17, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SMOKE_MODE = os.environ.get("WERM_SMOKE") == "1"
N_TRIALS = 2 if SMOKE_MODE else 100
N_JOBS = 2 if SMOKE_MODE else 36
SEED = 20260916
P = 100
N = 500
NOISE_SD = 0.5
M = lambda prediction, truth: 0.5 * (np.asarray(prediction) - np.asarray(truth)) ** 2

def make_paired_data(trial, true_cp):
    """Identical X, y and signals across methods within trial."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, trial]))
    delta = N / P
    beta = np.empty((P, len(true_cp) + 1))
    beta[:, 0] = rng.normal(size=P) * np.sqrt(delta)
    for segment in range(1, beta.shape[1]):
        beta[:, segment] = beta[:, segment - 1] + 1.5 * np.sqrt(delta) * rng.normal(size=P)
    X = rng.normal(size=(N, P)) / np.sqrt(N)
    signal = np.searchsorted(true_cp, np.arange(N), side="right")
    y = np.einsum("ij,ij->i", X, beta[:, signal].T) + rng.normal(0, NOISE_SD, N)
    return X, y, beta, signal

def error_metrics(cp, truth):
    """Normalized symmetric nearest-neighbor loss, defined even at the empty set.

    U(A,B) = 0.5*[mean_a min_b |a-b|/n + mean_b min_a |a-b|/n]
             + 0.25*||A|-|B||/max(1,|A|,|B|).
    Missing-direction distance is 1. Hausdorff is reported only for nonempty B.
    """
    a, b = np.asarray(truth, int), np.asarray(cp, int)
    if b.size:
        pair = np.abs(a[:, None] - b[None, :]) / N
        directed_a = float(pair.min(axis=1).mean())
        directed_b = float(pair.min(axis=0).mean())
        haus = float(max(pair.min(axis=1).max(), pair.min(axis=0).max()))
    else:
        directed_a = directed_b = 1.0
        haus = np.nan
    count_penalty = 0.25 * abs(len(a) - len(b)) / max(1, len(a), len(b))
    return dict(count=len(b), empty=(len(b) == 0),
                conditional_haus=haus,
                exact_count_haus=haus if len(a) == len(b) else np.nan,
                unconditional_loss=0.5 * (directed_a + directed_b) + count_penalty)

def conditional_coefficient_mse(X, beta, true_signal, phi):
    """Exact E_noise[mean((B_hat-B_true)^2) | X,B_true] for this weighted LS fit.

    B_hat_l = A_l y, A_l=(X'W_l X)^(-1) X'W_l. Thus the conditional
    expectation is squared bias plus noise_sd^2 * ||A_l||_F^2. This is an
    exact finite-sample coefficient prediction, not a CP-risk theorem.
    """
    from scipy.linalg import cho_factor, cho_solve
    mu = np.einsum("ij,ij->i", X, beta[:, true_signal].T)
    total = 0.0
    for l, weight in enumerate(phi):
        w = weight / weight.sum()
        gram = X.T @ (w[:, None] * X)
        right = X.T * w
        try:
            A = cho_solve(cho_factor(gram, lower=True, check_finite=False),
                          right, check_finite=False)
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(gram) @ right
        bias = A @ mu - beta[:, l]
        total += float(bias @ bias + NOISE_SD**2 * np.sum(A*A))
    return total / beta.shape[0] / beta.shape[1]

def population_prediction(beta, true_signal, phi, delta_n):
    """Conditional population-risk comparator for the *same* weighted fixed-L ERM.

    With X_i~N(0,I/n), E[w_i (y_i-X_i b)^2] is minimized at
    b_l = sum_i phi[l,i] beta_{s(i)} / sum_i phi[l,i].
    The expected per-time GOF is 0.5*(noise_sd^2+||beta_s-b_l||^2/n).
    This is a population prediction, NOT a finite-sample theorem or fixed-point
    prediction. It uses the identical phi, L, loss, and partition search.
    """
    L = phi.shape[0]
    row_mass = phi.sum(axis=1)
    mixture = np.column_stack([
        np.bincount(true_signal, weights=phi[l], minlength=beta.shape[1]) / row_mass[l]
        for l in range(L)
    ])
    beta_pop = beta @ mixture
    risk = np.empty((N, L))
    for l in range(L):
        squared = np.sum((beta[:, true_signal].T - beta_pop[:, l])**2, axis=1) / N
        risk[:, l] = 0.5 * (NOISE_SD**2 + squared)
    result = chgpt_gof_dp(lambda column, unused: column, risk, np.zeros(N),
                          N, L, delta_n, L_is_upper_bound=False)
    return result["changepoints"], beta_pop
'''

EL = r'''
# E-L: fixed-L least-squares WERM, including severe under/over-specification.
TRUE_CP = np.rint(N * np.array([0.20, 0.50, 0.80])).astype(int)
TRUE_L = len(TRUE_CP) + 1
WORKING_L = np.arange(1, 7)
DELTA_N = 1  # The package's exact-L default analytic marginal branch.
ARTIFACT = RESULTS_DIR / ("E_L_fixedL_L1to6_100paired_seed20260916_smoke.pkl" if SMOKE_MODE
                          else "E_L_fixedL_L1to6_100paired_seed20260916.pkl")

def analytic_fixed_l_marginals(L):
    # Exactly the Delta_n=1 formula used inside weighted_erm (i=1,...,n).
    x = np.arange(1, N + 1) / N
    phi = np.array([comb(L-1, l) * x**l * (1-x)**(L-1-l)
                    for l in range(L)], dtype=float)
    assert np.allclose(phi.sum(axis=0), 1)
    return phi

def run_trial(trial):
    X, y, beta, signal = make_paired_data(trial, TRUE_CP)
    output = {}
    for L in WORKING_L:
        tic = perf_counter()
        cp, beta_hat = weighted_erm(
            X, y, M, L=int(L), Delta_n=DELTA_N, gamma=0.0,
            L_is_upper_bound=False, return_B_hat=True,
            coefficient_loss="least_squares", coefficient_penalty="none")
        elapsed = perf_counter() - tic
        assert len(cp) == L - 1  # exact fixed-L, not model-selection CV
        output[int(L)] = dict(cp=np.asarray(cp), runtime=elapsed,
                              **error_metrics(cp, TRUE_CP))
        if L == TRUE_L:
            phi = analytic_fixed_l_marginals(int(L))
            theory_cp, beta_pop = population_prediction(beta, signal, phi, DELTA_N)
            output[int(L)]["population_cp"] = theory_cp
            output[int(L)]["population_loss"] = error_metrics(theory_cp, TRUE_CP)["unconditional_loss"]
            output[int(L)]["coefficient_mse"] = float(np.mean((beta_hat-beta)**2))
            output[int(L)]["conditional_expected_coefficient_mse"] = (
                conditional_coefficient_mse(X, beta, signal, phi))
            output[int(L)]["population_coefficient_bias"] = float(np.mean((beta_pop-beta)**2))
    return output

with parallel_config(backend="loky", inner_max_num_threads=1):
    trials = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run_trial)(t) for t in range(N_TRIALS))
assert len(trials) == N_TRIALS
assert all(set(row) == set(WORKING_L) for row in trials)
with ARTIFACT.open("wb") as handle:
    pickle.dump(dict(trials=trials, params=dict(N=N, P=P, true_cp=TRUE_CP,
        working_L=WORKING_L, delta_n=DELTA_N, n_trials=N_TRIALS,
        n_jobs=N_JOBS, seed=SEED, theory_kind="conditional population risk; matched L=4 only")), handle)
print(f"Saved {N_TRIALS} paired trials to {ARTIFACT}")
'''

PRIOR = r'''
# E-PRIOR: uniform joint prior over admissible ordered pairs, then restricted.
TRUE_CP = np.rint(N * np.array([0.30, 0.55])).astype(int)
L = 3
DELTA_N = 25
GOOD_SUPPORT_GAP = .04  # Actual 0.04 fractional-location gap, centered at 0.425.
PRIORS = ("Neutral", "Good", "Bad")
ARTIFACT = RESULTS_DIR / ("E_PRIOR_jointconfigs_actualgap_100paired_seed20260916_smoke.pkl" if SMOKE_MODE
                          else "E_PRIOR_jointconfigs_actualgap_100paired_seed20260916.pkl")

def admissible_pairs(n, delta_n):
    # Exactly the DP feasible set: no endpoint spacing, second-first >= delta_n.
    first, second = np.triu_indices(n-1, k=delta_n)
    return np.column_stack((first + 1, second + 1))

def in_window(index, center, half_width=1/8, trim_left=0, trim_right=0):
    # Half-open intervals; trims change the support itself, not its display.
    return ((index / N >= center - half_width + trim_left) &
            (index / N < center + half_width - trim_right))

def marginalize_joint_prior(configs, n):
    """Integrate a uniform joint prior on eta=(cp1,cp2) to P(signal=l at i)."""
    if len(configs) == 0:
        raise ValueError("prior has no admissible configurations")
    phi = np.zeros((3, n + 1), dtype=float)
    mass = 1.0 / len(configs)
    # Difference arrays: add mass to [0, cp1), [cp1, cp2), [cp2, n).
    np.add.at(phi[0], 0, mass * len(configs))
    np.add.at(phi[0], configs[:, 0], -mass)
    np.add.at(phi[1], configs[:, 0], mass)
    np.add.at(phi[1], configs[:, 1], -mass)
    np.add.at(phi[2], configs[:, 1], mass)
    np.add.at(phi[2], n, -mass * len(configs))
    phi = np.cumsum(phi[:, :n], axis=1)
    assert np.all(phi >= -1e-12) and np.allclose(phi.sum(axis=0), 1, atol=1e-10)
    assert np.all(phi.sum(axis=1) > 0)
    return np.clip(phi, 0, 1)

ALL_CONFIGS = admissible_pairs(N, DELTA_N)
first_ok = in_window(ALL_CONFIGS[:, 0], 0.30)
first_good = in_window(ALL_CONFIGS[:, 0], 0.30, trim_right=GOOD_SUPPORT_GAP/2)
second_good = in_window(ALL_CONFIGS[:, 1], 0.55, trim_left=GOOD_SUPPORT_GAP/2)
second_bad = in_window(ALL_CONFIGS[:, 1], 0.80)
assert not np.any(second_good & second_bad)
CONFIGS = {
    "Neutral": ALL_CONFIGS,
    "Good": ALL_CONFIGS[first_good & second_good],
    "Bad": ALL_CONFIGS[first_ok & second_bad],
}
assert (CONFIGS["Good"][:, 1].min() - CONFIGS["Good"][:, 0].max()) / N >= GOOD_SUPPORT_GAP
PHI = {name: marginalize_joint_prior(configs, N) for name, configs in CONFIGS.items()}
print("Joint prior support sizes:", {name: len(configs) for name, configs in CONFIGS.items()})
assert all(np.allclose(phi.sum(axis=0), 1) for phi in PHI.values())
EFFECTIVE_N = {name: (phi.sum(axis=1)**2 / (phi**2).sum(axis=1))
               for name, phi in PHI.items()}
print("Effective sample sizes by signal:", EFFECTIVE_N)
assert all(np.all(ess > P) for ess in EFFECTIVE_N.values()), \
    "A prior leaves a weighted fit effectively underdetermined; increase n or reduce p"

def run_trial(trial):
    X, y, beta, signal = make_paired_data(trial, TRUE_CP)
    output = {}
    for name in PRIORS:
        phi = PHI[name]
        tic = perf_counter()
        cp, beta_hat = weighted_erm(
            X, y, M, L=L, Delta_n=DELTA_N, gamma=0.0,
            L_is_upper_bound=False, return_B_hat=True,
            marginal_weights=phi, coefficient_loss="least_squares",
            coefficient_penalty="none")
        elapsed = perf_counter() - tic
        assert len(cp) == L - 1
        theory_cp, beta_pop = population_prediction(beta, signal, phi, DELTA_N)
        output[name] = dict(cp=np.asarray(cp), runtime=elapsed,
                            coefficient_mse=float(np.mean((beta_hat-beta)**2)),
                            conditional_expected_coefficient_mse=(
                                conditional_coefficient_mse(X, beta, signal, phi)),
                            population_cp=theory_cp,
                            population_loss=error_metrics(theory_cp, TRUE_CP)["unconditional_loss"],
                            population_coefficient_bias=float(np.mean((beta_pop-beta)**2)),
                            **error_metrics(cp, TRUE_CP))
    return output

with parallel_config(backend="loky", inner_max_num_threads=1):
    trials = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run_trial)(t) for t in range(N_TRIALS))
assert len(trials) == N_TRIALS
with ARTIFACT.open("wb") as handle:
    pickle.dump(dict(trials=trials, configs=CONFIGS, marginals=PHI,
        params=dict(N=N, P=P, true_cp=TRUE_CP, L=L, delta_n=DELTA_N,
            n_trials=N_TRIALS, n_jobs=N_JOBS, seed=SEED,
            windows="half-open; first center=.30; good second=.55; bad second=.80; width=1/4; good gap=.04",
            theory_kind="conditional population risk under identical fixed-L estimator")), handle)
print(f"Saved {N_TRIALS} paired trials to {ARTIFACT}")
'''

PLOT_COMMON = r'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.switch_backend("module://matplotlib_inline.backend_inline")
plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 18,
    "axes.linewidth": 1.5, "axes.titlesize": 18, "legend.fontsize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.usetex": True,
    "lines.linewidth": 3.0, "lines.markersize": 8,
    "figure.dpi": 72, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
SMOKE_MODE = __import__("os").environ.get("WERM_SMOKE") == "1"

def series(trials, key, metric):
    return np.array([row[key][metric] for row in trials], dtype=float)

def mean_se(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1], np.nan), where=count > 0)
    centered = np.where(finite, values - mean, 0)
    sd = np.sqrt(np.divide((centered**2).sum(axis=0), np.maximum(count-1, 1)))
    se = np.divide(sd, np.sqrt(count), out=np.zeros_like(sd), where=count > 0)
    return mean, se

def line_panel(ax, x, values, ylabel, color="black", theory=None):
    mean, se = mean_se(values)
    ax.errorbar(x, mean, yerr=se, color=color, marker="D", linestyle="-",
                linewidth=2.3, markersize=5, capsize=3, label="WERM")
    if theory is not None:
        ax.plot(x, theory, marker="o", linestyle="--", color="#AA3377",
                linewidth=1.7, markersize=4, label="Population")
        ax.legend(frameon=False)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)

def jitter_boxes(ax, grouped, labels, ylabel, colors):
    rng = np.random.default_rng(2901)
    positions = np.arange(1, len(grouped)+1)
    clean = [np.asarray(g)[np.isfinite(g)] for g in grouped]
    for k, values in enumerate(clean):
        if len(values):
            ax.scatter(rng.normal(positions[k], .055, len(values)), values,
                       s=13, alpha=.23, color=colors[k], zorder=2)
    bp = ax.boxplot([g if len(g) else [np.nan] for g in clean],
                    positions=positions, widths=.48, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=.78)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.18)
'''

EL_PLOT = r'''
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
ARTIFACT = Path("results") / ("E_L_fixedL_L1to6_100paired_seed20260916_smoke.pkl" if SMOKE_MODE
                              else "E_L_fixedL_L1to6_100paired_seed20260916.pkl")
with ARTIFACT.open("rb") as handle:
    payload = pickle.load(handle)
trials, params = payload["trials"], payload["params"]
L_values = np.asarray(params["working_L"])
true_cp, n = np.asarray(params["true_cp"]), params["N"]
metrics = ["conditional_haus", "empty", "count", "runtime", "unconditional_loss"]
arrays = {m: np.column_stack([series(trials, int(L), m) for L in L_values]) for m in metrics}
print(f"E-L: {len(trials)} paired trials, n={n}, p={params['P']}; artifact: {ARTIFACT}")
print("L | count | empty rate | mean / median conditional Hausdorff | exact-count Hausdorff | unconditional loss | runtime s")
for L in L_values:
    rows = [row[int(L)] for row in trials]
    avg = lambda key: np.nanmean([r[key] for r in rows])
    conditional = np.asarray([r["conditional_haus"] for r in rows], dtype=float)
    exact = np.array([r["exact_count_haus"] for r in rows])
    print(f"{L} | {avg('count'):.2f} | {avg('empty'):.2%} | "
          f"{np.nanmean(conditional):.3f} / {np.nanmedian(conditional):.3f} | "
          f"{np.nanmean(exact) if np.isfinite(exact).any() else np.nan:.3f} | "
          f"{avg('unconditional_loss'):.3f} | {avg('runtime'):.3f}")

plt.rcParams.update({"font.size": 14, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14})
print("Conditional localization by working L-hat (mean ± SE, then trial distributions).")
fig, ax = plt.subplots(figsize=(8, 4.7))
line_panel(ax, L_values, arrays["conditional_haus"], r"Hausdorff dist./$n$")
ax.set_xlabel(r"Working $\hat{L}$"); ax.set_xticks(L_values)
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
fig.tight_layout(pad=.6)
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.7))
labels = [str(v) for v in L_values]
colors = ["#4477AA" if v < 4 else "black" if v == 4 else "#EE6677" for v in L_values]
jitter_boxes(ax, [arrays["conditional_haus"][:, j] for j in range(len(L_values))],
             labels, r"Hausdorff dist./$n$", colors)
ax.set_xlabel(r"Working $\hat{L}$")
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
fig.tight_layout(pad=.6)
plt.show()

print("Estimated change-point locations; panels left-to-right: working L-hat = 1, 2, 3, 4, 5, 6. Dashed lines mark truth.")
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
for ax, L in zip(axes.flat, L_values):
    cps = np.concatenate([row[int(L)]["cp"] / n for row in trials])
    if len(cps):
        ax.hist(cps, bins=12, range=(0, 1), density=True,
                color="black", edgecolor="white", alpha=.8)
    for cp in true_cp / n: ax.axvline(cp, color="#EE6677", linestyle="--", linewidth=1.5)
    ax.set_title(rf"Working $\hat{{L}}={L}$")
    ax.set_xlabel("Fractional location")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axes[0, 0].set_ylabel("Density")
axes[1, 0].set_ylabel("Density")
fig.tight_layout(pad=.6, w_pad=1.0, h_pad=1.0)
plt.show()
'''

PRIOR_PLOT = r'''
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
ARTIFACT = Path("results") / ("E_PRIOR_jointconfigs_actualgap_100paired_seed20260916_smoke.pkl" if SMOKE_MODE
                              else "E_PRIOR_jointconfigs_actualgap_100paired_seed20260916.pkl")
with ARTIFACT.open("rb") as handle:
    payload = pickle.load(handle)
trials, params = payload["trials"], payload["params"]
phi, configs = payload["marginals"], payload["configs"]
names = ["Bad", "Neutral", "Good"]
colors = {"Neutral":"#4477AA", "Good":"#228833", "Bad":"#EE6677"}
true_cp, n = np.asarray(params["true_cp"]), params["N"]

def pooled_change_point_prior_mass(configurations, n):
    """Exact PMF of a uniformly selected change point from each configuration.

    Each configuration contributes half its mass at each of its two changes.
    The x coordinates are i/n, but the y values are probabilities summing to 1.
    """
    configurations = np.asarray(configurations, dtype=int)
    counts = np.bincount(configurations.reshape(-1), minlength=n)
    mass = counts.astype(float) / (2 * len(configurations))
    return mass, np.arange(n) / n

def exact_joint_prior_mass(configurations, n):
    """Exact joint configuration mass on the observed n-by-n index grid."""
    configurations = np.asarray(configurations, dtype=int)
    mass = np.zeros((n, n), dtype=float)
    np.add.at(mass, (configurations[:, 0], configurations[:, 1]),
              1.0 / len(configurations))
    return mass

print(f"E-PRIOR: {len(trials)} paired trials, n={n}, p={params['P']}; artifact: {ARTIFACT}")
print("Prior | configurations | mean selected changes | mean / median conditional Hausdorff")
for name in names:
    rows = [row[name] for row in trials]
    avg = lambda key: np.nanmean([r[key] for r in rows])
    conditional = np.asarray([r["conditional_haus"] for r in rows], dtype=float)
    print(f"{name} | {len(configs[name]):,} | {avg('count'):.2f} | "
          f"{np.nanmean(conditional):.3f} / {np.nanmedian(conditional):.3f}")

plt.rcParams.update({"font.size": 14, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14})
print("Conditional localization by prior (Bad, Neutral, Good); mean ± SE.")
fig, ax = plt.subplots(figsize=(8, 4.7))
values = np.column_stack([series(trials, name, "conditional_haus") for name in names])
mean, se = mean_se(values)
ax.errorbar(range(3), mean, yerr=se, color="black", marker="D",
            linestyle="-", linewidth=2.3, markersize=5, capsize=3)
ax.set_xticks(range(3), names)
ax.set_ylabel(r"Hausdorff dist./$n$")
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.grid(alpha=.18)
fig.tight_layout(pad=.6)
plt.show()

# Distribution of A change point: choose one of the two uniformly within each
# sampled configuration. This pools ordered positions without treating either
# one as special. Use exact probability mass per index, not a scaled density.
print("Exact prior for a randomly selected change point; panels: Bad, Neutral, Good. Dashed lines mark truth.")
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
for ax, name in zip(axes, names):
    mass, x_grid = pooled_change_point_prior_mass(configs[name], n)
    ax.plot(x_grid[1:], mass[1:], color=colors[name], linewidth=2.3)
    for cp in true_cp:
        ax.axvline(cp / n, color="#555555", linestyle="--",
                   linewidth=1.3, alpha=.75)
    ax.set_title(name)
    ax.set_xlabel("Fractional location")
    ax.set_xlim(-.02, 1.02)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(alpha=.16)
axes[0].set_ylabel("Prior prob.")
fig.tight_layout(pad=.6, w_pad=1.1)
plt.show()

# A second view keeps the joint dependence that a pair of 1-D curves hides.
# Each pixel is one exact configuration (cp1, cp2), not a coarse histogram bin.
print("Exact joint prior over configurations (x: first change, y: second); panels: Bad, Neutral, Good.")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
for ax, name in zip(axes, names):
    mass = exact_joint_prior_mass(configs[name], n)
    mesh = ax.imshow(mass.T, origin="lower", interpolation="none",
                     extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
    ax.scatter([true_cp[0] / n], [true_cp[1] / n], marker="x",
               color="#22D3EE", s=85, linewidth=2.2, label="Truth")
    ax.set_title(name)
    ax.set_xlabel("Fractional location")
    if ax is axes[0]: ax.set_ylabel("Fractional location")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.colorbar(mesh, ax=ax, shrink=.75, format="%.0e",
                 ticks=np.linspace(0, mass.max(), 4))
axes[-1].legend(frameon=False)
fig.tight_layout(pad=.5, w_pad=.7)
plt.show()

# Marginals are derived from the joint prior, not guessed as independent windows.
print("Signal-occupancy marginals derived from the joint priors; panels: Bad, Neutral, Good.")
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
line_styles = ("-", "--", ":")
for ax, name in zip(axes, names):
    for l in range(3):
        ax.plot(np.arange(n)/n, phi[name][l], linewidth=2.3,
                color=colors[name], linestyle=line_styles[l],
                label=rf"$\pi_{{i}}^{{({l+1})}}$")
    for cp in true_cp/n: ax.axvline(cp, color="black", linestyle="--", alpha=.65)
    ax.set_title(name)
    ax.set_xlabel("Fractional location")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(alpha=.16)
axes[0].set_ylabel("Sample weights")
axes[-1].legend(frameon=False, loc="center right")
fig.tight_layout(pad=.6, w_pad=1.1)
plt.show()

print("Conditional localization across paired trials (boxplots and faint jittered trial points).")
fig, ax = plt.subplots(figsize=(8, 4.7))
jitter_boxes(ax, [series(trials, name, "conditional_haus") for name in names],
             names, r"Hausdorff dist./$n$", [colors[name] for name in names])
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
fig.tight_layout(pad=.6)
plt.show()

print("Estimated change-point locations; panels: Bad, Neutral, Good. Dashed lines mark truth.")
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
for ax, name in zip(axes, names):
    cps = np.concatenate([row[name]["cp"] / n for row in trials])
    ax.hist(cps, bins=12, range=(0, 1), density=True, color=colors[name],
            alpha=.78, edgecolor="white", label="Empirical")
    for cp in true_cp/n: ax.axvline(cp, color="black", linestyle="--", linewidth=1.7)
    ax.set_title(name)
    ax.set_xlabel("Fractional location")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axes[0].set_ylabel("Density")
fig.tight_layout(pad=.6, w_pad=1.1)
plt.show()

print("Ordered estimated change-point locations; left: first, right: second.")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, coordinate in zip(axes, (0, 1)):
    for name in names:
        values = np.array([row[name]["cp"][coordinate] / n for row in trials])
        ax.hist(values, bins=12, range=(0, 1), density=True, histtype="step",
                linewidth=2.1, color=colors[name], label=name)
    ax.axvline(true_cp[coordinate]/n, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Fractional location")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axes[0].set_ylabel("Density")
axes[-1].legend(frameon=False)
fig.tight_layout(pad=.6, w_pad=1.1)
plt.show()
'''


EL_PAPER_DESCRIPTION = r"""Across 100 paired Monte Carlo trials (base seed 20260916), we generated $n=500$ observations with $p=100$ covariates, $x_i\sim N(0,I_p/n)$ and $y_i=x_i^\top\beta_{s(i)}+\varepsilon_i$, where $\varepsilon_i\sim N(0,0.5^2)$. The true $L^*=4$ regimes change at fractional locations $(0.20,0.50,0.80)$; the initial coefficient vector is $N(0,(n/p)I_p)$ and independent successive jumps are $N(0,1.5^2(n/p)I_p)$. On the same data in each trial, we fit fixed-$L$ unpenalized least-squares WERM for $L\in\{1,2,3,4,5,6\}$ with minimum change-point gap $\Delta_n=1$, half-squared-error loss, and the package's analytic signal weights $\pi_i^{(\ell)}=\binom{L-1}{\ell}(i/n)^\ell(1-i/n)^{L-1-\ell}$ ($i=1,\ldots,n$; $\ell=0,\ldots,L-1$). Each fit selects exactly $L-1$ changes without cross-validation; we record selected count, empty-fit rate, conditional Hausdorff error divided by $n$, unconditional location-and-count loss, and runtime. Conditional population-risk calculations are retained only for the correctly specified $L=4$ fit."""

PRIOR_PAPER_DESCRIPTION = r"""Across 100 paired Monte Carlo trials (base seed 20260916), we used $n=500$, $p=100$, $x_i\sim N(0,I_p/n)$, $y_i=x_i^\top\beta_{s(i)}+\varepsilon_i$ with $\varepsilon_i\sim N(0,0.5^2)$, initial coefficients $N(0,(n/p)I_p)$, and independent jumps $N(0,1.5^2(n/p)I_p)$; the two true changes are at fractions $(0.30,0.55)$. We fit fixed-$L=3$ unpenalized least-squares WERM with half-squared-error loss and minimum gap $\Delta_n=25$, without cross-validation. Each prior is uniform over its admissible ordered pairs $1\le c_1<c_2\le n-1$ with $c_2-c_1\ge25$: Neutral includes every such pair; Bad restricts $(c_1/n,c_2/n)$ to $[0.175,0.425)\times[0.675,0.925)$; Good restricts it to $[0.175,0.405)\times[0.445,0.675)$, leaving an actual 0.04 gap between its support windows. Exact marginalization of each joint configuration prior gives the observation weights for the three regression fits; the final change-point search still covers all gap-admissible partitions, so prior support is not a hard constraint on estimates. The mean/median conditional Hausdorff errors divided by $n$ are Bad: $0.133/0.140$, Neutral: $0.027/0.014$, and Good: $0.006/0.004$; we also report the empirical change-point locations."""


def write_notebook(name, title, explanation, run_source, plot_source, paper_description):
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(title + "\n\n" + explanation +
                                 "\n\n### Paper-ready experiment description\n\n" +
                                 paper_description),
        nbf.v4.new_code_cell(COMMON + "\n" + run_source),
        nbf.v4.new_markdown_cell(
            "## Load and plot with tighter layout\n\n"
            "This cell loads the unique saved artifact; re-running plots does not rerun trials."),
        nbf.v4.new_code_cell(PLOT_COMMON + "\n" + plot_source),
    ])
    notebook.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    notebook.metadata["language_info"] = dict(name="python")
    if name == "E_PRIOR_misspecified_prior.ipynb":
        notebook.cells[2].source += (
            "\n\n### Interpreting the prior and the estimates\n\n"
            "The configuration prior is used to compute signal-occupancy weights for "
            "the three weighted least-squares fits; it is **not a hard constraint** "
            "on the final estimated change points. The final exact dynamic program "
            "minimizes loss over all partitions satisfying the minimum gap, and does "
            "not use the configuration prior as a posterior or MAP score. In the "
            "saved 100-trial run, the bad prior assigns its second change point only "
            "to 0.676–0.924, yet 33 of 100 fitted second change points are earlier. "
            "At the true second change (0.55), the bad prior assigns occupancy "
            "weights (0, 1, 0); nevertheless the third regression learns from later "
            "observations in the genuine third regime, and the loss search can switch "
            "to it near the truth. The empirical estimate distribution is therefore "
            "not the prior distribution.\n\n"
            "The 1-D plot now shows the location of a **uniformly selected change point** "
            "from each admissible pair, rather than plotting the ordered first and "
            "second locations separately. Under a uniform prior on distinct pairs, "
            "this pooled prior is flat when the minimum gap is Δ=1; the experiment "
            "uses Δ=25, so excluding nearby pairs leaves a mild endpoint effect. "
            "At location 0 itself there is zero prior mass: the first admissible index "
            "is 1/500=0.002. The previous plot connected that zero to a nearby "
            "scaled density just above 1. A density can exceed 1; the new y-axis "
            "shows probability mass per index instead, which sums to 1. Increasing "
            "n with Δ fixed at 25 would reduce the relative edge effect, but "
            "increasing n while keeping Δ/n=0.05 fixed would not. Changing n just "
            "for display would no longer show the prior used by this experiment. "
            "The good prior now has an **actual support gap** of 0.04 in fractional "
            "location, centered at 0.425: its first window ends at 0.405 and its "
            "second begins at 0.445. Both the joint configuration prior and the "
            "signal-occupancy marginals use these trimmed windows; no plotting-only "
            "gap or background shading is applied. "
            "The earlier 40-bin location-prior picture looked jagged partly because "
            "500 discrete indices were grouped into bins holding alternately about "
            "12 or 13 indices. The plot below instead shows the pooled **exact discrete** "
            "location mass at every observation index on a fractional x-axis, "
            "plus exact joint mass for each configuration. "
            "No artificial plotting-only increase of n or KDE smoothing is needed. "
            "The small neutral and good-prior edge effects that remain are genuine "
            "consequences of the minimum-gap constraint, not binning artifacts."
        )
        notebook.cells.insert(3, nbf.v4.new_markdown_cell(
            '### Appendix figure captions\n\n'
            '<span style="color:#4477AA">**Figure 15.** Location priors under bad, neutral, and good support choices. Dashed lines mark the true changes; see the Appendix subsection “Sensitivity to change point prior”.</span>\n\n'
            '<span style="color:#4477AA">**Figure 16.** Sample weights obtained by marginalizing each joint change-point prior over admissible configurations. See the Appendix subsection “Sensitivity to change point prior” for the prior construction.</span>\n\n'
            '<span style="color:#4477AA">**Figure 17.** Estimated change-point locations across the paired trials, with true locations marked by dashed lines. See the Appendix subsection “Sensitivity to change point prior” for the comparison.</span>'
        ))
    target = ROOT / name
    if target.exists():
        previous = nbf.read(target, as_version=4)
        for new_cell in notebook.cells:
            for old_cell in previous.cells:
                if (new_cell.cell_type == old_cell.cell_type == "code"
                        and new_cell.source == old_cell.source):
                    new_cell.outputs = old_cell.outputs
                    new_cell.execution_count = old_cell.execution_count
                    break
    nbf.write(notebook, ROOT / name)


if __name__ == "__main__":
  import sys
  targets = set(sys.argv[1:]) or {"prior"}
  if "el" in targets:
   raise RuntimeError(
    "E_L_misspecified_L.ipynb contains preserved original cells and an "
    "append-only CV extension; this builder must not rewrite it."
   )
  if "prior" in targets:
   write_notebook(
    "E_PRIOR_misspecified_prior.ipynb",
    "# E-PRIOR: bad, neutral, and good location priors",
    "100 paired linear-model trials; fixed L=3, changes at 0.30 and 0.55. "
    "Joint configuration priors are uniform over the DP-admissible pairs, "
    "then filtered by half-open 1/4-width windows, with a real 0.04 gap "
    "between the two trimmed good windows. The bad second window is "
    "centered at 0.80 and disjoint from the good second window. "
    "Each prior is explicitly marginalized into signal occupancy probabilities; "
    "those are passed to WERM and the matching conditional population-risk "
    "calculation. The population calculation is not a fixed-point theorem. "
    "Set WERM_SMOKE=1 to run two paired trials before full execution.",
    PRIOR, PRIOR_PLOT, PRIOR_PAPER_DESCRIPTION,
   )
  print("Built:", ", ".join(sorted(targets)))
