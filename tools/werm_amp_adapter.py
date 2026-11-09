"""Gaussian AMP adapter for the WERM comparison, without design whitening.

The denoisers and their derivatives are imported from AMP_chgpt_reg.  The
recursion is its ``GAMP_gaussian_clean`` recursion, with two bookkeeping fixes:
output-channel state expectations draw Gaussian noise (the upstream MC helper
currently passes the noise standard deviation as a constant observation
offset), and the final pseudo-data are paired with their *input* state.

Unlike the upstream exhaustive posterior, configuration MAP is computed by
exact dynamic programming. Its signal-label support matches the source MAP:
unused signals may be skipped and a no-change configuration can use any
working signal. The experiment supplies marginal weights from this same
count/location/label prior. No true coefficients or true boundaries enter
this module. The Gaussian prior and noise variance are supplied parameters.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import io
from pathlib import Path
import sys
import time
import types
import unicodedata

import numpy as np


_SOURCE = None


def _normalized(name):
    return unicodedata.normalize("NFKC", name)


def load_amp_source(amp_root=None):
    """Load the neighboring package without its broken Unicode star export.

    Python normalizes identifiers, but upstream ``__all__`` strings retain
    variant phi characters.  Bypassing only that convenience initializer
    leaves the actual algorithm modules unmodified on disk and in memory.
    """
    global _SOURCE
    if _SOURCE is not None:
        return _SOURCE
    root = (Path(amp_root) if amp_root is not None else
            Path(__file__).resolve().parents[2] / "AMP_chgpt_reg")
    package_path = root / "amp"
    if not (package_path / "marginal_separable_jax_linear.py").is_file():
        raise FileNotFoundError(f"AMP source not found in {package_path}")
    if "amp" in sys.modules:
        loaded = Path(sys.modules["amp"].__file__).resolve().parent
        if loaded != package_path.resolve():
            raise ImportError(f"Another AMP package is already loaded: {loaded}")
    else:
        package = types.ModuleType("amp")
        package.__file__ = str(package_path / "__init__.py")
        package.__path__ = [str(package_path)]
        package.__package__ = "amp"
        package.__spec__ = importlib.util.spec_from_file_location(
            "amp", package.__file__, submodule_search_locations=package.__path__)
        package.PAL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        setattr(package, _normalized("ϵ_0"), 1e-6)
        sys.modules["amp"] = package
    source = importlib.import_module("amp.marginal_separable_jax_linear")
    import jax
    # Upstream enables expensive debugging globally. Explicit finite checks
    # below retain failure detection without Python reruns of large JITs.
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_enable_x64", True)
    _SOURCE = source
    return source


def source_metadata():
    source = load_amp_source()
    path = Path(source.__file__)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "recursion": "GAMP_gaussian_clean", "denoiser": "Gaussian posterior mean",
            "corrections": ["Gaussian noise in output-channel MC",
                            "final pseudo-data/state iteration alignment",
                            "consistent marginal and MAP signal-label support"],
            "design_transform": "none"}


def _psd_square_root(a, name):
    a = (np.asarray(a) + np.asarray(a).T) / 2
    if not np.isfinite(a).all():
        raise FloatingPointError(f"{name} is nonfinite")
    vals, vecs = np.linalg.eigh(a)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(vals))))
    if vals.min() < -tolerance:
        raise FloatingPointError(f"{name} is indefinite: min eigenvalue {vals.min():.3g}")
    return vecs * np.sqrt(np.maximum(vals, 0.0))


def _finite(name, value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} is nonfinite")
    return value


def gaussian_row_loglik(theta, y, rho, noise_std, nu, kappa):
    """The upstream joint Gaussian row likelihood, excluding shared p(theta).

    Conditioning Y on theta removes a label-independent term and permits
    stable low-dimensional Gaussian likelihood evaluation.  Thus the MAP
    partition is exactly the same as using the joint likelihood.
    """
    theta, y = np.asarray(theta), np.asarray(y).reshape(-1)
    v_cov = nu.T @ np.linalg.solve(rho, nu) + kappa
    # Same 1e-10 Gaussian-density numerical ridge as the source denoiser.
    v_cov = (v_cov + v_cov.T) / 2 + 1e-10 * np.eye(len(rho))
    weights = np.linalg.solve(v_cov, nu.T)
    means = theta @ weights
    variances = np.diag(rho) + noise_std**2 + 1e-10 - np.sum(nu.T * weights, axis=0)
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise FloatingPointError("Nonpositive conditional Gaussian variance")
    return -0.5 * (np.log(2 * np.pi * variances)[None, :] +
                   (y[:, None] - means)**2 / variances[None, :])


def output_state_mc(source, phi, rho, noise_std, nu, kappa, rng,
                    mc_samples=16384, batch_size=4096):
    """Unbiased MC for the source E[g g^T], averaging labels exactly.

    Random observation indices integrate the finite average over phi columns.
    A shared Gaussian noise draw across labels is common-random-number
    integration, not a constant offset.  All draws use the local trial RNG.
    """
    import jax.numpy as jnp
    L, n = phi.shape
    root_rho = _psd_square_root(rho, "rho")
    root_kappa = _psd_square_root(kappa, "kappa")
    transform = np.linalg.solve(rho, nu)
    total = np.zeros((L, L))
    done = 0
    while done < mc_samples:
        size = min(batch_size, mc_samples - done)
        idx = rng.integers(n, size=size)
        marginal = phi[:, idx]
        z = rng.standard_normal((size, L)) @ root_rho.T
        v = z @ transform + rng.standard_normal((size, L)) @ root_kappa.T
        noise = noise_std * rng.standard_normal(size)
        for label in range(L):
            # Source E_Z prints tracing arrays; suppress that incidental output.
            with contextlib.redirect_stdout(io.StringIO()):
                g = source.g(jnp.asarray(v), jnp.asarray(z[:, label] + noise)[:, None],
                             jnp.asarray(marginal), noise_std, jnp.asarray(rho),
                             jnp.asarray(nu), jnp.asarray(kappa))
            g = _finite("state g", g).reshape(size, L)
            total += (g * marginal[label, :, None]).T @ g
        done += size
    return (total + total.T) / (2 * mc_samples)


def fit_amp(X, y, delta, noise_std, prior_cov, phi, gap, seed,
            iterations=10, mc_samples=16384, fixed_count=False,
            amp_root=None, return_details=False):
    """Run usual Gaussian AMP and exact MAP count/location estimation.

    ``X`` is used unchanged.  Its model normalization is Sigma/n; for unit
    marginal prediction variance the experiment uses prior_cov=delta*I.
    ``phi`` is (L,n), supplied by ``amp_prior_marginals`` under the same
    feasible-partition/count/signal-label prior as MAP.
    A failed or incomplete run has cp/count=None and status='failed'; it is
    never silently substituted by a better earlier iterate.
    """
    from tools.werm_amp_partition import amp_map_partition
    start = time.perf_counter()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    phi = np.asarray(phi, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)
    n, p = X.shape
    L = len(prior_cov)
    if not np.isclose(delta, n / p):
        raise ValueError("delta must equal n/p")
    if phi.shape != (L, n) or y.shape != (n, 1) or prior_cov.shape != (L, L):
        raise ValueError("Incompatible X/y/prior/phi dimensions")
    if np.any(phi < 0) or not np.allclose(phi.sum(axis=0), 1):
        raise ValueError("phi must be probability vectors")
    if noise_std <= 0 or iterations < 1 or mc_samples < 1:
        raise ValueError("noise_std, iterations and mc_samples must be positive")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Data must be finite")
    rho = prior_cov / delta
    prior_root = _psd_square_root(prior_cov, "prior_cov")
    if np.linalg.eigvalsh(rho).min() <= 0:
        raise ValueError("Gaussian prior covariance must be positive definite")
    # Match source tiny zero-marginal padding, then normalize exactly.
    phi = np.where(phi == 0, 1e-10 / n, phi)
    phi /= phi.sum(axis=0, keepdims=True)
    init_seed, state_seed = np.random.SeedSequence(seed).spawn(2)
    init_rng = np.random.default_rng(init_seed)
    state_rng = np.random.default_rng(state_seed)
    B = init_rng.standard_normal((p, L)) @ prior_root.T
    residual = np.zeros((n, L))
    nu, kappa, F = np.zeros((L, L)), rho.copy(), np.eye(L)
    diagnostics = []
    result = {"status": "failed", "cp": None, "count": None, "segments": None,
              "iterations_requested": int(iterations), "iterations_completed": 0,
              "mc_samples": int(mc_samples), "fixed_count": bool(fixed_count),
              "seed": int(seed), "design_transform": "none",
              "configuration_support": "source_MAP_signal_labels"}
    try:
        source = load_amp_source(amp_root)
        import jax.numpy as jnp
        derivative = getattr(source, _normalized("dgdθ"))
        for t in range(iterations):
            theta = _finite("theta", X @ B - residual @ F.T)
            # These are the state matrices for this theta, before the update.
            posterior_nu, posterior_kappa = nu.copy(), kappa.copy()
            with contextlib.redirect_stdout(io.StringIO()):
                residual = source.g(jnp.asarray(theta), jnp.asarray(y),
                    jnp.asarray(phi), noise_std, jnp.asarray(rho),
                    jnp.asarray(nu), jnp.asarray(kappa))
                jacobian = derivative(jnp.asarray(theta), jnp.asarray(y),
                    jnp.asarray(phi), noise_std, jnp.asarray(rho),
                    jnp.asarray(nu), jnp.asarray(kappa))
            residual = _finite("residual", residual).reshape(n, L)
            C = _finite("output derivative", jacobian).mean(axis=0)
            b_input = X.T @ residual - B @ C.T
            kappa_b = output_state_mc(source, phi, rho, noise_std, nu, kappa,
                                      state_rng, mc_samples=mc_samples)
            nu_hat = kappa_b
            B = _finite("B", source.f(jnp.asarray(b_input), delta,
                        jnp.asarray(rho), jnp.asarray(nu_hat), jnp.asarray(kappa_b)))
            B = B.reshape(p, L)
            chi = rho @ nu_hat @ np.linalg.pinv(nu_hat.T @ rho @ nu_hat + kappa_b / delta)
            F = chi.T / delta
            nu = rho @ nu_hat @ chi.T
            nu = (nu + nu.T) / 2
            kappa = nu - nu.T @ np.linalg.solve(rho, nu)
            kappa = (kappa + kappa.T) / 2
            _psd_square_root(kappa, "updated kappa")
            diagnostics.append({"iteration": t, "theta_rms": float(np.sqrt(np.mean(theta**2))),
                                "B_rms": float(np.sqrt(np.mean(B**2))),
                                "residual_rms": float(np.sqrt(np.mean(residual**2))),
                                "kappa_min_eigenvalue": float(np.linalg.eigvalsh(kappa).min())})
            result["iterations_completed"] = t + 1
        ll = _finite("posterior row loglik", gaussian_row_loglik(
            theta, y, rho, noise_std, posterior_nu, posterior_kappa))
        selected = amp_map_partition(ll, gap, fixed_count=fixed_count)
        result.update(selected)
        result["status"] = "ok"
        if return_details:
            result.update(theta=theta, posterior_nu=posterior_nu,
                          posterior_kappa=posterior_kappa, row_loglik=ll, B=B)
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["diagnostics"] = diagnostics
    result["runtime_seconds"] = time.perf_counter() - start
    return result
