import numpy as np

### Import R interfaces
import rpy2
import rpy2.robjects as ro
## To aid in printing HTML in notebooks
import rpy2.ipython.html
# rpy2.ipython.html.init_printing()
from rpy2.robjects.packages import importr, data

charcoal = importr('charcoal')

# MODERN rpy2: Use conversion context instead of deprecated activate/deactivate
import rpy2.robjects.numpy2ri
from rpy2.robjects import conversion, default_converter

# Create converter with numpy support
converter = default_converter + rpy2.robjects.numpy2ri.converter

# Import DPDU (R package)
chgpts = importr('changepoints')
inferchange_package = importr('inferchange') # McScan is a part of inferchange: https://github.com/tobiaskley/inferchange
moseg_package = importr('moseg') # MOSEG package. 
_moseg_native_serial = ro.r('''local({
    # The experiment parallelizes trials. Native moseg normally starts a
    # one-worker socket cluster even with ncores=1. Use the identical function
    # body with a scoped sequential foreach backend instead.
    f <- moseg::moseg
    e <- new.env(parent=environment(f))
    e$makeCluster <- function(...) NULL
    e$registerDoParallel <- function(...) foreach::registerDoSEQ()
    e$stopCluster <- function(...) invisible(NULL)
    environment(f) <- e
    f
})''')
import WERM.DCDP_utils as DCDP_utils


def _r_cv_matrix(value, expected_shape, method_name):
    """Recover an R CV list-matrix after rpy2 drops its dimensions."""
    array = np.asarray(value, dtype=float)
    if array.shape == expected_shape:
        return array
    expected_size = int(np.prod(expected_shape))
    assert array.size == expected_size, (
        f"unexpected {method_name} test_error size {array.size}, "
        f"expected {expected_size} for shape {expected_shape}"
    )
    return array.reshape(-1).reshape(expected_shape, order="F")


def DCDP(X_unit, Y_unit,
         λ_list = [0.01, 0.1, 1, 5, 10],
         γ_list = list(np.logspace(0, 3, 7))):
    """ Divide and Conquer Dynamic Programming (DCDP) for changepoint detection (Li, Wang, Rinaldo 2023). 
    X_unit: n x p matrix of covariates, where n is the number of observations and p is the number of covariates. Entries i.i.d N(0, 1).
    Y_unit: n x 1 vector of responses. Assumes a sparse signal β with sparsity d₀.
    λ_list: list of regularization parameters for the LASSO.
    γ_list: list of other parameters.

    Returns:
    cp_best: estimated changepoint locations.
    TESTED.
    """

    # DCDP's loss_func computes `Y - X @ beta` where `X @ beta` is (m, 1).
    # A 1-D Y broadcasts to (m, m) and silently destroys the objective.
    Y_unit = np.asarray(Y_unit).reshape(-1, 1)
    assert Y_unit.ndim == 2 and Y_unit.shape[1] == 1, "DCDP requires Y of shape (n, 1)"
    n = Y_unit.shape[0]
    # The algorithm requires dividing the training set into two parts, 
    # so we do that here:
    Y_train = Y_unit[np.arange(0, n, 2)]
    Y_test = Y_unit[np.arange(1, n, 2)]
    X_train = X_unit[np.arange(0, n, 2), :]
    X_test = X_unit[np.arange(1, n, 2), :]
    grid_n = 100
    dcdp = DCDP_utils.dcdp_cv_grid_linear(grid_n, λ_list, γ_list, smooth = 2, 
                    buffer = 2, step_refine = 1, buffer_refine = 2, lam_refine = 0.1)
    cp_best, param_best, cp_best_cand = dcdp.fit((Y_train, X_train), (Y_test, X_test))

    # Run the algorithm on the full dataset and return the best changepoint
    best_lam, best_gamma = param_best
    if best_gamma in (min(γ_list), max(γ_list)):
        print(f"WARNING: DCDP selected gamma at grid boundary ({best_gamma}); widen γ_list")
    n_half = len(Y_train)
    gamma_full = best_gamma * (n / n_half)
    grid_n = min(dcdp.grid_n, n - 1)
    step = n / (grid_n + 1)
    grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)
    cp_loc, obj = dcdp.dp_grid((Y_unit, X_unit), grid, best_lam, gamma_full)
    cp_loc_refined = dcdp.local_refine((Y_unit, X_unit), cp_loc)
    return np.asarray(cp_loc_refined, dtype=int).reshape(-1)

def DPDU(X_unit, Y_unit, λs = [0.01, 0.1, 1, 2], ζs = [10.0, 15.0, 20.0]):
    """ Dynamic Programming with Dynamic Updates (Xu, Wang, Zhao, Yu 2022).
    X_unit: n x p matrix of covariates, where n is the number of observations and p is the number of covariates. Entries i.i.d N(0, 1). 
    Y_unit: n x 1 vector of responses. Assumes a sparse signal β with sparsity d₀. 
    λs: list of regularization parameters for the LASSO.
    ζs: list of other parameters. 

    Returns:
    cpt_est_DPDU: estimated changepoint locations.
    """

    with conversion.localconverter(converter):
        DPDU_res = chgpts.CV_search_DPDU_regression(np.array(Y_unit), np.array(X_unit), 
                        ro.FloatVector(λs), ro.FloatVector(ζs), eps = 0.001)

        test_error = _r_cv_matrix(
            DPDU_res[2], (len(ζs), len(λs)), method_name="DPDU"
        )

        zi, li = np.unravel_index(int(np.argmin(test_error)), test_error.shape)
        ζ_min, λ_min = ζs[zi], λs[li]

        # R stores the CV result matrices/lists in column-major order.
        r_flat = li * len(ζs) + zi
        cpt_est_DPDU = DPDU_res[0][r_flat]
        B̂_DPDU = DPDU_res[4][r_flat]

        # McScan Section D.2.1 returns the locally refined CV estimate.
        if len(cpt_est_DPDU) > 0:
            cpt_final = chgpts.local_refine_DPDU_regression(
                cpt_est_DPDU, B̂_DPDU, np.array(Y_unit), X_unit, w=0.9
            )
        else:
            cpt_final = cpt_est_DPDU
        return np.asarray(cpt_final, dtype=int).reshape(-1)

def _moseg_getG(p, n):
    """Port of moseg:::getG, the package's default bandwidth rule."""
    err = 0.012
    c = (-0.01538316, 0.02009430, 0.02669467)
    out = np.exp(
        err / c[0]
        - (c[1] / c[0]) * np.log(np.log(p))
        - (c[2] / c[0]) * np.log(np.log(n))
    )
    return int(round(out))


def _moseg_default_Gset(p, n):
    """Return the package default Gset clipped to moseg's n/2 constraint."""
    G = _moseg_getG(p, n)
    candidates = [int(round(g)) for g in (G, 4 / 3 * G, 5 / 3 * G)]
    return sorted({min(g, n // 2) for g in candidates if g > 0})


def MOSEG(X_unit, Y_unit, num_chgpts=None, ms=False, debug=False):
    """Run MOSEG without silently changing routines after a failure."""
    ro.r('options(device = function(...) png(tempfile(), ...))')

    try:
        with conversion.localconverter(converter):
            n = X_unit.shape[0] if hasattr(X_unit, "shape") else len(X_unit)
            if debug:
                print(f"Data dimensions: n={n}, num_chgpts={num_chgpts}")
            if n < 4:
                raise ValueError(
                    f"Insufficient data: only {n} observations. Need at least 4."
                )

            max_chgpts = n // 2
            if num_chgpts is not None and num_chgpts > max_chgpts:
                if debug:
                    print(
                        f"Warning: num_chgpts ({num_chgpts}) exceeds "
                        f"maximum ({max_chgpts}); clipping."
                    )
                num_chgpts = max_chgpts
            if num_chgpts is not None and num_chgpts <= 0:
                if debug:
                    print("Warning: num_chgpts <= 0, using cross-validation.")
                num_chgpts = None

            if num_chgpts is not None:
                if not ms:
                    return _moseg_native_serial(
                        X_unit,
                        Y_unit,
                        n_cps=num_chgpts,
                        do_scale=True,
                        do_plot=False,
                    )
                moseg_ms = getattr(moseg_package, "moseg.ms")
                return moseg_ms(
                    X_unit,
                    Y_unit,
                    ro.IntVector([40, 80]),
                    do_scale=False,
                    do_plot=False,
                )

            if not ms:
                moseg_cv = getattr(moseg_package, "moseg.cv")
                return moseg_cv(
                    X_unit, Y_unit, do_scale=False, do_plot=False
                )

            Gset = _moseg_default_Gset(X_unit.shape[1], X_unit.shape[0])
            print(
                "MOSEG effective Gset (package default, clipped to n/2): "
                f"{Gset}"
            )
            return moseg_package.moseg_ms_cv(
                X_unit, Y_unit, ro.IntVector(Gset), do_plot=False
            )
    finally:
        try:
            ro.r("graphics.off()")
        except Exception:
            pass


### --- Logistic Regression Wang, Liu, Zhang, Liu 2023 --- ###

# glm_package = importr('myglm') # associated with paper by Wang, Liu, Zhang, Liu 2023
glm3_package = importr('myglm3') # fidelity-repaired automatic BSA; see myglm-3/PROVENANCE.md


# Validation refits for GLMBSA-CV use the same glmnet call as myglm3. Keeping
# this as a small R helper avoids rpy2's ambiguous translation of R's
# reserved-word lambda argument.
_glm_bsa3_coef_r = ro.r(
    """
    function(X, Y, lam) {
      fit <- glmnet::glmnet(
        X, as.numeric(Y), intercept=FALSE, family="binomial", lambda=lam
      )
      as.vector(stats::coef(fit))[-1]
    }
    """
)

_glm_bsa3_quiet_r = ro.r(
    """
    function(X, Y, kmax, c1, m1, delta) {
      suppressMessages(
        myglm3::bsa_chgpt(X, Y, kmax, c1, m1, delta)
      )
    }
    """
)



def glm_bsa3_lambda(c1, p, segment_length):
    """Return myglm3's segment-specific glmnet penalty."""
    c1 = float(c1)
    p = int(p)
    segment_length = int(segment_length)
    if not np.isfinite(c1) or c1 <= 0:
        raise ValueError("c1 must be finite and positive")
    if p < 1 or segment_length < 1:
        raise ValueError("p and segment_length must be positive")
    log_term = np.log(2.0 * p)
    return c1 * (
        np.sqrt(2.0 * log_term / segment_length)
        + log_term / segment_length
    )


def glm_bsa3_segment_beta(X, Y, c1):
    """Fit one intercept-free logistic segment exactly as myglm3 does."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != Y.size:
        raise ValueError("X and Y must have matching sample dimensions")
    lam = glm_bsa3_lambda(c1, X.shape[1], X.shape[0])
    with conversion.localconverter(converter):
        beta = _glm_bsa3_coef_r(X, Y, lam)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if beta.size != X.shape[1] or not np.all(np.isfinite(beta)):
        raise ValueError("glmnet returned invalid GLMBSA segment coefficients")
    return beta


# def glm_bsa(X_unit, Y_unit, kmax, c1, m1, delta):



def glm_bsa3(
    X, Y, kmax=2, c1=0.18, m1=None, delta=0.1, exact=False, quiet=False
):
    """Run the repo-local myglm3 BSA without altering the legacy myglm2 adapter.
    The automatic routine restores the original no-split candidate. Here
    ``kmax`` controls binary-segmentation passes; because every active interval
    may split in a pass, it is not a strict changepoint-count cap. The exact
    routine is exposed only for compatibility and is not a true-count oracle.
    """

    n, _ = X.shape
    if m1 is None:
        m1 = int(np.ceil(n * delta))

    with conversion.localconverter(converter):
        if not exact:
            if quiet:
                res = _glm_bsa3_quiet_r(X, Y, kmax, c1, m1, delta)
            else:
                print("Running myglm3 bsa_chgpt with kmax=", kmax, " binary-segmentation passes (not a strict changepoint-count cap).")
                res = glm3_package.bsa_chgpt(X, Y, kmax, c1, m1, delta)
        else:
            print("Running myglm3 bsa_exactchgpt with kmax=", kmax, " chgpts.")
            res = glm3_package.bsa_exactchgpt(X, Y, kmax, c1, m1, delta)
        assert len(res) == 2, "Expected result to be a list of length 2."
        _, cp_loc = res
    return np.asarray(cp_loc, dtype=int).reshape(-1)


### --- Algorithms for sparse difference priors --- %%%


CHARCOAL_THRESHOLD = 10.0
CHARCOAL_SEED = 42




def run_charcoal(X_unit, Y_unit):
    """Run charcoal without oracle inputs, using its documented fixed threshold.

    The charcoal help page demonstrates ``not_cpreg(..., thresh=10)``.  This
    avoids the very unstable GEV extrapolation used when ``alpha * permSize <
    1`` (the previous one-shot cache produced thresholds from about 20 to 43
    for this experiment).  Omitting ``sigma`` deliberately leaves scale
    estimation to charcoal rather than leaking the simulation noise level.
    A fixed interval seed makes reruns reproducible without depending on the
    true changepoints or any other DGP parameter.
    """
    n, p = X_unit.shape
    assert len(Y_unit) == n
    if n <= p:
        # Complementary sketching requires n > p: this is not an empty estimate.
        return None

    with conversion.localconverter(converter):
        result = charcoal.not_cpreg(
            X_unit,
            Y_unit,
            thresh=CHARCOAL_THRESHOLD,
            verbose=False,
            seed=CHARCOAL_SEED,
        )

    if result is ro.NULL:
        return np.array([], dtype=int)

    if np.asarray(result[-1]).size == 0:
        return np.array([], dtype=int)
    return np.asarray([item[0] for item in result[-1]], dtype=int).reshape(-1)


def run_mcscan(X_unit, Y_unit, num_chgpts = None, method='not'):
    """ Runs the McScan procedure from inferchange R package in this paper: https://arxiv.org/abs/2402.06915. 
    X_unit is nxp, Y is nx1.
    If num_chgpts is None, McScan estimates the number of changepoints in addition to location.
    If num_chgpts is an integer, McScan uses that number of changepoints and only estimates location.

    method is 'not' (fixed threshold) or 'auto' (automatic threshold estimation).
    """

    # set.seed(12345)
    # data <- dgp_gauss_sparse(n = 200, p = 20, z = 100, s = 3, rho = 1, sigma = 1)
    # X <- data$X
    # y <- data$y
    # res <- inferchange(X, y)

    with conversion.localconverter(converter):
        if num_chgpts is not None:
            print(f"Running McScan with upper bound num_chgpts = {num_chgpts}.")
            res = inferchange_package.McScan(X_unit, Y_unit, num_chgpts)
            cp = res[0]
        else:
            print("Running McScan with automatic changepoint estimation.")
            # res = inferchange_package.inferchange(X_unit, Y_unit)
            # cp = res[0]
            res = inferchange_package.McScan(X_unit, Y_unit, method=method)
            cp = res[0]  # Extract the estimated changepoints from the result
            print("Estimated changepoints from McScan: ", cp)
            # delta = res[1]
            # ci = res[2]
        return cp

# Helper functions for modern rpy2 conversion

