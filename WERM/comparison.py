import numpy as np

### Import R interfaces
import rpy2
import rpy2.robjects as ro
## To aid in printing HTML in notebooks
import rpy2.ipython.html
# rpy2.ipython.html.init_printing()
from rpy2.robjects.packages import importr, data

from WERM.performance_measures import hausdorff
utils = importr('utils')
base = importr('base')
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
import WERM.DCDP_utils as DCDP_utils

def DP(X_unit, Y_unit, Δ, λs = [0.01, 0.1, 0.5, 1, 2, 4], γs = [0.01, 0.1, 1, 2, 4, 8, 16]):
    """ Dynamic Programming (DP) for changepoint regression (Rinaldo, Wang, Wen, Willett and Yu 2020) <arxiv:2010.10410>.
    X_unit: n x p matrix of covariates, where n is the number of observations and p is the number of covariates. Entries i.i.d N(0, 1).
    Y_unit: n x 1 vector of responses. Assumes a sparse signal β with sparsity d₀.
    Δ: maximum number of changepoints.
    λs: list of regularization parameters for the LASSO.
    γs: list of other parameters.
    
    Returns:
    DP_chgpt_est: estimated changepoint locations.
    TESTED.
    """
    
    with conversion.localconverter(converter):
        DP_res = chgpts.CV_search_DP_regression(np.array(Y_unit), np.array(X_unit), ro.FloatVector(γs), ro.FloatVector(λs), Δ, eps = 0.001)
        test_error = np.asarray(DP_res[2])
        assert test_error.shape == (len(γs), len(λs)), (
            f"unexpected DP test_error shape {test_error.shape}, "
            f"expected {(len(γs), len(λs))}"
        )
        gi, li = np.unravel_index(int(np.argmin(test_error)), test_error.shape)
        γ_min, λ_min = γs[gi], λs[li]

        # Run DP on the whole dataset and return the best changepoint
        best_res = chgpts.DP_regression(np.array(Y_unit), np.array(X_unit),γ_min, λ_min, Δ)
        return best_res[1] # The final idx contains the chgpt estimate list

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

        test_error = np.asarray(DPDU_res[2])
        assert test_error.shape == (len(ζs), len(λs)), (
            f"unexpected DPDU test_error shape {test_error.shape}, "
            f"expected {(len(ζs), len(λs))}"
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

def MOSEG(X_unit, Y_unit, num_chgpts=None, ms=False, debug=False): 
    """ MOSEG (Cho, Owens 2024). """
    
    # Set up headless graphics at the start to prevent X11 errors
    ro.r('options(device = function(...) png(tempfile(), ...))')
    
    try:
        with conversion.localconverter(converter):
            # Get the number of observations for validation
            if hasattr(X_unit, 'shape'):
                n = X_unit.shape[0]  # For numpy arrays
            else:
                try:
                    n = len(X_unit)
                except:
                    n = int(ro.r('length')(Y_unit)[0])
            
            if debug:
                print(f"Data dimensions: n={n}, num_chgpts={num_chgpts}")
                if hasattr(X_unit, 'shape'):
                    print(f"X shape: {X_unit.shape}")
                if hasattr(Y_unit, 'shape'):
                    print(f"Y shape: {Y_unit.shape}")
            
            # Validate num_chgpts constraint: G must be at most n/2
            max_chgpts = n // 2
            if num_chgpts is not None and num_chgpts > max_chgpts:
                if debug:
                    print(f"Warning: num_chgpts ({num_chgpts}) exceeds maximum allowed ({max_chgpts}). Setting to {max_chgpts}")
                num_chgpts = max_chgpts
            
            # Additional validation
            if n < 4:
                raise ValueError(f"Insufficient data: only {n} observations. Need at least 4.")
            
            if num_chgpts is not None and num_chgpts <= 0:
                if debug:
                    print("Warning: num_chgpts <= 0, using cross-validation instead")
                num_chgpts = None

            try:
                if num_chgpts is not None:
                    if not ms:
                        # Method 1: Using underscore conversion (recommended)
                        # rpy2 automatically converts underscores to dots in parameter names
                        out = moseg_package.moseg(X_unit, Y_unit, n_cps=num_chgpts, do_scale=True, do_plot = False)
                        
                    else: 
                        # Access the moseg.ms function using getattr
                        moseg_ms = getattr(moseg_package, 'moseg.ms')
                        
                        # Create R vector for the multiscale parameters
                        r_vector = ro.IntVector([40, 80])
                        
                        # Call the function with proper parameter handling
                        out = moseg_ms(X_unit, Y_unit, r_vector, do_scale=False, do_plot=False)

                else:
                    # Handle case where num_chgpts is None
                    # You might want to use cross-validation or other methods
                    if not ms:
                        # Example: using default parameters or CV
                        moseg_cv = getattr(moseg_package, 'moseg.cv')
                        out = moseg_cv(X_unit, Y_unit, penalty=50, do_scale=False, do_plot=False)
                    else:
                        # moseg_ms_cv = getattr(moseg_package, 'moseg.ms.cv')
                        moseg_ms_cv = moseg_package.moseg_ms_cv

                        # r_vector = ro.IntVector([50, 100, 150])
                        # out = moseg_ms_cv(X_unit, Y_unit, r_vector, do_scale=False, do_plot=False)
                        r_vector = ro.IntVector([40, 80]) # Allows us to work with delta = 1.0, otherwise there is an error
                        out = moseg_ms_cv(X_unit, Y_unit, r_vector, do_plot=False)
                        # out = moseg_ms_cv(X_unit, Y_unit, do_plot=False)

            except Exception as e:
                if debug:
                    print(f"Error in MOSEG call: {e}")
                # Fallback to cross-validation
                try:
                    # moseg_cv = getattr(moseg_package, 'moseg.ms.cv')
                    moseg_cv = moseg_package.moseg_ms_cv
                    r_vector = ro.IntVector([40, 80])
                    out = moseg_cv(X_unit, Y_unit, r_vector, do_plot=False)
                    # out = moseg_cv(X_unit, Y_unit, do_plot=False)
                except Exception as e2:
                    if debug:
                        print(f"Fallback also failed: {e2}")
                    raise e2

            return out
    
    finally:
        # Clean up graphics devices when done (even if an error occurs)
        try:
            ro.r('graphics.off()')
        except:
            pass  # Ignore errors during cleanup

### --- Logistic Regression Wang, Liu, Zhang, Liu 2023 --- ###

# glm_package = importr('myglm') # associated with paper by Wang, Liu, Zhang, Liu 2023
glm_package = importr('myglm2') # associated with paper by Wang, Liu, Zhang, Liu 2023, explicitly pass in lambda parameter into glmnet for more numerical stability
glm3_package = importr('myglm3') # fidelity-repaired automatic BSA; see myglm-3/PROVENANCE.md

# def glm_bsa(X_unit, Y_unit, kmax, c1, m1, delta):
def glm_bsa(X, Y, kmax=3, delta=0.1, is_exact=False):
    """Efficient multiple change point detection for high-dimensional 
    generalized linear models by Wang, Liu, Zhang, Liu 2023
    This paper proposed three algs, I pick the most versatile and fastest: 
    binary segmentation algorithm (bsa) to compare with ours.

    X doesn't need to be iid standard Gaussian, because this alg works
    for any covariance of X.

    is_exact: if True, the algorithm searches through configs with exactly kmax chgpts
    if False, it searches through configs with at most kmax chgpts, and 
    at least 1 (not zero) chgpts. However the way the authors coded this bsa 
    makes it possible for the alg to return 0 chgpts.
    """
    
    with conversion.localconverter(converter):
        # For some reason, the np.array() wrappers are necessary for this function to run.
        if is_exact:
            print(f"Running bsa with exactly kmax={kmax} chgpts.")
            res = glm_package.bsa_exact_wrapper(X, Y, kmax, delta)
        else:
            print(f"Running bsa with at most kmax={kmax} chgpts, at least 1 chgpt.")
            res = glm_package.bsa_wrapper(X, Y, kmax, delta)
        # res = glm_package.bsachgpt(X_unit, Y_unit, kmax, c1, m1, delta) # this is a list
        if res == rpy2.rinterface.NULL:
            print("glm_bsa res is NULL.")
            raise ValueError("BSA algorithm failed to converge.")
            # return np.array([]) # Return empty array if no changepoints found, original
        num_cp = res[0]
        cp_loc = res[1]
        print("Estimated number of changepoints: ", num_cp)
        print("Estimated changepoint locations: ", cp_loc)
        return cp_loc

def glm_bsa2(X, Y, kmax=2, c1=0.18, m1 = None, delta = 0.1, exact=False):
    """
    #' This function provides change point estimation by Binary Segmentation Algorithm
    #'  for the high dimensional generalized linear regression model.
    #' @export
    #' @param X an n*p data matrix, n=observation, p=dimension
    #' @param Y an n dimensional data vector, n=observation
    #' @param kmax the maximum change point number.
    #' @param c1 a constant specified by users.
    #' @param m1 the sample size of the minimal interval.
    #' @param delta the minimal interval length specified by users.
    #' @return  A list including: the estimated change point number, the estimated
    #'   change point locations.
    #' @importFrom stats coef
    #' @importFrom glmnet glmnet
    """

    n, p = X.shape
    # Set default parameters like they do in their R code
    if m1 is None: 
        m1 = np.ceil(n * delta)

    with conversion.localconverter(converter):
        # For some reason, the np.array() wrappers are necessary for this function to run.
        if not exact:
            print("Running bsa_chgpt with at most kmax=", kmax, " chgpts.")
            res = glm_package.bsa_chgpt(X, Y, kmax, c1, m1, delta)
        else:
            print("Running bsa_exactchgpt with exactly kmax=", kmax, " chgpts.")
            res = glm_package.bsa_exactchgpt(X, Y, kmax, c1, m1, delta)
        assert len(res) == 2, "Expected result to be a list of length 2."
        num_cps, cp_loc = res
    return cp_loc


def glm_bsa3(X, Y, kmax=2, c1=0.18, m1=None, delta=0.1, exact=False):
    """Run the repo-local myglm3 BSA without altering the legacy myglm2 adapter.

    The automatic routine restores the original no-split candidate. The exact
    routine is exposed only for compatibility and is not a true-count oracle.
    """
    n, _ = X.shape
    if m1 is None:
        m1 = int(np.ceil(n * delta))

    with conversion.localconverter(converter):
        if not exact:
            print("Running myglm3 bsa_chgpt with at most kmax=", kmax, " chgpts.")
            res = glm3_package.bsa_chgpt(X, Y, kmax, c1, m1, delta)
        else:
            print("Running myglm3 bsa_exactchgpt with kmax=", kmax, " chgpts.")
            res = glm3_package.bsa_exactchgpt(X, Y, kmax, c1, m1, delta)
        assert len(res) == 2, "Expected result to be a list of length 2."
        _, cp_loc = res
    return np.asarray(cp_loc, dtype=int).reshape(-1)


### --- Algorithms for sparse difference priors --- %%%

def run_charcoal_updated(X_unit, Y_unit, σ, η_true):
    """
    X_unit is nxp, Y is nx1. AMP δ is n/p.
    σ is AMP's noise level. charcoal uses noise level σ * sqrt(n).
    η_true is the true chgpt locations.
    
    Compared to three methods above, this function returns 1/n*hausdorff 
    distance from η_true directly, but can be modified to return the 
    estimated changepoint locations.
    
    Returns:
    cp (numpy.ndarray): Estimated changepoint locations. Returns np.array([-1]) if no changepoints are found or an error occurs.
    haus (float): Normalized Hausdorff distance. Returns np.nan if no changepoints are found or an error occurs.
    """
    
    # Define default fallback values
    default_cp = np.array([-1])
    default_haus = np.nan

    with conversion.localconverter(converter):
        (n, p) = X_unit.shape
        assert len(Y_unit) == n
        
        if n < p:
            # GW22 needs n > p, otherwise charcoal crashes
            print(f"Warning: n ({n}) < p ({p}). Charcoal method requires n >= p. Returning default fallback.")
            return default_cp, default_haus
        
        try:
            charcoal_func = charcoal.not_cpreg 
            # print(f"Successfully retrieved charcoal_func: {charcoal_func}") # Commenting out verbose debugging prints for cleaner output
            # print(f"Type of charcoal_func: {type(charcoal_func)}")

        except AttributeError as e:
            print(f"AttributeError: Could not find 'not_cpreg' in 'charcoal' package: {e}")
            return default_cp, default_haus # Return default on function retrieval error

        try:
            charcoal_res = charcoal_func(X_unit, Y_unit, sigma=σ*np.sqrt(n), verbose=True)
            # print(f"charcoal.not_cpreg call successful. Result type: {type(charcoal_res)}") # Commenting out
            # print(f"charcoal_res content: {charcoal_res}") # Commenting out

        except rpy2.rinterface.RRuntimeError as e:
            print(f"R runtime error during charcoal.not_cpreg call: {e}")
            print("This could indicate issues with input types, R function internal errors, etc.")
            return default_cp, default_haus # Return default on R runtime error
        except Exception as e:
            print(f"An unexpected Python error occurred during charcoal.not_cpreg call: {e}")
            return default_cp, default_haus # Return default on other Python errors

        if charcoal_res == rpy2.rinterface.NULL:
            print("charcoal_res is NULL. No changepoints found or algorithm failed to converge. Returning default fallback.")
            return default_cp, default_haus # Return default if charcoal returns NULL
        else: 
            # Validate output structure
            if not isinstance(charcoal_res, (list, tuple, ro.vectors.ListVector)) or len(charcoal_res) != 5:
                print(f"Warning: charcoal_res is not a list/tuple of length 5 as expected. Actual type: {type(charcoal_res)}, length: {len(charcoal_res) if hasattr(charcoal_res, '__len__') else 'N/A'}. Returning default fallback.")
                # print(f"Raw charcoal_res: {charcoal_res}") # Uncomment for deeper debugging if needed
                return default_cp, default_haus

            # Process valid results
            stage_str_list = ["initial", "test_refine", "midpoint_refine", "final"]
            stage_cp_list = [] # initial, test_refine, midpoint_refine, final estimate of chgpt indices
            stage_hausdorff_list = []
            
            for i_stage, stage_str in enumerate(stage_str_list):
                # print(f"========== {i_stage}/4 {stage_str} ==========") # Commenting out
                current_stage_res = charcoal_res[i_stage+1]
                
                if current_stage_res == rpy2.rinterface.NULL:
                    # print(f"charcoal {stage_str} result is NULL.") # Commenting out
                    stage_hausdorff_list.append(np.inf) # Keep inf for internal stages if null, indicates no estimate for that stage
                    stage_cp_list.append(np.array([])) # Empty array if NULL for the stage
                else:
                    try:
                        cp_stats = np.asarray(current_stage_res) 
                        
                        # Ensure cp_stats is not empty and has a shape compatible with indexing
                        if cp_stats.ndim >= 1 and cp_stats.shape[0] > 0:
                            # Assuming cp_stats[0, :] extracts the changepoint locations
                            stage_cp_list.append(cp_stats[0, :].flatten())
                        else:
                            # Handle cases where cp_stats is an empty array or a single value not suitable for [0,:]
                            stage_cp_list.append(np.array([]))
                        
                        # print(f"cp (0-indexing i.e. Python indexing," + \ # Commenting out
                        #       f" idx of 1st observation of new signal): \n{stage_str}={stage_cp_list[i_stage]},") # Commenting out
                        
                        # Calculate Hausdorff distance only if there are estimated changepoints
                        if stage_cp_list[i_stage].size > 0:
                            tmp_haus = 1/n * hausdorff(η_true, stage_cp_list[i_stage])
                            stage_hausdorff_list.append(tmp_haus)
                        else:
                            stage_hausdorff_list.append(np.inf) # No CPs found for this stage, so infinite distance
                            
                    except Exception as inner_e:
                        print(f"Error processing stage {stage_str} result: {inner_e}")
                        stage_hausdorff_list.append(np.inf)
                        stage_cp_list.append(np.array([])) # Fallback if internal processing fails
                # print(f"Normalised hausdorff distance: {stage_hausdorff_list[i_stage]}") # Commenting out
            
            # Determine final changepoint estimate and Hausdorff distance
            # If the final stage produced no CPs (empty array), return the default fallback values
            final_cp_est = stage_cp_list[-1] if stage_cp_list and stage_cp_list[-1].size > 0 else default_cp
            haus = stage_hausdorff_list[-1] if stage_hausdorff_list and stage_cp_list[-1].size > 0 else default_haus
            
            # If final_cp_est is np.array([-1]), ensure haus is np.nan
            if np.array_equal(final_cp_est, default_cp):
                haus = default_haus

        return final_cp_est, haus

_CHARCOAL_THRESH_CACHE = {}


def _get_charcoal_threshold(n, p):
    """Return charcoal's cached package-default threshold for an (n, p) design."""
    if n <= p:
        return None

    key = (int(n), int(p))
    if key not in _CHARCOAL_THRESH_CACHE:
        no_intervals = n // 5
        with conversion.localconverter(converter):
            threshold = charcoal.getTestThreshold(
                n,
                p,
                burnIn=0,
                alpha=0.01 / no_intervals,
                permSize=1000,
                sigma=1.0,
                b=1,
                verbose=False,
            )
        _CHARCOAL_THRESH_CACHE[key] = float(threshold[0][0])
    return _CHARCOAL_THRESH_CACHE[key]


def run_charcoal(X_unit, Y_unit, σ, η_true):
    """Run charcoal with its Bonferroni-adjusted package-default threshold."""
    del σ, η_true  # kept in the public signature for legacy call compatibility
    n, p = X_unit.shape
    assert len(Y_unit) == n
    if n <= p:
        # Complementary sketching requires n > p: this is not an empty estimate.
        return None

    threshold = _get_charcoal_threshold(n, p)
    with conversion.localconverter(converter):
        result = charcoal.not_cpreg(
            X_unit, Y_unit, thresh=threshold, verbose=False
        )

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
def convert_to_r_matrix(X):
    """Convert numpy array to R matrix using modern rpy2"""
    with conversion.localconverter(converter):
        if isinstance(X, np.ndarray):
            return ro.r.matrix(ro.FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1])
        else:
            return X

def convert_to_r_vector(y):
    """Convert numpy array to R vector using modern rpy2"""
    with conversion.localconverter(converter):
        if isinstance(y, np.ndarray):
            return ro.FloatVector(y)
        else:
            return y
