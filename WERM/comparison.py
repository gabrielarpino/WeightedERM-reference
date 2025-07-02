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
# charcoal = importr('charcoal')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

## Import DPDU (R package)
chgpts = importr('changepoints')
inferchange_package = importr('inferchange') # McScan is a part of inferchange: https://github.com/tobiaskley/inferchange
moseg_package = importr('moseg') # MOSEG package. 

## Import DCDP (Python file in home directory)
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

    DP_res = chgpts.CV_search_DP_regression(np.array(Y_unit), np.array(X_unit), ro.FloatVector(γs), ro.FloatVector(λs), Δ, eps = 0.001)
    DP_cpt_hat = DP_res[0]
    DP_test_error = DP_res[2]
    min_idx_unravel = np.array(np.unravel_index(np.argmin(DP_test_error, axis=None), (len(λs), len(γs)))).astype(int)
    min_idx_flat = np.argmin(DP_test_error, axis=None)
    λ_min = λs[min_idx_unravel[0]]
    γ_min = γs[min_idx_unravel[1]]
    DP_chgpt_est = DP_cpt_hat[int(min_idx_flat)]

    # Run DP on the whole dataset and return the best changepoint
    best_res = chgpts.DP_regression(np.array(Y_unit), np.array(X_unit),γ_min, λ_min, Δ)
    return best_res[1] # The final idx contains the chgpt estimate list

def DCDP(X_unit, Y_unit, λ_list = [0.1, 1, 5, 10], γ_list = [100, 500, 1000]):
    """ Divide and Conquer Dynamic Programming (DCDP) for changepoint detection (Li, Wang, Rinaldo 2023). 
    X_unit: n x p matrix of covariates, where n is the number of observations and p is the number of covariates. Entries i.i.d N(0, 1).
    Y_unit: n x 1 vector of responses. Assumes a sparse signal β with sparsity d₀.
    λ_list: list of regularization parameters for the LASSO.
    γ_list: list of other parameters.

    Returns:
    cp_best: estimated changepoint locations.
    TESTED.
    """

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
    grid_n = min(dcdp.grid_n, n - 1)
    step = n / (grid_n + 1)
    grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)
    cp_loc, obj = dcdp.dp_grid((Y_unit, X_unit), grid, best_lam, best_gamma)
    cp_loc_refined = dcdp.local_refine((Y_unit, X_unit), cp_loc)
    return cp_loc_refined

def DPDU(X_unit, Y_unit, λs = [0.3, 0.5, 1, 2], ζs = [10.0, 15.0, 20.0]):
    """ Dynamic Programming with Dynamic Updates (Xu, Wang, Zhao, Yu 2022).
    X_unit: n x p matrix of covariates, where n is the number of observations and p is the number of covariates. Entries i.i.d N(0, 1). 
    Y_unit: n x 1 vector of responses. Assumes a sparse signal β with sparsity d₀. 
    λs: list of regularization parameters for the LASSO.
    ζs: list of other parameters. 

    Returns:
    cpt_est_DPDU: estimated changepoint locations.
    """

    DPDU_res = chgpts.CV_search_DPDU_regression(np.array(Y_unit), np.array(X_unit), 
                    ro.FloatVector(λs), ro.FloatVector(ζs), eps = 0.001)

    # find the indices of gamma_set and lambda_set which minimizes the test error
    test_error = np.array(DPDU_res[2])
    cpt_hat = (DPDU_res[0])
    B̂_list = (DPDU_res[4])
    min_idx_unravel = np.array(np.unravel_index(np.argmin(test_error, axis=None), (len(λs), len(ζs)))).astype(int)
    min_idx_flat = np.argmin(test_error, axis=None)
    λ_min = λs[min_idx_unravel[0]]
    ζ_min = ζs[min_idx_unravel[1]]
    cpt_est_DPDU = cpt_hat[int(min_idx_flat)]
    B̂_DPDU_original = B̂_list[int(min_idx_flat)]

    # Refine changepoint estimate
    if len(cpt_est_DPDU) > 0:
        cpt_LR = chgpts.local_refine_DPDU_regression(cpt_est_DPDU, 
            B̂_DPDU_original, np.array(Y_unit), X_unit, w = 0.9)
        cpt_est_DPDU = cpt_LR

    # Compute the chgpt estimate on the full data
    DPDU_res_full = chgpts.DPDU2_regression(np.array(Y_unit), np.array(X_unit), λ_min, ζ_min)
    

    # full_cpt_est = np.block([0, cpt_est_DPDU, n]).astype(int)

    # # Probably better to obtain B̂_DPDU from doing least squares on the estimated changepoint locations, since the output of the above built-in CV is a vector of size p+1 for some reason.
    # B̂_DPDU = np.zeros((p, max(L, len(cpt_est_DPDU) + 1)))
    # for i in range(len(full_cpt_est) - 1):
    #     β̂ = linear_model.Lasso(alpha = λ_min / (2 * np.sqrt(full_cpt_est[i+1] - full_cpt_est[i])), fit_intercept = False).fit(X_unit[full_cpt_est[i]:full_cpt_est[i+1], :], Y_unit[full_cpt_est[i]:full_cpt_est[i+1]]).coef_ # Using the lambda as specified in the paper and in the sklearn.linear_model.Lasso documentation
    #     B̂_DPDU[:, i] = β̂

    # Compute confidence intervals
    # CI_DPDU = []
    # alpha_vec = ro.FloatVector([0.05]) # For the DPDU CI
    # if len(cpt_est_DPDU) > 0:
    #     cpt_LR = chgpts.local_refine_DPDU_regression(cpt_est_DPDU, B̂_DPDU_original, np.array(Y), X, w = 0.9)
    #     try: 
    #         CI_DPDU = chgpts.CI_regression(cpt_est_DPDU, cpt_LR, B̂_DPDU_original, np.array(Y), X, 0.9, 1000, n, alpha_vec)
    #     except:
    #         print("CI regression failed, returning empty list.")
    #     cpt_est_DPDU = cpt_LR

    return DPDU_res_full[-1] # The final index contains the chgpt estimate list

def MOSEG(X_unit, Y_unit, num_chgpts=None, ms=False): 
    """ MOSEG (Cho, Owens 2024). """

    if num_chgpts is not None:
        if not ms:
            # Method 1: Using underscore conversion (recommended)
            # rpy2 automatically converts underscores to dots in parameter names
            out = moseg_package.moseg(X_unit, Y_unit, n_cps=num_chgpts, do_scale=True)
            
        else: 
            # Access the moseg.ms function using getattr
            moseg_ms = getattr(moseg_package, 'moseg.ms')
            
            # Create R vector for the multiscale parameters
            r_vector = ro.IntVector([50, 100, 150])
            
            # Call the function with proper parameter handling
            out = moseg_ms(X_unit, Y_unit, r_vector, do_scale=False)
            
    else:
        # Handle case where num_chgpts is None
        # You might want to use cross-validation or other methods
        if not ms:
            # Example: using default parameters or CV
            moseg_cv = getattr(moseg_package, 'moseg.cv')
            out = moseg_cv(X_unit, Y_unit, penalty=50, do_scale=False)
        else:
            moseg_ms_cv = getattr(moseg_package, 'moseg.ms.cv')
            r_vector = ro.IntVector([50, 100, 150])
            out = moseg_ms_cv(X_unit, Y_unit, r_vector, do_scale=False)

    return out

### --- Logistic Regression Wang, Liu, Zhang, Liu 2023 --- ###

glm_package = importr('myglm') # associated with paper by Wang, Liu, Zhang, Liu 2023

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
    # For some reason, the np.array() wrappers are necessary for this function to run.
    if is_exact:
        print(f"Running bsa with exactly kmax={kmax} chgpts.")
        res = glm_package.bsa_exact_wrapper(np.array(X), np.array(Y), kmax, delta)
    else:
        print(f"Running bsa with at most kmax={kmax} chgpts, at least 1 chgpt.")
        res = glm_package.bsa_wrapper(np.array(X), np.array(Y), kmax, delta)
    # res = glm_package.bsachgpt(X_unit, Y_unit, kmax, c1, m1, delta) # this is a list
    if res == rpy2.rinterface.NULL:
        print("glm_bsa res is NULL.")
        return np.array([])
    num_cp = res[0]
    cp_loc = res[1]
    print("Estimated number of changepoints: ", num_cp)
    print("Estimated changepoint locations: ", cp_loc)
    return cp_loc

### --- Algorithms for sparse difference priors --- %%%

def run_charcoal(X_unit, Y_unit, σ, η_true):
    """
    X_unit is nxp, Y is nx1. AMP δ is n/p.
    σ is AMP's noise level. charcoal uses noise level σ * sqrt(n).
    η_true is the true chgpt locations.
    
    Compared to three methods above, this function returns 1/n*hausdorff 
    distance from η_true directly, but can be modified to return the 
    estimated changepoint locations.
    """
    (n, p) = X_unit.shape
    assert len(Y_unit) == n
    if n < p:
        # GW22 needs n > p, otherwise charcoal crashes
        return np.inf
    charcoal_res = charcoal.not_cpreg(X_unit, Y_unit, 
                    sigma=σ*np.sqrt(n), verbose=True)
    if charcoal_res == rpy2.rinterface.NULL:
        print("charcoal_res is NULL.")
        haus = np.inf
    else: 
        assert len(charcoal_res) == 5
        stage_str_list = ["initial", "test_refine", "midpoint_refine", "final"]
        stage_cp_list = [] # initial, test_refine, midpoint_refine, final estimate of chgpt indices
        stage_hausdorff_list = []
        for i_stage, stage_str in enumerate(stage_str_list):
            print(f"========== {i_stage}/4 {stage_str} ==========")
            if charcoal_res[i_stage+1] == rpy2.rinterface.NULL:
                print("charcoal initial is NULL.")
                stage_hausdorff_list.append(np.inf)
            else:
                cp_stats = np.array(charcoal_res[i_stage+1]) 
                stage_cp_list.append(cp_stats[0, :])
                print(f"cp (0-indexing i.e. Python indexing," + \
                    f" idx of 1st observation of new signal): \ninitial={stage_cp_list[i_stage]},")
                # Compare the estimated and true index of the first observation of the new signal 
                # (with python indexing):
                tmp_haus = 1/n * hausdorff(η_true, stage_cp_list[i_stage])
                # if tmp_haus == np.Inf:
                #     stage_hausdorff_list.append(np.inf)
                # else:
                stage_hausdorff_list.append(tmp_haus)
            print(f"Normalised hausdorff distance: {stage_hausdorff_list[i_stage]}")
        haus = stage_hausdorff_list[-1]
    return haus

def run_mcscan(X_unit, Y_unit, num_chgpts = None):
    """ Runs the McScan procedure from inferchange R package in this paper: https://arxiv.org/abs/2402.06915. 
    X_unit is nxp, Y is nx1.
    If num_chgpts is None, McScan estimates the number of changepoints in addition to location.
    If num_chgpts is an integer, McScan uses that number of changepoints and only estimates location.
    """

    # set.seed(12345)
    # data <- dgp_gauss_sparse(n = 200, p = 20, z = 100, s = 3, rho = 1, sigma = 1)
    # X <- data$X
    # y <- data$y
    # res <- inferchange(X, y)

    if num_chgpts is not None:
        print(f"Running McScan with num_chgpts = {num_chgpts}.")
        res = inferchange_package.McScan(X_unit, Y_unit, num_chgpts)
        cp = res[0]
    else:
        res = inferchange_package.inferchange(X_unit, Y_unit)
        cp = res[0]
        print("Estimated changepoints from McScan: ", cp)
        delta = res[1]
        ci = res[2]
    return cp
