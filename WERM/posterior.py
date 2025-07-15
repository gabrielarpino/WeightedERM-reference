import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jaxopt import ScipyBoundedMinimize
import WERM.config
import WERM.general_fixed_point_eqns as fp
from tqdm.auto import tqdm
from jax import random
import scipy.optimize
import jax.scipy.linalg
import itertools
from scipy.special import comb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from jaxopt import AndersonAcceleration
from sklearn.metrics import mean_squared_error
from jax.scipy.stats import multivariate_normal, norm
from jax import value_and_grad
import matplotlib.pyplot as plt
import jaxopt

# Set small regularization constant for numerical stability
ϵ_0 = 1e-10

### --- Functions for extrapolating posterior computation ---

def fixed_pt_sol_extrapolate(
    n_large, n_small, L, Delta_fn, M_func, ρ, δ, π_small, ε_small, q_func, 
    master_key, batch_size=4096, use_cpu_parallelization=False, cpu_workers=None, coarse_grain_factor_train=1):

    # --- Step 1 & 2: Generate and Structure Training Data ---
    print("### Step 1 & 2: Generating Training Data... ###")
    training_data = generate_training_data(
        n_values=[n_small], L=L, Delta_fn=Delta_fn, M_func=M_func, ρ=ρ, δ=δ, π_small=π_small, ε_small=ε_small, q_func=q_func,
        master_key=master_key, batch_size=batch_size, use_cpu_parallelization=use_cpu_parallelization, cpu_workers=cpu_workers,
        coarse_grain_factor=coarse_grain_factor_train
    )

    # --- Step 3: Train the Interpolator ---
    print("\n### Step 3: Training the Interpolator... ###")
    interpolator = FixedPointInterpolator(L)
    interpolator.train(training_data)

    # --- Step 4: generate grid of chgpts ---
    print("\n### Step 4: Generating Change Point Grid... ###")
    chgpt_grid_absolute = generate_changepoint_grid(
        n_large, L, Delta_fn(n_large)
    )

    if chgpt_grid_absolute:
        # 2. Convert the list of arrays to a single 2D NumPy array
        grid_stacked = np.array(chgpt_grid_absolute)

        # 3. Normalize the grid by dividing by n_large to get fractional locations
        grid_fractional = grid_stacked / n_large

        # 4. Now you can pass this to the predictor
        # predicted_solutions = interpolator.predict(grid_fractional)
    
    # --- Step 5: Predict Fixed Point Solutions ---
    predicted_solutions = interpolator.predict(grid_fractional)
    return predicted_solutions, grid_fractional

class FixedPointInterpolator:
    """
    A wrapper for a Gaussian Process model to learn and predict fixed-point solutions.
    
    This class handles:
    1.  Structuring and scaling the training data.
    2.  Training the Gaussian Process regressor.
    3.  Evaluating and reporting the training fit (MSE and Max Error).
    4.  Predicting new solutions and reshaping them.
    """
    def __init__(self, L, kernel=None):
        """
        Initializes the interpolator.
        
        Args:
            L (int): The dimension of the system (number of signals).
            kernel (sklearn.gaussian_process.kernels.Kernel, optional):
                A GP kernel. Defaults to a robust RBF + WhiteKernel.
        """
        self.L = L
        self.is_trained = False
        
        self.b_size = L
        self.nu_size = L * L
        self.kappa_size = L * L
        
        if kernel is None:
            # --- Kernel Explanation ---
            # The kernel (or covariance function) defines the assumption about the
            # shape of the function we are trying to model.
            # RBF (Radial Basis Function) kernel: This is a universal default choice for
            # smooth functions. It has a `length_scale` parameter that determines
            # how quickly the correlation between points fades with distance. A smaller
            # length_scale allows for more "wiggly" functions.
            # WhiteKernel: This adds a noise term to the model. It assumes that the
            # observations are not perfect but contain some level of random noise.
            # This is crucial for preventing the model from overfitting to the training data.
            # The combination `RBF + WhiteKernel` is a robust and standard choice.
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
                     WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
            
        # n_restarts_optimizer > 0 helps the model find a better set of kernel
        # parameters (like length_scale and noise_level) by running the
        # optimization from multiple starting points.
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            random_state=42
        )
        self.scaler = StandardScaler()

    def _flatten_outputs(self, b, nu, kappa):
        """Flattens the b, nu, and kappa outputs into a single 1D vector."""
        return np.concatenate([b.flatten(), nu.flatten(), kappa.flatten()])

    def _unflatten_outputs(self, y_flat):
        """Reshapes a flattened 1D vector back into b, nu, and kappa."""
        if y_flat.ndim > 1:
             return [self._unflatten_outputs(row) for row in y_flat]
        
        b_start, b_end = 0, self.b_size
        nu_start, nu_end = b_end, b_end + self.nu_size
        kappa_start, kappa_end = nu_end, nu_end + self.kappa_size
        
        b = y_flat[b_start:b_end]
        nu = y_flat[nu_start:nu_end].reshape(self.L, self.L)
        kappa = y_flat[kappa_start:kappa_end].reshape(self.L, self.L)
        return b, nu, kappa

    def train(self, training_data):
        """
        Trains the Gaussian Process model and reports on the quality of the fit.
        """
        L = self.L
        num_change_points = L - 1

        if num_change_points == 0:
            X = np.zeros((len(training_data), 1))
        else:
            X = np.array([d[0] for d in training_data])

        Y = np.array([self._flatten_outputs(d[1], d[2], d[3]) for d in training_data])
        
        Y_scaled = self.scaler.fit_transform(Y)
        
        print(f"Training GP with {X.shape[0]} data points...")
        self.gp.fit(X, Y_scaled)
        self.is_trained = True
        print("--- Training Complete ---")

        # --- Evaluate and Report Fit Quality ---
        # Predict on the training data itself to see how well the model learned it.
        Y_pred_scaled = self.gp.predict(X)
        
        # Calculate Mean Squared Error on the (scaled) training data.
        mse = mean_squared_error(Y_scaled, Y_pred_scaled)
        
        # Calculate the Maximum Absolute Error to find the worst-case fit.
        max_abs_error = np.max(np.abs(Y_scaled - Y_pred_scaled))
        
        print(f"\nFit Quality Report:")
        print(f"  - Training MSE: {mse:.2e}")
        print(f"  - Max Absolute Error: {max_abs_error:.2e}")

        # The `kernel_` attribute holds the kernel with its parameters optimized during fit.
        print(f"  - Optimized Kernel: {self.gp.kernel_}")
        print("-" * 25)


    def predict(self, fractional_etas):
        """
        Predicts the fixed-point solutions for new fractional change point locations.
        """
        if not self.is_trained:
            raise RuntimeError("The interpolator must be trained before calling predict().")

        if fractional_etas.ndim == 1:
            fractional_etas = fractional_etas.reshape(1, -1)
            
        y_pred_scaled = self.gp.predict(fractional_etas)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        
        results = self._unflatten_outputs(y_pred)
        return results

def generate_training_data(n_values, L, Delta_fn, M_func, ρ, δ, π_small, ε_small, q_func, 
                           master_key, batch_size=4096, use_cpu_parallelization=False, cpu_workers=None, coarse_grain_factor=1
                           ):
    """
    Generates training data by running the solver for small n and normalizing results.

    
    """
    all_data = []
    assert len(n_values) == 1, 'Only implemented for n_values of length 0 for now, π input must be extended.'
    print("Generating training data for n values:", n_values)
    for n_small in n_values:
        results = fp.compute_all_fixed_points(n_small, L, Delta_fn(n_small), M_func, 
                        ρ, δ, π_small, ε_small, n_small, q_func, master_key, 
                        batch_size=batch_size, use_cpu_parallelization=use_cpu_parallelization, 
                        cpu_workers=cpu_workers, coarse_grain_factor=coarse_grain_factor
                        )

        for eta_abs, b, nu, kappa, success in results:
            # if success:
            # Normalize eta to get fractional locations
            fractional_eta = eta_abs / n_small
            all_data.append((fractional_eta, b, nu, kappa))
    return all_data

def generate_changepoint_grid(n, L, Delta_n, coarse_grain_factor=1):
    """
    Generates a grid of valid change point configurations.

    A "valid" configuration eta = [eta_1, ..., eta_k] for a series of length `n`
    (with indices 0 to n-1) satisfies three constraints:
    1. eta_1 >= Delta_n (distance from the start)
    2. eta_{i+1} - eta_i >= Delta_n (minimum separation between points)
    3. (n - 1) - eta_k >= Delta_n (distance from the end)

    Args:
        n (int): The total length of the time series.
        L (int): The number of segments, which implies k = L-1 change points.
        Delta_n (int): The minimum distance between change points and from boundaries.
        coarse_grain_factor (int): Factor to thin the grid. 1 returns all valid
                                   configurations, >1 returns a sparser subset.

    Returns:
        list[np.ndarray]: A list of numpy arrays, where each array is a valid
                          change point configuration (in absolute terms, not fractional). Returns an empty list if
                          no configurations are possible.
    """
    num_changepoints = L - 1

    if num_changepoints <= 0:
        # For L=1, there are no change points. The single valid configuration is an empty array.
        # For L<=0, this is ill-defined, so we return an empty list.
        return [np.array([], dtype=int)] if L == 1 else []

    # To solve this constrained combination problem, we map the change points `eta`
    # to a set of unconstrained, strictly increasing integers `x`.
    # The mapping is: x_i = eta_i - (i-1) * (Delta_n - 1) for i=1,...,k.
    # We need to find the range of these `x` values.

    # The first transformed value `x_1` is `eta_1`. Its minimum is Delta_n.
    x_start = Delta_n

    # The last transformed value `x_k` is derived from the max value of `eta_k`.
    # max(eta_k) = n - 1 - Delta_n
    # So, max(x_k) = max(eta_k) - (k-1)*(Delta_n - 1)
    x_end = (n - 1 - Delta_n) - (num_changepoints - 1) * (Delta_n - 1)

    # The size of the set of integers we can choose from is [x_start, x_end].
    effective_set_size = x_end - x_start + 1

    if effective_set_size < num_changepoints:
        # Not enough unique numbers in the transformed space to choose from.
        return []

    # We need to choose `num_changepoints` integers from the `effective_set_size` options.
    # `itertools.combinations` is perfect for this. It generates tuples of indices.
    # The indices are from range(effective_set_size).
    all_combinations_iterator = itertools.combinations(range(effective_set_size), num_changepoints)

    # Apply the coarse grain factor to sample from the iterator without storing it all.
    if coarse_grain_factor > 1:
        # Calculate total number of combinations to provide context if needed.
        # Note: This can be a very large number.
        total_combos = comb(effective_set_size, num_changepoints, exact=True)
        # Use islice to pick every N-th item without realizing the whole list.
        selected_combinations = itertools.islice(all_combinations_iterator, 0, total_combos, coarse_grain_factor)
    else:
        selected_combinations = all_combinations_iterator

    # This array helps in applying the reverse transformation efficiently.
    # reverse_transform[i] = i * (Delta_n - 1)
    reverse_transform = np.arange(num_changepoints) * (Delta_n - 1)

    # Process the selected combinations.
    grid = []
    for combo_indices in selected_combinations:
        # `combo_indices` is a tuple of indices, e.g., (0, 2, 5).
        # Convert these indices to the actual values in the `x` space by adding the offset.
        x_values = x_start + np.array(combo_indices)

        # Transform back from `x` to `eta` values.
        eta = x_values + reverse_transform
        grid.append(eta.astype(int))

    return grid

### --- Functions for solving for ρ ---

def estimate_rho_iterative(X, y, M, pi, L, δ, Delta_n, ε, q, L_is_upper_bound=False, brute_force=False, true_chgpt_locations=None, num_iters = 5, rho_0=None, rho_diag = None, rho_list_output = False):
    """
    Estimate the ρ parameter iteratively
    """
    L = pi.shape[0]
    n, p = X.shape
    ϕ = pi

    # Estimate the change point locations first using weighted ERM
    print("Estimating change point locations using weighted ERM...")
    chgpt_est, gof, B_hat = WERM.estimators.weighted_erm(
        X, y, M, L, Delta_n, 0.0, L_is_upper_bound=L_is_upper_bound,
        brute_force=brute_force, return_gof=True, return_B_hat=True,
    )
    print("Fractional Change point locations estimated:", chgpt_est/n)
    η_ = chgpt_est
    if true_chgpt_locations is not None: # For debugging purposes
        η_ = true_chgpt_locations

    Theta_hat = X @ B_hat

    print("Computing debiased Theta estimate...")
    Theta_deb, b_hat = compute_debiased_theta(Theta_hat, y, M, pi, δ, output_b_hat=True)
    print("Debiased Theta estimate computed.")
    lhs_rho = 1/n * Theta_deb.T @ Theta_deb
    print("Left-hand side for estimating ρ:", lhs_rho)

    if rho_0 is None:
        rho_est = jnp.eye(L)
    else:
        rho_est = rho_0
    rho_L_est = jax.scipy.linalg.cholesky(rho_est + 1e-10 * jnp.eye(L), lower=True)  # Initial guess for the lower triangular part of ρ
    rho_L_flat_est = rho_L_est.flatten()  # Flatten the lower triangular part of ρ
    rho_est_list = []

    for _ in tqdm(range(num_iters)):
        print("Current estimate of ρ:", rho_est)
        b, nu, kappa = fp.general_fp_old(M, rho_est, δ, η_, pi, ε, n, q, est_b = True, X=X, y=y, Delta_n=Delta_n)
        print("b - b_hat:", b - b_hat)
        print("b:", b)
        print("b_hat:", b_hat)
        
        # def obj(rho_L_flat):
        #     """ Objective function to minimize """
        #     rho_L = rho_L_flat.reshape((L, L))
        #     rho = rho_L @ rho_L.T  # Ensure positive semi-definiteness
        #     rhs = nu.T @ rho @ nu + kappa
        #     return (lhs_rho - rhs).flatten()  # Return as a flat array for optimization
        
        # print("Iteration and objective value:", _, obj(rho_L_flat_est)) 
        # # Solve for rho_est
        # # Print the rank of nu and its eigenvalues
        print("nu: ", nu)
        print("Rank of nu:", jnp.linalg.matrix_rank(nu))
        print("Eigenvalues of nu:", jnp.linalg.eigvalsh(nu))
        print("kappa: ", kappa)
        print("Rank of kappa:", jnp.linalg.matrix_rank(kappa))
        print("Eigenvalues of kappa:", jnp.linalg.eigvalsh(kappa))
        # # nu_inv = jnp.linalg.inv(nu + ϵ_0 * jnp.eye(L))  # Add small regularization for numerical stability
        # # rho_est = nu_inv.T @ (lhs_rho - kappa) @ nu_inv

        # # Solve the optimization problem to find rho_est
        # # lower = 1e-6 * jnp.ones((L * (L + 1) // 2,))
        # # upper = jnp.inf * jnp.ones((L * (L + 1) // 2,))
        # # rho_flat_init = jnp.zeros((L * (L + 1) // 2,))
        # # rho_init = rho_0.flatten()
        # rho_L_flat_init = jax.scipy.linalg.cholesky(rho_0 + 1e-10* jnp.eye(L), lower=True).flatten()  # Initial guess for the lower triangular part of ρ
        # # Use scipy.optimize.least_squares to solve the optimization problem
        # res = scipy.optimize.least_squares(
        #     obj,
        #     rho_L_flat_init,
        #     # bounds=(lower, upper),
        #     xtol=1e-6,
        #     ftol=1e-6,
        #     max_nfev=200
        # )
        # rho_L_flat_est = res.x
        # rho_L_est = rho_L_flat_est.reshape((L, L))
        # rho_est = rho_L_est @ rho_L_est.T  # Ensure positive semi-definiteness

        rho_est = optimize_rho_as_cov(lhs_rho, nu, kappa, rho_0, L, rho_diag=rho_diag)
        rho_est_list.append(rho_est)

    if rho_list_output is True: 
        return rho_est_list
    return rho_est

def optimize_rho_as_cov(lhs_rho, nu, kappa, rho_0, L, rho_diag=None, epsilon=1e-10):
    """
    Optimize covariance matrix with option to fix diagonal elements.
    
    Parameters:
    -----------
    lhs_rho : array
        Left-hand side of the equation
    nu : array
        Nu matrix
    kappa : array
        Kappa matrix
    rho_0 : array
        Initial covariance matrix estimate
    L : int
        Dimension of the covariance matrix
    rho_diag : array, optional
        Pre-computed diagonal elements to fix. If None, optimize all elements.
    epsilon : float
        Small regularization constant for numerical stability
    
    Returns:
    --------
    rho_est : array
        Optimized covariance matrix
    """
    
    if rho_diag is not None:
        # Fix diagonal case: parameterize using correlation coefficients
        # We'll parameterize the off-diagonal elements through bounded correlation coefficients
        
        # Number of off-diagonal parameters (strictly lower triangular)
        n_params = L * (L - 1) // 2
        
        def params_to_rho(params):
            """Convert parameter vector to covariance matrix with fixed diagonal"""
            # Create correlation matrix
            corr_matrix = jnp.eye(L)
            
            # Fill in the correlation coefficients (bounded between -1 and 1)
            idx = 0
            for i in range(1, L):
                for j in range(i):
                    # Use tanh to ensure correlation coefficients are in [-1, 1]
                    corr_coef = jnp.tanh(params[idx])
                    corr_matrix = corr_matrix.at[i, j].set(corr_coef)
                    corr_matrix = corr_matrix.at[j, i].set(corr_coef)  # Symmetry
                    idx += 1
            
            # Convert correlation matrix to covariance matrix using fixed diagonal
            D_sqrt = jnp.sqrt(rho_diag + epsilon)
            rho = jnp.outer(D_sqrt, D_sqrt) * corr_matrix
            
            # Ensure the diagonal is exactly as specified (numerical precision)
            rho = rho.at[jnp.diag_indices(L)].set(rho_diag)
            
            return rho
        
        def obj_fixed_diag(params):
            """Objective function with fixed diagonal"""
            rho = params_to_rho(params)
            rhs = nu.T @ rho @ nu + kappa
            return (lhs_rho - rhs).flatten()
        
        # Initialize parameters from initial covariance matrix
        # Convert initial covariance to correlation coefficients
        D_sqrt_init = jnp.sqrt(jnp.diag(rho_0) + epsilon)
        corr_init = rho_0 / jnp.outer(D_sqrt_init, D_sqrt_init)
        
        # Extract correlation coefficients and convert to unbounded parameters
        params_init = []
        for i in range(1, L):
            for j in range(i):
                corr_coef = jnp.clip(corr_init[i, j], -0.99, 0.99)  # Avoid infinities
                params_init.append(jnp.arctanh(corr_coef))
        params_init = jnp.array(params_init)
        
        # Solve optimization
        res = scipy.optimize.least_squares(
            obj_fixed_diag,
            params_init,
            xtol=1e-6,
            ftol=1e-6,
            max_nfev=200
        )
        
        rho_est = params_to_rho(res.x)
        
    else:
        # Original case with bounds on Cholesky factor elements
        # For a valid covariance matrix, we need to ensure proper bounds
        
        def obj(rho_L_flat):
            """Objective function to minimize"""
            rho_L = rho_L_flat.reshape((L, L))
            rho = rho_L @ rho_L.T  # Ensure positive semi-definiteness
            rhs = nu.T @ rho @ nu + kappa
            return (lhs_rho - rhs).flatten()
        
        # Initial guess using Cholesky decomposition
        rho_L_flat_init = jax.scipy.linalg.cholesky(rho_0 + epsilon * jnp.eye(L), lower=True).flatten()
        
        # Set bounds for Cholesky factor elements
        # Diagonal elements must be positive, off-diagonal can be negative
        lower_bounds = []
        upper_bounds = []
        for i in range(L):
            for j in range(L):
                if j > i:
                    # Upper triangular - should be zero
                    # continue
                    lower_bounds.append(-epsilon)
                    upper_bounds.append(epsilon)
                elif i == j:
                    # Diagonal elements must be positive
                    lower_bounds.append(epsilon)
                    upper_bounds.append(jnp.inf)
                else:
                    # Off-diagonal elements can be positive or negative
                    # but bounded to ensure numerical stability
                    max_val = 1000.0 * jnp.sqrt(jnp.max(jnp.diag(rho_0)))
                    lower_bounds.append(-max_val)
                    upper_bounds.append(max_val)
        
        lower_bounds = jnp.array(lower_bounds)
        upper_bounds = jnp.array(upper_bounds)
        
        # Solve optimization with bounds
        res = scipy.optimize.least_squares(
            obj,
            rho_L_flat_init,
            bounds=(lower_bounds, upper_bounds),
            xtol=1e-6,
            ftol=1e-6,
            max_nfev=200
        )
        
        rho_L_flat_est = res.x
        rho_L_est = rho_L_flat_est.reshape((L, L))
        rho_est = rho_L_est @ rho_L_est.T
    
    # Print diagnostics
    print("Optimization converged:", res.success)
    print("Number of function evaluations:", res.nfev)
    print("Final objective norm:", jnp.linalg.norm(res.fun))
    
    # Verify the result is a valid covariance matrix
    eigenvals = jnp.linalg.eigvalsh(rho_est)
    print("Minimum eigenvalue:", jnp.min(eigenvals))
    
    if rho_diag is not None:
        # Check that diagonal is preserved
        print("Max diagonal error:", jnp.max(jnp.abs(jnp.diag(rho_est) - rho_diag)))
        
        # Check correlation bounds
        D_sqrt = jnp.sqrt(jnp.diag(rho_est) + epsilon)
        corr_matrix = rho_est / jnp.outer(D_sqrt, D_sqrt)
        off_diag_mask = ~jnp.eye(L, dtype=bool)
        max_corr = jnp.max(jnp.abs(corr_matrix[off_diag_mask]))
        print("Max absolute correlation:", max_corr)
    
    return rho_est

def estimate_rho_diagonals(X, y, M, L, Delta_n, ε, q, L_is_upper_bound=False, brute_force=False, true_chgpt_locations=None):

    n, p = X.shape
    if true_chgpt_locations is None:
        chgpt_est, gof, B_hat = WERM.estimators.weighted_erm(
            X, y, M, L, Delta_n, 0.0, L_is_upper_bound=L_is_upper_bound,
            brute_force=brute_force, return_gof=True, return_B_hat=True,
        )
    else:
        chgpt_est = true_chgpt_locations

    # Split the data (X, y) into segments according to the estimated changepoints
    if chgpt_est.size > 0:
        X_segments = np.split(X, chgpt_est)
        y_segments = np.split(y, chgpt_est)
        ε_segments = np.split(ε, chgpt_est)
    else:
        X_segments = [X]
        y_segments = [y]
        ε_segments = [ε]

    L_eff = 1
    rho_hat_diag = np.eye(L)
    for (l, data_seg) in enumerate(zip(X_segments, y_segments, ε_segments)):
        X_seg, y_seg, ε_seg = data_seg
        n_eff = X_seg.shape[0]  # Effective n for the segment
        δ_eff = (n_eff/p)  # Effective δ for the segment
        pi_eff = np.ones((1, n_eff)) / n_eff  # Uniform prior for the segment
        rho_hat_l = WERM.posterior.estimate_rho_iterative(X_seg, y_seg, M, pi_eff, L_eff, δ_eff, 1, ε_seg, q, num_iters = 5, 
                                                        rho_0 = 0.1*np.eye(L_eff), L_is_upper_bound=L_is_upper_bound, true_chgpt_locations=np.array([]),
                                                        brute_force=False)
        rho_hat_diag[l, l] = rho_hat_l[0]

    return np.diag(rho_hat_diag)

def estimate_rho(X, y, M, pi, Delta_n, ε, q, L_is_upper_bound=False, brute_force=False, true_chgpt_locations=None, rho_0=None, rho_est_weight=1.0, max_iter=100):
    """
    Estimate the ρ parameter using fixed point equations
    """
    L = pi.shape[0]
    n, p = X.shape
    δ = n / p
    ϕ = pi

    # Estimate the change point locations first using weighted ERM
    print("Estimating change point locations using weighted ERM...")
    chgpt_est, gof, B_hat = WERM.estimators.weighted_erm(
        X, y, M, L, Delta_n, 0.0, L_is_upper_bound=L_is_upper_bound,
        brute_force=brute_force, return_gof=True, return_B_hat=True,
    )
    print("Fractional change point locations estimated:", chgpt_est / n)
    η_abs = chgpt_est
    if true_chgpt_locations is not None: # For debugging purposes
        η_abs = true_chgpt_locations

    Theta_hat = X @ B_hat


    # Define higher order derivatives of M
    global M_prime
    M_prime = jax.grad(M, argnums=0)
    global M_double_prime
    M_double_prime = jax.grad(M_prime, argnums=0)
    # M_partial_12 = jax.grad(M_double_prime, argnums=1)
    M_prime_mapped = jax.vmap(M_prime, in_axes=(0, 0))
    M_double_prime_mapped = jax.vmap(M_double_prime, in_axes=(0, 0))
    # M_partial_12_mapped = jax.vmap(M_partial_12, in_axes=(0, 0))
    prox_mapped = lambda z, b, π_i, v: jax.vmap(fp.prox, in_axes=(0, None, 0, 0, None, None))(z, b, π_i, v, M_prime, M_double_prime)

    # The following are functions used to construct Theta_debiased in obj_4
    M_prime_evaluated = np.zeros((n, L))
    m_prime = np.zeros((n, L))
    for l in range(L):
        M_prime_evaluated[:, l] = M_prime_mapped(Theta_hat[:, l], y)
        m_prime[:, l] = jnp.multiply(M_prime_evaluated[:, l].flatten(), pi[l, :].flatten())

    num_samples = n

    if η_abs.size == 0:
        ψ = jnp.zeros(num_samples).astype(int) 
    else:
        ψ = WERM.config.η_to_ψ((η_abs * num_samples).astype(int), num_samples) # Seems to be correct, converts chgpt to signal config vector

    key = random.PRNGKey(0)
    key1, key_G = random.split(key)

    num_samples_mult = 100 # Number of samples used to estimate a single expectation

    Z_raw_full = random.multivariate_normal(key1, jnp.zeros((L, )), jnp.eye(L), shape=(num_samples_mult, n,)) # num_samp \times n × L
    # Pre-sample standard normal G_raw
    G_raw_full = random.multivariate_normal(key_G, jnp.zeros((L, )), jnp.eye(L), shape=(num_samples_mult, n,)) # num_samp × n × L

    q_mapped = jax.vmap(q, in_axes=(0, 0, 0)) # q_mapped(Z, ψ, ε) = q(Z @ ψ + ε)

    # q_mapped_input_test
    assert q_mapped(Z_raw_full[0, :, :], ψ, ε).shape == (n,), "q_mapped(Z_raw_full[0, :, :], ψ, ε) should be of shape (n,)"
    diag_reg_const = 1e-9

    def W_b(b, ν̃, κ̃, rho, Z_raw, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        Z = Z_raw @ rho_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_prime_mapped( prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)] 
        assert jnp.array(arr).T.shape == (n, L)
        return jnp.array(arr).T

    def W_b_prime(b, ν̃, κ̃, rho, Z_raw, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        Z = Z_raw @ rho_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_double_prime_mapped( prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)]
        assert jnp.array(arr).T.shape == (n, L)
        return jnp.array(arr).T

    def W_z(b, ν̃, κ̃, rho, Z_raw, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L), upper=False)  # stabilize
        G_sample = G_raw @ κ̃_sqrt.T
        Z = Z_raw @ rho_sqrt.T
        # print("g_grad_mapped shape: ", g_grad_mapped(Z, Z @ ν̃[:, 0] + G_sample[:, 0], b[0], ϕ[0, :], ψ, ε, q).shape)
        arr = [fp.g_grad_mapped(Z, Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], ψ, ε, q) for ℓ in range(L)]
        # print("arr.shape = ", jnp.array(arr).shape)
        jnparr = jnp.array(arr)
        jnparr = jnp.moveaxis(jnparr, 0, -1) # move so that: n × dim of gradient \times L
        assert jnparr.shape == (n, L, L)
        return jnparr
    
    # def unpack_x(x):
    #     """ Optimize the cholesky factorization of κ̃ instead of the full matrix. """
    #     b = x[0:L]
    #     ν̃_flat = x[L:L+L**2]
    #     ν̃ = ν̃_flat.reshape((L, L))
    #     L_flat = x[L+L**2:L+L**2 + len(jnp.tril_indices(L)[0])] # Double check whether this is correct
    #     L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
    #     κ̃ = L_mat @ L_mat.T
    #     rho_L_flat = x[L+L**2 + len(jnp.tril_indices(L)[0]):]
    #     rho_L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(rho_L_flat)
    #     rho = rho_L_mat @ rho_L_mat.T  # Ensure positive semi-definiteness
    #     return b, ν̃, κ̃, rho

    def unpack_x(x):
        """ Optimize with ν̃ constrained to be positive definite (non-symmetric). """
        b = x[0:L]
        
        # A is a full L×L matrix
        A_flat = x[L:L+L**2]
        A = A_flat.reshape((L, L))
        
        # ν̃ = A + A^T + ε*I ensures positive definiteness
        ε = 0.0001  # Adjust based on your needs
        ν̃ = A + A.T + ε * jnp.eye(L)
        
        offset = L + L**2
        L_flat = x[offset:offset + len(jnp.tril_indices(L)[0])]
        L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
        κ̃ = L_mat @ L_mat.T
        
        offset += len(jnp.tril_indices(L)[0])
        rho_L_flat = x[offset:]
        rho_L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(rho_L_flat)
        rho = rho_L_mat @ rho_L_mat.T
        
        return b, ν̃, κ̃, rho

    # def unpack_x(x):
    #     """ Optimize with ν̃ constrained to be positive definite (non-symmetric). """
    #     # A is a full L×L matrix
    #     A_flat = x[0:L**2]
    #     A = A_flat.reshape((L, L))
        
    #     # ν̃ = A + A^T + ε*I ensures positive definiteness
    #     ε = 0.0001  # Adjust based on your needs
    #     ν̃ = A + A.T + ε * jnp.eye(L)
        
    #     offset = L**2
    #     L_flat = x[offset:offset + len(jnp.tril_indices(L)[0])]
    #     L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(L_flat)
    #     κ̃ = L_mat @ L_mat.T
        
    #     offset += len(jnp.tril_indices(L)[0])
    #     rho_L_flat = x[offset:]
    #     rho_L_mat = jnp.zeros((L, L)).at[jnp.tril_indices(L)].set(rho_L_flat)
    #     rho = rho_L_mat @ rho_L_mat.T
        
    #     return ν̃, κ̃, rho

    def obj_1(x): 
        b, ν̃, κ̃, rho = unpack_x(x)

        def body(carry, idx):
            Z_raw = Z_raw_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            Wbp = W_b_prime(b, ν̃, κ̃, rho, Z_raw, G_raw)
            term = jnp.pow(Wbp + 1, -1).T @ jnp.ones((n,)) / n
            return carry + term, None
        
        init = jnp.zeros((L,))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = (1 - 1/δ) * jnp.ones((L,))
        assert (lhs - rhs).shape == (L,)
        return lhs - rhs

    def obj_2(x): 
        b, ν̃, κ̃, rho = unpack_x(x)
        rho_sqrt = jnp.linalg.cholesky(rho + 1e-7 * jnp.eye(L), upper=False)  # stabilize

        def body(carry, idx):
            Z_raw = Z_raw_full[idx, :, :]
            Z = Z_raw @ rho_sqrt.T
            G_raw = G_raw_full[idx, :, :]
            term = Z.T @ W_b(b, ν̃, κ̃, rho, Z_raw, G_raw) / n
            return carry + term, None
        
        init = jnp.zeros((L, L))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = jnp.zeros((L, L))
        assert (lhs - rhs).shape == (L, L)
        return lhs - rhs

    def obj_3(x):
        b, ν̃, κ̃, rho = unpack_x(x)

        def body(carry, idx):
            Z_raw = Z_raw_full[idx, :, :]
            G_raw = G_raw_full[idx, :, :]
            Wb_sample = W_b(b, ν̃, κ̃, rho, Z_raw, G_raw)
            term = δ * Wb_sample.T @ Wb_sample / n
            return carry + term, None
        
        init = jnp.zeros((L, L))
        carry_final, _ = jax.lax.scan(body, init, jnp.arange(num_samples_mult))
        rhs = carry_final / num_samples_mult

        lhs = κ̃
        assert (lhs - rhs).shape == (L, L)
        return lhs - rhs

    def obj_4(x): 
        """ Equation for estimating rho. """
        b, nu, kappa, rho = unpack_x(x)

        Theta_deb = jnp.zeros((n, L))
        for l in range(L): 
            theta_deb_l = Theta_hat[:, l] + b[l] * m_prime[:, l]
            Theta_deb = Theta_deb.at[:, l].set(theta_deb_l)
            
        lhs_rho = 1/n * Theta_deb.T @ Theta_deb
        rhs = nu.T @ rho @ nu + kappa
        return lhs_rho - rhs

    def fsolve_obj(x): 
        # return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x).flatten(), rho_est_weight * obj_4(x).flatten()])
        return jnp.concatenate([obj_1(x).flatten(), obj_2(x).flatten(), obj_3(x).flatten(), rho_est_weight * obj_4(x).flatten()])
    fsolve_obj_jit = jax.jit(fsolve_obj)

    # Solve using fsolve, faster than minimize with the jacobian
    J_fsolve = jax.jacrev(fsolve_obj_jit, argnums=0)
    L0 = 0.75 * jnp.eye(L)
    if rho_0 is None: 
        rho_0 = jnp.eye(L)  # Initial guess for the covariance matrix ρ
    L0_flat = L0[jnp.tril_indices(L)]
    rho_0_flat = rho_0[jnp.tril_indices(L)]

    x0 = jnp.concatenate([
        0.66 * jnp.ones((L,)),
        0.85 * jnp.eye(L).flatten(),
        L0_flat, 
        rho_0_flat  # Initial guess for the lower triangular part of ρ
    ])
    # x0 = jnp.concatenate([
    #     0.66 * jnp.ones((L,)),
    #     0.85 * jnp.ones((L,L)).flatten(),
    #     0.75 * jnp.eye(L).flatten()
    # ])

    # Use constrained optimization instead of fsolve, seems to be more stable since enforcing b and diagonal of κ̃ to be positive
    lower_b = 1e-7 * jnp.ones((L,))
    lower_nu = -jnp.inf * jnp.ones((L * L,))
    L_indices = jnp.tril_indices(L)

    # Set bounds for each element in L
    lower_L = -jnp.inf * jnp.ones(len(L_indices[0]))
    for i in range(L):  # enforce positivity on L_ii
        diag_idx = jnp.where((L_indices[0] == i) & (L_indices[1] == i))[0][0]
        lower_L = lower_L.at[diag_idx].set(1e-6)

    lower = jnp.concatenate([lower_b, lower_nu, lower_L, lower_L])
    # lower = jnp.concatenate([lower_nu, lower_L, lower_L])
    
    upper_b = jnp.inf * jnp.ones((L,))
    upper_nu = jnp.inf * jnp.ones((L * L,))
    upper_L = jnp.inf * jnp.ones(len(L_indices[0]))
    upper = jnp.concatenate([upper_b, upper_nu, upper_L, upper_L])
    # upper = jnp.concatenate([upper_nu, upper_L, upper_L])

    # Replace the scipy.optimize.least_squares call with:
    solver = jaxopt.LevenbergMarquardt(
        residual_fun=fsolve_obj_jit,
        maxiter=max_iter,
        tol=1e-3,
        verbose=True
    )
    res = solver.run(x0)
    # print("Optimization successful:", res.state.success)

    b, ν̃, κ̃, rho = unpack_x(res.params)

    return b, ν̃, κ̃, rho

def estimate_rho_stochastic(
    X, y, M, pi, Delta_n, ε, q, L_is_upper_bound=False, brute_force=False,
    true_chgpt_locations=None, rho_0=None, rho_est_weight=1.0,
    # Hyperparameters for the stochastic optimizer
    num_steps=1000,
    batch_size=32,
    initial_lr=0.01,
    decay_rate=0.01,
    huber_delta=1.0,  # Threshold for Huber loss
    true_rho=None,
):
    """
    Estimate the ρ parameter using a robust stochastic optimization loop with Huber loss.
    ν̃ (nu) is an unconstrained L x L matrix.
    """
    L = pi.shape[0]
    n, p = X.shape
    δ = n / p
    ϕ = pi

    # --- Step 1: Initial setup (same as before) ---
    print("Estimating change point locations using weighted ERM...")
    chgpt_est, gof, B_hat = WERM.estimators.weighted_erm(
        X, y, M, L, Delta_n, 0.0, L_is_upper_bound=L_is_upper_bound,
        brute_force=brute_force, return_gof=True, return_B_hat=True,
    )
    print("Change point locations estimated:", chgpt_est)
    η_abs = chgpt_est
    if true_chgpt_locations is not None:
        η_abs = true_chgpt_locations
    Theta_hat = X @ B_hat

    # Define higher order derivatives of M
    global M_prime, M_double_prime
    M_prime = jax.grad(M, argnums=0)
    M_double_prime = jax.grad(M_prime, argnums=0)
    M_prime_mapped = jax.vmap(M_prime, in_axes=(0, 0))
    M_double_prime_mapped = jax.vmap(M_double_prime, in_axes=(0, 0))
    prox_mapped = lambda z, b, π_i, v: jax.vmap(fp.prox, in_axes=(0, None, 0, 0, None, None))(z, b, π_i, v, M_prime, M_double_prime)

    m_prime = jnp.zeros((n, L))
    for l in range(L):
        m_prime = m_prime.at[:, l].set(jnp.multiply(M_prime_mapped(Theta_hat[:, l], y).flatten(), pi[l, :].flatten()))

    num_samples = n
    ψ = WERM.config.η_to_ψ((η_abs * num_samples).astype(int), num_samples) if η_abs.size > 0 else jnp.zeros(num_samples).astype(int)

    # Pre-sample a large pool of standard normal variables
    key = random.PRNGKey(0)
    key1, key_G = random.split(key)
    num_pool_samples = 2048
    Z_raw_full = random.multivariate_normal(key1, jnp.zeros(L), jnp.eye(L), shape=(num_pool_samples, n))
    G_raw_full = random.multivariate_normal(key_G, jnp.zeros(L), jnp.eye(L), shape=(num_pool_samples, n))

    q_mapped = jax.vmap(q, in_axes=(0, 0, 0))
    diag_reg_const = 1e-9

    # --- Step 2: Define parameter packing and objective functions ---

    def unpack_x_stochastic(x):
        log_b = x[0:L]
        b = jnp.exp(log_b)
        
        nu_flat = x[L:L+L**2]
        ν̃ = nu_flat.reshape((L, L))

        offset = L + L**2
        tril_indices = jnp.tril_indices(L)
        
        L_flat_kappa = x[offset : offset + len(tril_indices[0])]
        L_mat_kappa = jnp.zeros((L, L)).at[tril_indices].set(L_flat_kappa)
        κ̃ = L_mat_kappa @ L_mat_kappa.T
        
        offset += len(tril_indices[0])
        L_flat_rho = x[offset:]
        L_mat_rho = jnp.zeros((L, L)).at[tril_indices].set(L_flat_rho)
        rho = L_mat_rho @ L_mat_rho.T
        
        return b, ν̃, κ̃, rho
    
    def W_b(b, ν̃, κ̃, rho, Z_raw, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + diag_reg_const * jnp.eye(L))
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        Z = Z_raw @ rho_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_prime_mapped(prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)]
        return jnp.array(arr).T

    def W_b_prime(b, ν̃, κ̃, rho, Z_raw, G_raw):
        κ̃_sqrt = jnp.linalg.cholesky(κ̃ + diag_reg_const * jnp.eye(L))
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L))
        G_sample = G_raw @ κ̃_sqrt.T
        Z = Z_raw @ rho_sqrt.T
        q_mapped_with_input = q_mapped(Z, ψ, ε)
        arr = [b[ℓ] * jnp.multiply(ϕ[ℓ, :], M_double_prime_mapped(prox_mapped(Z @ ν̃[:, ℓ] + G_sample[:, ℓ], b[ℓ], ϕ[ℓ, :], q_mapped_with_input), q_mapped_with_input)) for ℓ in range(L)]
        return jnp.array(arr).T
        
    def obj_1_stochastic(x, Z_raw_batch, G_raw_batch):
        b, ν̃, κ̃, rho = unpack_x_stochastic(x)
        Wbp_batch = jax.vmap(W_b_prime, in_axes=(None, None, None, None, 0, 0))(b, ν̃, κ̃, rho, Z_raw_batch, G_raw_batch)
        # rhs = jnp.mean(jnp.power(Wbp_batch + 1, -1).T @ jnp.ones((n,)) / n, axis=1)
        # With this corrected line:
        rhs = jnp.mean(jnp.einsum('bni,n->bi', jnp.power(Wbp_batch + 1, -1), jnp.ones(n)) / n, axis=0)
        lhs = (1 - 1/δ) * jnp.ones((L,))
        return lhs - rhs

    def obj_2_stochastic(x, Z_raw_batch, G_raw_batch):
        b, ν̃, κ̃, rho = unpack_x_stochastic(x)
        rho_sqrt = jnp.linalg.cholesky(rho + diag_reg_const * jnp.eye(L))
        
        def single_sample_calc(Z_raw, G_raw):
            Z = Z_raw @ rho_sqrt.T
            Wb_sample = W_b(b, ν̃, κ̃, rho, Z_raw, G_raw)
            return Z.T @ Wb_sample / n
        
        rhs = jnp.mean(jax.vmap(single_sample_calc)(Z_raw_batch, G_raw_batch), axis=0)
        lhs = jnp.zeros((L, L))
        return lhs - rhs

    def obj_3_stochastic(x, Z_raw_batch, G_raw_batch):
        b, ν̃, κ̃, rho = unpack_x_stochastic(x)
        Wb_batch = jax.vmap(W_b, in_axes=(None, None, None, None, 0, 0))(b, ν̃, κ̃, rho, Z_raw_batch, G_raw_batch)
        rhs = δ * jnp.mean(jax.vmap(lambda Wb: Wb.T @ Wb / n)(Wb_batch), axis=0)
        lhs = κ̃
        return lhs - rhs
        
    def obj_4(x):
        b, ν̃, κ̃, rho = unpack_x_stochastic(x)
        Theta_deb = Theta_hat + b * m_prime
        lhs_rho = 1/n * Theta_deb.T @ Theta_deb
        rhs = ν̃.T @ rho @ ν̃ + κ̃
        return lhs_rho - rhs
    
    def fsolve_obj_stochastic(x, Z_raw_batch, G_raw_batch):
        return jnp.concatenate([
            obj_1_stochastic(x, Z_raw_batch, G_raw_batch).flatten(),
            obj_2_stochastic(x, Z_raw_batch, G_raw_batch).flatten(),
            obj_3_stochastic(x, Z_raw_batch, G_raw_batch).flatten(),
            rho_est_weight * obj_4(x).flatten()
        ])

    # --- Step 3: Define Robust Loss and Gradient for SGD ---
    
    def huber_loss(residual, delta):
        """ A robust, element-wise Huber loss function. """
        abs_residual = jnp.abs(residual)
        return jnp.where(
            abs_residual <= delta,
            0.5 * jnp.square(residual),  # Squared loss for small errors
            delta * (abs_residual - 0.5 * delta)  # Absolute loss for large errors
        )

    def loss_fn(x, Z_raw_batch, G_raw_batch):
        """ Calculates the total Huber loss over a batch of residuals. """
        residuals = fsolve_obj_stochastic(x, Z_raw_batch, G_raw_batch)
        # Apply Huber loss to each residual element-wise and sum the result
        return jnp.sum(jax.vmap(huber_loss, in_axes=(0, None))(residuals, huber_delta))

    @jax.jit
    def update_step(x, key, step):
        """ Performs a single update step of the stochastic optimizer. """
        batch_indices = random.choice(key, num_pool_samples, shape=(batch_size,), replace=False)
        Z_raw_batch = Z_raw_full[batch_indices]
        G_raw_batch = G_raw_full[batch_indices]
        
        # JAX automatically computes the correct gradient for the Huber loss
        grad = jax.grad(loss_fn, argnums=0)(x, Z_raw_batch, G_raw_batch)
        
        lr = initial_lr / (1.0 + step * decay_rate)
        x_new = x - lr * grad
        return x_new

    # --- Step 4: Run the Stochastic Optimization Loop ---

    L0 = 0.75 * jnp.eye(L)
    rho_0 = rho_0 if rho_0 is not None else jnp.eye(L)
    L0_flat = L0[jnp.tril_indices(L)]
    rho_0_flat = rho_0[jnp.tril_indices(L)]

    x0 = jnp.concatenate([
        jnp.log(0.66 * jnp.ones((L,))),
        0.85 * jnp.eye(L).flatten(),
        L0_flat,
        rho_0_flat
    ])
    
    x = x0
    opt_key = random.PRNGKey(42)
    print(f"Starting stochastic optimization for {num_steps} steps with Huber loss (delta={huber_delta})...")
    for step in range(num_steps):
        opt_key, step_key = random.split(opt_key)
        x = update_step(x, step_key, step)
        if step % 100 == 0:
            # Monitor convergence by logging the loss on a consistent batch
            current_loss = loss_fn(x, G_raw_full[:batch_size], Z_raw_full[:batch_size])
            print(f"Step {step}, Current Loss: {current_loss:.4f}")
            if true_rho is not None:
                _, _, _, rho_est = unpack_x_stochastic(x)
                frob_error = jnp.linalg.norm(rho_est - true_rho, ord='fro')
                print(f" Difference between estimated and true rho:", true_rho - rho_est)
                print(f" Frobenius norm error in rho estimate: {frob_error:.4f}")

    print("Optimization finished.")
    b, ν̃, κ̃, rho = unpack_x_stochastic(x)
    return b, ν̃, κ̃, rho

### --- Functions for computing debiased theta estimate ---

def compute_debiased_theta(Theta_hat, y, M, pi, delta, output_b_hat=False, history_size=5, ridge=1e-8, maxiter=1000, learning_rate=0.1, use_iterate_averaging=True):
    """
    Compute the debiased estimate of Theta.

    Theta_hat: array of shape (n, L)
    """

    L = Theta_hat.shape[1]
    b_hat_arr = jnp.zeros((L,))
    Theta_deb = jnp.zeros_like(Theta_hat)
    M_prime = jax.grad(M, argnums=0)
    M_prime_mapped = jax.vmap(M_prime, in_axes=(0, 0))
    for l in range(L):
        theta_hat = Theta_hat[:, l]
        # b_hat_l = compute_b_hat(theta_hat, y, M, pi[l, :], delta)
        b_hat_l = compute_b_hat_anderson_jax(theta_hat, y, M, pi[l, :], delta, history_size=history_size, ridge=ridge, maxiter=maxiter, learning_rate=learning_rate, use_iterate_averaging=use_iterate_averaging)
        b_hat_arr = b_hat_arr.at[l].set(b_hat_l)

        M_prime_evaluated = M_prime_mapped(theta_hat, y)
        assert pi[l, :].shape == M_prime_evaluated.shape == (Theta_hat.shape[0],)
        m_prime = jnp.multiply(M_prime_evaluated.flatten(), pi[l, :].flatten())
        assert m_prime.shape == (Theta_hat.shape[0],)
        theta_deb = theta_hat + b_hat_l * m_prime
        Theta_deb = Theta_deb.at[:, l].set(theta_deb)

    if output_b_hat:
        return Theta_deb, b_hat_arr
    return Theta_deb

def compute_b_hat_anderson_jax(
    theta_hat,
    y,
    M,
    sample_weight,
    δ,
    tol=1e-6,
    maxiter=1000,
    learning_rate=0.1,
    ridge=1e-8,
    history_size=5,
    use_iterate_averaging=True
):
    """
    Solves the empirical fixed point equation for b using robust techniques
    for stochastic/noisy objectives.

    Args:
        ... (standard args) ...
        learning_rate (float): Damping factor for the update step to improve stability.
        ridge (float): Regularization parameter for the Anderson solver to stabilize
                       it against noise.
        history_size (int): The history size for Anderson acceleration.
        use_iterate_averaging (bool): If True, returns the average of all iterates,
                                     which is more stable under noise.

    Returns:
        jnp.ndarray: The estimated scalar b.
    """
    # Define the second derivative of M, vmapped over the data
    M_prime = jax.grad(M, argnums=0)
    M_double_prime_mapped = jax.vmap(jax.grad(M_prime, argnums=0), in_axes=(0, 0))

    def fixed_point_equation(b):
        """The residual of the fixed point equation."""
        lhs = 1 - 1/δ
        rhs = jnp.mean(jnp.pow(1 + b * jnp.multiply(sample_weight, M_double_prime_mapped(theta_hat, y)), -1))
        return lhs - rhs

    def fixed_point_update(b):
        """The damped and projected fixed-point iteration."""
        residual = fixed_point_equation(b)
        b_next = b - learning_rate * residual
        return jnp.maximum(1e-10, b_next)

    # Set up the AndersonAcceleration solver with regularization
    solver = AndersonAcceleration(
        fixed_point_fun=fixed_point_update,
        history_size=history_size,
        ridge=ridge,
        tol=tol,
        maxiter=maxiter,
    )

    # CORRECTED: The body function must accept the iteration index `i` and the `state` tuple.
    def loop_body(i, state):
        b, b_avg, solver_state = state
        # Perform one update step
        b_next, solver_state_next = solver.update(params=b, state=solver_state)
        # Update the running average using the current iteration index `i`
        b_avg_next = (i * b_avg + b_next) / (i + 1)
        return b_next, b_avg_next, solver_state_next

    # CORRECTED: The initial state no longer needs to contain the counter `k`.
    b_init = jnp.array(0.1)
    initial_solver_state = solver.init_state(b_init)
    initial_loop_state = (b_init, b_init, initial_solver_state)
    
    # Run the solver for a fixed number of iterations
    b_final, b_avg_final, _ = jax.lax.fori_loop(0, maxiter, loop_body, initial_loop_state)

    # Return either the averaged solution or the final iterate
    return jax.lax.cond(
        use_iterate_averaging,
        lambda: b_avg_final,
        lambda: b_final
    )

### --- Functions for computing Gaussian posterior (old) ---

def exact_posterior_gaussian(η_arr, p_η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated):

    a = logP_V_Y_given_η_jax_allη(η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated).reshape(-1, 1)
    a_max = jnp.max(a)
    b = p_η_arr.reshape(-1, 1)
    log_num, sgn_log_num = logsumexp(a=a-a_max, axis=1, b=b, return_sign=True)
    log_denom, sgn_log_denom = logsumexp(a=a-a_max, axis=0, b=b, return_sign=True)
    posterior = jnp.exp(sgn_log_num*log_num - sgn_log_denom*log_denom) # length num_configs vector 
    assert jnp.isclose(jnp.sum(posterior), 1)
    return posterior

def logP_V_Y_given_η_jax(η, V, Y, ρ, σ, ν, κ_T):
    n = Y.shape[0]
    # Use the new, improved function
    ψ = η_to_ψ_jax_new(η, n) # length-n vector
    return jnp.sum(logP_Vj_Yj_given_ψj_allj(V, Y, ψ.reshape(-1,1), ρ, σ, ν, κ_T))

def logP_V_Y_given_η_jax_allη(η_arr, V, Y, ρ, σ, ν_list, κ_T_list):
    return jax.vmap(logP_V_Y_given_η_jax, (0, None, None, None, None, 0, 0), 0)(η_arr, V, Y, ρ, σ, ν_list, κ_T_list)

### ---- Functions for the (approximate) posterior --- ###

def logP_Vj_Yj_given_ψj(Vj, Yj, ψj, ρ, σ, ν, κ_T):
    """
    Calculates log P(Vj, Yj | ψj).
    
    Vj ∈ R^L is the j-th row of V.
    Yj ∈ R is the j-th entry of q(Z, ψ, ϵ).
    ψj ∈ R is the j-th entry of ψ.
    """
    L = ρ.shape[0]
    Σj = Σ_V_j_Y_j(ψj, ρ, σ, ν, κ_T) # (L+1)x(L+1) matrix
    V_flat = Vj.flatten()
    x = jnp.block([
        [V_flat.reshape((L, 1))], 
        [Yj]
    ]).flatten() # length-(L+1) vector
    return multivariate_normal.logpdf(x, mean=jnp.zeros(((L+1),)), 
        cov=Σj + ϵ_0 * jnp.eye((L+1))) # scalar
logP_Vj_Yj_given_ψj_allj = jax.jit(jax.vmap(logP_Vj_Yj_given_ψj, (0, 0, 0, 
    None, None, None, None), 0)) # length-n vector

def Σ_V_j_Y_j(C_j, ρ, σ, ν, κ_T):
    """Covariance of V_j ∈ R^L, Y_j ∈ R given C_j"""
    L = ρ.shape[0]
    ν_ρ_inv = ν.T @ jnp.linalg.inv(ρ)
    A = jnp.block([
        [ν_ρ_inv @ ν + κ_T, (ν_ρ_inv @ ρ[:, C_j]).reshape((L, 1)) ],
        [(ν_ρ_inv @ ρ[:, C_j]).reshape((1, L)), ρ[C_j, C_j] + σ**2]
    ])
    return A # (L+1)x(L+1) matrix

def η_to_ψ_jax_new(η, n):
    """
    Converts a change point vector (η) to a signal configuration vector (ψ)
    using jnp.searchsorted.

    Args:
        η: A 1D array of change point start indices, sorted in ascending order.
           -1 is used as a placeholder for non-existent change points.
        n: The total number of data points.

    Returns:
        A length-n vector (ψ) where ψ[i] is the signal index for data point i.
    """
    # Create an array of indices [0, 1, ..., n-1]
    indices = jnp.arange(n)

    # Replace -1 placeholders with a value larger than any index (n),
    # so they don't affect the search.
    # The `where` clause handles the case where η might be padded with -1s.
    valid_η = jnp.where(η == -1, n, η)

    # Find the insertion point for each index. This directly gives the signal segment.
    # 'right' means that if an index `i` is equal to a changepoint, it belongs to the new segment.
    return jnp.searchsorted(valid_η, indices, side='right')

### --- Logistic posterior ---

def exact_posterior_logistic(η_arr, p_η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated, normalize=True):

    a = logP_V_Y_given_η_jax_allη_logistic(η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated).reshape(-1, 1)
    a_max = jnp.max(a)
    b = p_η_arr.reshape(-1, 1)
    log_num, sgn_log_num = logsumexp(a=a-a_max, axis=1, b=b, return_sign=True)
    log_denom, sgn_log_denom = logsumexp(a=a-a_max, axis=0, b=b, return_sign=True)
    if normalize:
        posterior = jnp.exp(sgn_log_num*log_num - sgn_log_denom*log_denom) # length num_configs vector 
        assert jnp.isclose(jnp.sum(posterior), 1)
    else:
        posterior = jnp.exp(sgn_log_num*log_num)
    return posterior

def logP_V_Y_given_η_jax_logistic(η, V, Y, ρ, σ, ν, κ_T):
    """ TODO: This uses η_to_ψ_jax, which assumes η has to -1's in it. Need to change η_to_ψ_jax to fix this. """
    n = Y.shape[0]
    ψ = η_to_ψ_jax_new(η, n) # length-n vector
    # log P(V, Y | η) = log ∑_j P(Vj, Yj | ψj)
    # sum over all rows of V, Y, ψ:
    return jnp.sum(logP_Vj_Yj_given_ψj_allj_logistic(V, Y, ψ.reshape(-1,1), ρ, σ, ν, κ_T)) # scalar

def logP_V_Y_given_η_jax_allη_logistic(η_arr, V, Y, ρ, σ, ν_list, κ_T_list):
    return jax.vmap(logP_V_Y_given_η_jax_logistic, (0, None, None, None, None, 0, 0), 0)(η_arr, V, Y, ρ, σ, ν_list, κ_T_list)

def logP_Vj_Yj_given_ψj_allj_logistic(V, Y, ψ, ρ, σ, ν, κ_T):

    L = V.shape[1]
    n = V.shape[0]
    def helper(x):
        V = x[:L]
        Y = x[L]
        ψ = x[L+1].astype(int)
        return logP_Vj_Yj_given_ψj_logistic(V, Y, ψ, ρ, σ, ν, κ_T)

    x = jnp.column_stack((V, Y, ψ))
    return jax.lax.map(helper, x)
    
def logP_Vj_Yj_given_ψj_logistic(Vj, Yj, ψj, ρ, σ, ν, κ_T):
    """
    Calculates log P(Vj, Yj | ψj).
    
    Vj ∈ R^L is the j-th row of V.
    Yj ∈ R is the j-th entry of q(Z, ψ, ϵ).
    ψj ∈ R is the j-th entry of ψ.
    """
    L = ρ.shape[0]
    lam = np.sqrt(np.pi/8)

    Σ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T     # covariance matrix of V
    Σ_Z_given_V = (ρ - ν.T @ jnp.linalg.inv(Σ) @ ν)
    μ_s = (ν @ jnp.linalg.inv(Σ) @ (Vj.reshape(L,1).flatten()))[ψj]

    σ_sq_s = Σ_Z_given_V.diagonal().flatten()[ψj]

    ρ_s = lam/jnp.sqrt(1+lam**2*σ_sq_s)

    res_0 = norm.logcdf(-ρ_s*μ_s, loc=0, scale=1)
    res_1 = norm.logcdf(ρ_s*μ_s, loc=0, scale=1)

    res =  (Yj==0).astype(int)*res_0 + (Yj==1).astype(int)*res_1
    return res # scalar