import itertools
import jax.numpy as jnp
import jax
import numpy as np
from scipy.special import comb
from tqdm.auto import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.interpolate import splrep, splev, UnivariateSpline, interp1d
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

### --- Compute the marginals by extrapolating from smaller solution --- ###

def extrapolated_marginals(Lmin, Lmax, Delta_small, n_small=50, n_large=1000, p_l=None, verbose='summary'):
    """ 
    Extrapolate the marginal probabilities from a small problem size to a larger one.
    
    Parameters:
    -----------
    verbose : str or bool
        - True or 'full': Print all progress and diagnostic information
        - 'summary': Print only final fitting summary (average MSE and max normalization error)
        - False or 'silent': Print nothing
    """
    
    # Compute the exact marginals for the small problem
    # p_l = 1/L * np.ones(L) # puts equal weight on all number of signals, set None otherwise. Best for estimating size. 
    # p_l = None
    num_valid_configs, η_arr, p_η_arr, ϕ = unif_prior_to_η_ϕ(Lmin, Lmax, Delta_small, n_small, p_l=p_l) # Lxn matrix
    assert η_arr.shape == (num_valid_configs, Lmax-1) and \
        len(p_η_arr) == num_valid_configs
    assert ϕ.shape == (Lmax, n_small)
    π_original = ϕ

    # Extrapolate to larger problem size
    # Step 2: Fit exact models
    fitter = ExactCurveFitter(L=Lmax, target_mse=1e-8, verbose=verbose)
    fitter.fit_from_small_problem(π_original.shape[1], π_original, method='auto')

    # Step 3: Generate exact curves for large n
    π = fitter.predict_for_size(n_large=n_large)
    return π

### --- Helper functions for extrapolation --- ###

class ExactCurveFitter:
    """
    High-capacity curve fitting for EXACT reconstruction of marginal probability curves.
    
    Implements multiple state-of-the-art flexible models:
    1. Gaussian Process Regression with optimized kernels
    2. Deep Neural Networks (PyTorch)
    3. B-spline interpolation with automatic knot selection
    4. Radial Basis Function Networks
    5. Ensemble methods combining multiple approaches
    """
    
    def __init__(self, L, delta_ratio=0.1, target_mse=1e-8, verbose='summary'):
        """
        Parameters:
        -----------
        L : int
            Number of segments
        delta_ratio : float
            Ratio Δ/n
        target_mse : float
            Target MSE for "exact" fitting (default: 1e-8)
        verbose : str or bool
            - True or 'full': Print all progress and diagnostic information
            - 'summary': Print only final fitting summary (average MSE and max normalization error)
            - False or 'silent': Print nothing
        """
        self.L = L
        self.delta_ratio = delta_ratio
        self.target_mse = target_mse
        self.verbose = verbose
        self.fitted_models = {}
        self.model_types = {}
        self.scalers = {}
        self.fit_scores = {}
        
    def _print(self, *args, **kwargs):
        """Print only if verbose is True or 'full'."""
        if self.verbose is True or self.verbose == 'full':
            print(*args, **kwargs)
    
    def _print_summary(self, *args, **kwargs):
        """Print if verbose is True, 'full', or 'summary'."""
        if self.verbose is True or self.verbose == 'full' or self.verbose == 'summary':
            print(*args, **kwargs)
        
    def fit_from_small_problem(self, n_small, phi_matrix, method='auto'):
        """
        Fit high-capacity models to achieve near-exact reconstruction.
        
        Parameters:
        -----------
        n_small : int
            Size of small problem
        phi_matrix : ndarray of shape (L, n_small)
            Original marginal probabilities
        method : str
            'auto', 'gp', 'neural', 'spline', 'rbf', 'ensemble'
        """
        self._print(f"Fitting exact models to curves from n={n_small} problem...")
        self._print(f"Target MSE: {self.target_mse}")
        
        # Normalize positions to [0, 1]
        positions_norm = np.linspace(0, 1, n_small)
        
        # Fit each segment with the best possible model
        for ell in range(self.L):
            probs = phi_matrix[ell, :]
            
            self._print(f"\nFitting segment {ell}...")
            
            if method == 'auto':
                # Try all methods and pick the best
                best_model, best_type, best_score = self._fit_best_exact_model(positions_norm, probs, ell)
            else:
                # Use specified method
                best_model, best_score = self._fit_single_method(positions_norm, probs, method, ell)
                best_type = method
            
            self.fitted_models[ell] = best_model
            self.model_types[ell] = best_type
            self.fit_scores[ell] = best_score
            
            self._print(f"  Best model: {best_type} (MSE = {best_score:.2e})")
            
            if best_score > self.target_mse:
                self._print(f"  Warning: MSE {best_score:.2e} exceeds target {self.target_mse:.2e}")
        
        self.n_reference = n_small
        
        # Summary output for fitting quality
        avg_mse = np.mean(list(self.fit_scores.values()))
        self._print_summary(f"Fitting complete! Average MSE: {avg_mse:.2e} (target: {self.target_mse:.2e})")
        
        # Detailed output only for full verbose
        self._print(f"\nExact fitting complete! Average MSE: {avg_mse:.2e}")
    
    def _fit_best_exact_model(self, x, y, segment_idx):
        """
        Try all high-capacity methods and return the best.
        """
        methods_to_try = [
            ('gp_rbf', lambda: self._fit_gaussian_process(x, y, kernel_type='rbf')),
            ('gp_matern', lambda: self._fit_gaussian_process(x, y, kernel_type='matern')),
            ('neural_deep', lambda: self._fit_neural_network(x, y, architecture='deep')),
            ('neural_wide', lambda: self._fit_neural_network(x, y, architecture='wide')),
            ('spline_smooth', lambda: self._fit_spline(x, y, smoothing=True)),
            ('spline_exact', lambda: self._fit_spline(x, y, smoothing=False)),
            ('rbf_network', lambda: self._fit_rbf_network(x, y)),
        ]
        
        best_model = None
        best_type = None
        best_score = float('inf')
        
        for method_name, fit_func in methods_to_try:
            try:
                model, score = fit_func()
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_type = method_name
                    
                    # If we hit target accuracy, stop early
                    if score < self.target_mse:
                        self._print(f"    {method_name}: MSE = {score:.2e} ✓ (target reached)")
                        break
                    else:
                        self._print(f"    {method_name}: MSE = {score:.2e}")
                        
            except Exception as e:
                self._print(f"    {method_name}: Failed ({str(e)[:50]})")
                continue
        
        return best_model, best_type, best_score
    
    def _fit_gaussian_process(self, x, y, kernel_type='rbf'):
        """
        Fit Gaussian Process with optimized hyperparameters.
        """
        # Import here to avoid issues if sklearn is not available
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
        except ImportError:
            raise ImportError("sklearn required for Gaussian Process fitting")
            
        if kernel_type == 'rbf':
            # RBF kernel with multiple length scales to try
            kernels = [
                RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-10),
                RBF(length_scale=0.05) + WhiteKernel(noise_level=1e-10),
                RBF(length_scale=0.2) + WhiteKernel(noise_level=1e-10),
            ]
        else:  # matern
            kernels = [
                Matern(length_scale=0.1, nu=2.5) + WhiteKernel(noise_level=1e-10),
                Matern(length_scale=0.05, nu=1.5) + WhiteKernel(noise_level=1e-10),
                Matern(length_scale=0.2, nu=0.5) + WhiteKernel(noise_level=1e-10),
            ]
        
        best_gp = None
        best_score = float('inf')
        
        for kernel in kernels:
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-10,  # Very low noise
                    optimizer='fmin_l_bfgs_b',
                    n_restarts_optimizer=10,
                    normalize_y=True
                )
                
                gp.fit(x.reshape(-1, 1), y)
                y_pred = gp.predict(x.reshape(-1, 1))
                mse = np.mean((y - y_pred)**2)
                
                if mse < best_score:
                    best_score = mse
                    best_gp = gp
                    
            except Exception as e:
                continue
        
        if best_gp is None:
            raise ValueError("All GP kernels failed")
        
        def predict_func(x_new):
            pred = best_gp.predict(x_new.reshape(-1, 1))
            return np.clip(pred, 0.0, 1.0)
        
        return predict_func, best_score
    
    def _fit_neural_network(self, x, y, architecture='deep'):
        """
        Fit deep neural network with PyTorch for maximum flexibility.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required for neural network fitting")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(device)
        
        # Define network architecture
        if architecture == 'deep':
            # Deep narrow network
            hidden_sizes = [64, 64, 64, 32, 16]
        else:  # wide
            # Wide shallow network
            hidden_sizes = [256, 128, 64]
        
        class FlexibleNet(nn.Module):
            def __init__(self, hidden_sizes):
                super().__init__()
                layers = []
                in_size = 1
                
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(in_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    in_size = hidden_size
                
                layers.append(nn.Linear(in_size, 1))
                layers.append(nn.Sigmoid())  # Ensure [0,1] output
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Create and train network
        model = FlexibleNet(hidden_sizes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(2000):  # High number of epochs for exact fitting
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Save best model state
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Early stopping if we hit target or patience exceeded
            if current_loss < self.target_mse or patience_counter > patience:
                break
        
        # Load best model
        model.load_state_dict(best_state)
        model.eval()
        
        def predict_func(x_new):
            with torch.no_grad():
                x_tensor_new = torch.FloatTensor(x_new.reshape(-1, 1)).to(device)
                pred = model(x_tensor_new).cpu().numpy().flatten()
                return np.clip(pred, 0.0, 1.0)
        
        return predict_func, best_loss
    
    def _fit_spline(self, x, y, smoothing=True):
        """
        Fit B-spline with automatic knot selection.
        """
        try:
            from scipy.interpolate import UnivariateSpline, interp1d
        except ImportError:
            raise ImportError("scipy required for spline fitting")
            
        if smoothing:
            # Smoothing spline with automatic smoothing parameter
            spline = UnivariateSpline(x, y, s=None)  # s=None for automatic smoothing
            spline.set_smoothing_factor(1e-10)  # Very low smoothing for near-exact fit
        else:
            # Interpolating spline (exact at data points)
            spline = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        def predict_func(x_new):
            if smoothing:
                pred = spline(x_new)
            else:
                pred = spline(x_new)
            return np.clip(pred, 0.0, 1.0)
        
        # Compute fit quality
        y_pred = predict_func(x)
        mse = np.mean((y - y_pred)**2)
        
        return predict_func, mse
    
    def _fit_rbf_network(self, x, y):
        """
        Fit Radial Basis Function network.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("sklearn required for RBF network fitting")
            
        # Automatically select number of RBF centers
        n_centers = min(len(x) // 2, 20)
        
        # Use k-means to place RBF centers
        if n_centers > 1:
            kmeans = KMeans(n_clusters=n_centers, random_state=42)
            centers = kmeans.fit(x.reshape(-1, 1)).cluster_centers_.flatten()
        else:
            centers = np.array([np.mean(x)])
        
        # Compute RBF matrix
        sigma = np.std(x) / np.sqrt(n_centers)  # Automatic width selection
        rbf_matrix = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * sigma**2))
        
        # Add bias term
        rbf_matrix_bias = np.column_stack([rbf_matrix, np.ones(len(x))])
        
        # Solve for weights using least squares
        weights = np.linalg.lstsq(rbf_matrix_bias, y, rcond=None)[0]
        
        def predict_func(x_new):
            rbf_new = np.exp(-((x_new[:, None] - centers[None, :]) ** 2) / (2 * sigma**2))
            rbf_new_bias = np.column_stack([rbf_new, np.ones(len(x_new))])
            pred = rbf_new_bias @ weights
            return np.clip(pred, 0.0, 1.0)
        
        # Compute fit quality
        y_pred = predict_func(x)
        mse = np.mean((y - y_pred)**2)
        
        return predict_func, mse
    
    def _fit_single_method(self, x, y, method, segment_idx):
        """
        Fit a single specified method.
        """
        if method == 'gp':
            return self._fit_gaussian_process(x, y, kernel_type='rbf')
        elif method == 'neural':
            return self._fit_neural_network(x, y, architecture='deep')
        elif method == 'spline':
            return self._fit_spline(x, y, smoothing=False)
        elif method == 'rbf':
            return self._fit_rbf_network(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_for_size(self, n_large, enforce_normalization=True):
        """
        Predict marginal probabilities for larger problem using exact models.
        """
        if not self.fitted_models:
            raise ValueError("Must fit models first")
        
        self._print(f"Predicting with exact models for n={n_large}...")
        
        # Create normalized positions
        positions_norm = np.linspace(0, 1, n_large)
        
        # Predict each segment
        phi_pred = np.zeros((self.L, n_large))
        
        for ell in range(self.L):
            predict_func = self.fitted_models[ell]
            predictions = predict_func(positions_norm)
            phi_pred[ell, :] = predictions
        
        # Optional normalization
        if enforce_normalization:
            col_sums = np.sum(phi_pred, axis=0)
            col_sums = np.where(col_sums == 0, 1.0, col_sums)
            phi_pred = phi_pred / col_sums[None, :]
        
        # Report quality
        if enforce_normalization:
            max_norm_error = np.max(np.abs(np.sum(phi_pred, axis=0) - 1.0))
            self._print_summary(f"Max normalization error: {max_norm_error:.2e}")
            self._print(f"Normalization enforced. Max error: {max_norm_error:.2e}")
        
        min_val, max_val = np.min(phi_pred), np.max(phi_pred)
        self._print(f"Value bounds: [{min_val:.6f}, {max_val:.6f}]")
        
        return phi_pred
    
    def plot_fit_quality(self, n_small, phi_original):
        """
        Plot the quality of fits on the original data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
            
        positions_norm = np.linspace(0, 1, n_small)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for ell in range(min(self.L, 6)):  # Plot up to 6 segments
            if ell < len(axes):
                ax = axes[ell]
                
                # Original data
                original = phi_original[ell, :]
                predicted = self.fitted_models[ell](positions_norm)
                
                ax.plot(positions_norm, original, 'o-', label='Original', alpha=0.7)
                ax.plot(positions_norm, predicted, '--', label='Fitted', alpha=0.7)
                
                mse = self.fit_scores[ell]
                model_type = self.model_types[ell]
                
                ax.set_title(f'Segment {ell}: {model_type}\nMSE = {mse:.2e}')
                ax.set_xlabel('Normalized Position')
                ax.set_ylabel('Probability')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for ell in range(self.L, len(axes)):
            axes[ell].set_visible(False)
        
        plt.tight_layout()
        plt.show()

### --- Exact solution for the marginals --- ###

def unif_prior_to_η_ϕ_combinations(Lmin, Lmax, Δ, n, p_l: np.array = None):
    """
    Returns: Lxn matrix ϕ, where ϕ[l, t] is the marginal probability of observing 
    signal l at time t.

    Allow at most L signals i.e. 0, 1, 2, ... L signals.
    When L_tmp < L signals, assume signals present are sequential: 0, ..., L_tmp-1.
    
    p_l: length-L vector storing the prior probability over the number of signals.
    When left blank, assume uniform prior over all η configs, across l ∈ [Lmin, Lmax].
    Can be uniform or arbitrary as long as it sums to 1. [tested in test_chgpt_configs.py]

    η: tot_nconfigs x (L-1) matrix, where each row stores a config via 
       the starting index i of each signal l=1, ..., (L-1). 
       Observation index i and signal index l are 0-indexed.

    Considers all possible combinations of signals with number of signals l < Lmax, and computes the marginal probability of observing signal l at time t.
    """

    assert Lmin > 0 and Lmax >= Lmin
    num_L = Lmax - Lmin + 1
    if p_l is not None:
        assert len(p_l) == num_L, "p_l must store the prior probability over [Lmin, Lmax]"
        assert np.isclose(np.sum(p_l), 1), "p_l must be normalised to 1"

    assert Δ >= 1 and Δ * Lmax <= n, \
        "If Δ*Lmax > n, then some L∈[Lmin, Lmax] will " + \
        "have no valid configs satisfying minimum Δ intersignal distance" + \
        "making p_l meaningless as well."
    # Each config stores starting index i of each signal:
    η_arr = -np.ones((0, Lmax-1), dtype=int)
    # Stores number of configs when there exist l signals, for each l ∈ [Lmin, Lmax]:
    nconfigs_arr = np.zeros(num_L, dtype=int) # exactly l signals
    marg_nconfigs_arr = np.zeros((n, Lmax), dtype=int)
    if p_l is not None:
        # Marginal probability that each yi in i∈[n] is from signal l:
        ϕ = np.zeros((n, Lmax))
    for l in tqdm(range(Lmin, Lmax+1)): 

        nchgpts_l = l - 1
        η_l = configs_arr_for_exact_L(n, l, Δ) # nconfigs_l x nchgpts_l. Most computationally expensive, stays the same regardless of combinations. 
        nconfigs_l = η_l.shape[0]
        nconfigs_arr[l-Lmin] = nconfigs_l
        η_l_padded = -np.ones((nconfigs_l, Lmax-1), dtype=int) # -1s represent non-existent chgpts
        η_l_padded[:, :nchgpts_l] = η_l
        η_arr = np.concatenate((η_arr, η_l_padded), axis=0)

        marg_nconfigs_arr_l = nconfigs_arr_for_exact_L(η_l, n) # nxl, this is going to hold for each possible combination of signals
        if l <= 2: # No combinations need to be considered, back to the original code.
            marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
            if p_l is not None:
                ϕ[:, :l] += marg_nconfigs_arr_l / nconfigs_l * p_l[l-Lmin] # This is where the key update happens, and where the combo will be used.
                assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
                # assuming given l, chgpts are uniform:
                # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
            continue
        
        # Combinations code begins
        combs = list(itertools.combinations(range(1, Lmax), l-1))

        for j in range(len(combs)): 
            combs_idx = np.array([0, *combs[j]]) # Prepend the 0 because we always assume β_0 always takes the first spot
            marg_nconfigs_arr[:, combs_idx] += marg_nconfigs_arr_l
            # marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
        
        if p_l is not None:
            for j in range(len(combs)): 
                combs_idx = np.array([0, *combs[j]]) # Prepend the 0 because we always assume β_0 always takes the first spot
                ϕ[:, combs_idx] += marg_nconfigs_arr_l / nconfigs_l * (p_l[l-Lmin] / len(combs)) # This is where the key update happens, and where the combo will be used.
            assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
            # assuming given l, chgpts are uniform:
            # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
    
    tot_nconfigs = np.sum(nconfigs_arr)
    assert η_arr.shape[0] == tot_nconfigs

    # Marginal probability that each yi in i∈[n] is from signal l:
    if p_l is None: # assume uniform prior over all η configs.
        ϕ = marg_nconfigs_arr / tot_nconfigs # nxLmax
    assert np.allclose(np.sum(ϕ, axis=1), 1)

    # This section stays the same as it only updates p_η_arr, which is independent of combinations (only change-point location dependent). 
    if p_l is None:
        p_η_arr = np.ones(tot_nconfigs) / tot_nconfigs
    else:
        p_η_arr = np.zeros(tot_nconfigs)
        nconfigs_so_far = 0
        for l in range(Lmin, Lmax+1):
            nconfigs_l = nconfigs_arr[l - Lmin]
            idx_start = nconfigs_so_far
            idx_end = nconfigs_so_far + nconfigs_l
            p_η_arr[idx_start:idx_end] = p_l[l - Lmin] / nconfigs_l 
            nconfigs_so_far += nconfigs_l
    assert np.allclose(np.sum(p_η_arr), 1)
    # η: tot_nconfigs x (Lmax-1)
    # p_η: length-tot_nconfigs vector
    # ϕ.T: Lmax x n matrix
    return tot_nconfigs, η_arr, p_η_arr, ϕ.T 

def unif_prior_to_η_ϕ(Lmin, Lmax, Δ, n, p_l: np.array = None):
    """
    Returns: Lxn matrix ϕ, where ϕ[l, t] is the marginal probability of observing 
    signal l at time t.

    Allow at most L signals i.e. 0, 1, 2, ... L signals.
    When L_tmp < L signals, assume signals present are sequential: 0, ..., L_tmp-1.
    
    p_l: length-L vector storing the prior probability over the number of signals.
    When left blank, assume uniform prior over all η configs, across l ∈ [Lmin, Lmax].
    Can be uniform or arbitrary as long as it sums to 1. [tested in test_chgpt_configs.py]

    η: tot_nconfigs x (L-1) matrix, where each row stores a config via 
       the starting index i of each signal l=1, ..., (L-1). 
       Observation index i and signal index l are 0-indexed.
    """
    assert Lmin > 0 and Lmax >= Lmin
    num_L = Lmax - Lmin + 1
    if p_l is not None:
        assert len(p_l) == num_L, "p_l must store the prior probability over [Lmin, Lmax]"
        assert np.isclose(np.sum(p_l), 1), "p_l must be normalised to 1"

    print(f"\Delta: {Δ}, n: {n}, Lmin: {Lmin}, Lmax: {Lmax}")
    assert Δ >= 1 and Δ * Lmax <= n, \
        "If Δ*Lmax > n, then some L∈[Lmin, Lmax] will " + \
        "have no valid configs satisfying minimum Δ intersignal distance" + \
        "making p_l meaningless as well."
    # Each config stores starting index i of each signal:
    η_arr = -np.ones((0, Lmax-1), dtype=int)
    # Stores number of configs when there exist l signals, for each l ∈ [Lmin, Lmax]:
    nconfigs_arr = np.zeros(num_L, dtype=int) # exactly l signals
    marg_nconfigs_arr = np.zeros((n, Lmax), dtype=int)
    if p_l is not None:
        # Marginal probability that each yi in i∈[n] is from signal l:
        ϕ = np.zeros((n, Lmax))
    for l in tqdm(range(Lmin, Lmax+1)): 

        nchgpts_l = l - 1
        η_l = configs_arr_for_exact_L(n, l, Δ) # nconfigs_l x nchgpts_l. The slowest part of the code.
        nconfigs_l = η_l.shape[0]
        nconfigs_arr[l-Lmin] = nconfigs_l
        η_l_padded = -np.ones((nconfigs_l, Lmax-1), dtype=int) # -1s represent non-existent chgpts
        η_l_padded[:, :nchgpts_l] = η_l
        η_arr = np.concatenate((η_arr, η_l_padded), axis=0)

        marg_nconfigs_arr_l = nconfigs_arr_for_exact_L(η_l, n) # nxl, this is going to hold for each possible combination of signals
        marg_nconfigs_arr[:, :l] += marg_nconfigs_arr_l
        if p_l is not None:
            ϕ[:, :l] += marg_nconfigs_arr_l / nconfigs_l * p_l[l-Lmin] # This is where the key update happens, and where the combo will be used.
            assert np.allclose(np.sum(ϕ, axis=1), np.sum(p_l[:l-Lmin + 1]))
            # assuming given l, chgpts are uniform:
            # p_η_arr[idx_start:idx_end] = p_l[nchgpts_l] / tot_nconfigs_l 
    
    tot_nconfigs = np.sum(nconfigs_arr)
    assert η_arr.shape[0] == tot_nconfigs

    # Marginal probability that each yi in i∈[n] is from signal l:
    if p_l is None: # assume uniform prior over all η configs.
        ϕ = marg_nconfigs_arr / tot_nconfigs # nxLmax
    try: 
        assert np.allclose(np.sum(ϕ, axis=1), 1)
    except:
        print("Warning: ϕ does not sum to 1 across rows. Check p_l and Δ.")
        print(f"Sum across rows (should be 1): {np.sum(ϕ, axis=1)}")

    if p_l is None:
        p_η_arr = np.ones(tot_nconfigs) / tot_nconfigs
    else:
        p_η_arr = np.zeros(tot_nconfigs)
        nconfigs_so_far = 0
        for l in range(Lmin, Lmax+1):
            nconfigs_l = nconfigs_arr[l - Lmin]
            idx_start = nconfigs_so_far
            idx_end = nconfigs_so_far + nconfigs_l
            p_η_arr[idx_start:idx_end] = p_l[l - Lmin] / nconfigs_l 
            nconfigs_so_far += nconfigs_l
    assert np.allclose(np.sum(p_η_arr), 1)
    # η: tot_nconfigs x (Lmax-1)
    # p_η: length-tot_nconfigs vector
    # ϕ.T: Lmax x n matrix
    return tot_nconfigs, η_arr, p_η_arr, ϕ.T 

def configs_arr_for_exact_L(n:int, L:int, Δ:int=1):
    """
    Returns a nconfigs x (L-1) matrix configs_arr, where each row 
    stores a config via the starting index of each signal l.
    
    configs_arr[j, l] = starting index of signal l in config j.

    This function may cause memory issues for larger L (L>5).

    Tested on Jupyter notebook with (L = 4, L^* = 3), (L = 4, L^* = 2). 
    """
    assert L >= 1
    nchgpts = L - 1
    # tot_nconfigs = int(comb(n-1, nchgpts))

    # η_arr = np.zeros((tot_nconfigs, nchgpts), dtype=int)
    # for i, combo in tqdm(enumerate(itertools.combinations(range(1, n), nchgpts)), total=tot_nconfigs):
    #     η_arr[i] = combo
    range_start = np.maximum(1, Δ).astype(int)
    desired_range = range(np.maximum(1, Δ).astype(int),n - Δ)
    tot_nconfigs = int(comb(len(desired_range), nchgpts))
    η_arr = np.array(list(itertools.combinations(desired_range, nchgpts))
                           ).reshape(tot_nconfigs, nchgpts)

    # η_arr is nconfigs x nchgpts. Signal 0 always starts from index 0.
    if Δ > 1:
        # Discard violating configs:
        # if Δ > 1:
        #     # Discard violating configs:
        #     def is_ok_config(η):
        #         if np.all(η == -1):
        #             return True
        #         η_extended = np.concatenate(([0], η[η!=-1], [n-1]))
        #         # adjacent chgpts must be at least Δ apart, first/last chgpt 
        #         # must be at least Δ away from the endpoints:
        #         return np.all(np.diff(η_extended) > Δ)
        #     η_arr = η_arr[np.apply_along_axis(is_ok_config, axis=1, arr=η_arr)]

        # Vectorize the above for-loop. This is correct because η_arr will have no -1's, and so we can take the diff. 
        η_extended = jnp.block([jnp.zeros((η_arr.shape[0], 1)), η_arr , (n-1) * jnp.ones((η_arr.shape[0], 1))])
        η_ext_diff = jnp.diff(η_extended, axis=1)
        # mask = jnp.all(jnp.logical_or(η_ext_diff > Δ, (η_ext_diff == 0)), axis = 1)
        mask = jnp.all(η_ext_diff > Δ, axis = 1)
        # Discard violating configs according to mask: 
        η_arr = η_arr[mask]

    assert np.all(np.diff(η_arr, axis=1) > 0), "Each row is non-decreasing"
    return η_arr.astype(int)
        
def nconfigs_arr_for_exact_L(η_arr, n):
    """
    Returns nxL matrix, where each row stores the number of configs
    that have yi underlain by signal l, for each i ∈ [n] and l ∈ [L].

    η_arr is nconfigs x (L-1)
    """
    nconfigs = η_arr.shape[0]
    L = η_arr.shape[1] + 1
    nconfigs_arr = np.zeros((n, L), dtype=int)
    for i_config in tqdm(range(nconfigs)):
        η = η_arr[i_config, :]
        num_chgpts = np.sum(η != -1)
        if num_chgpts == 0:
            nconfigs_arr[:, 0] += 1
        else:
            η_extended = np.concatenate(([0], η[η!=-1], [n]))
            for l in range(num_chgpts+1):
                nconfigs_arr[η_extended[l]:η_extended[l+1], l] += 1
    assert np.all(np.sum(nconfigs_arr, axis=1) == nconfigs)
    return nconfigs_arr

def η_to_ψ(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.
    """
    # Compatible with the -1 entries:
    num_chgpts = np.sum(η != -1)
    ψ = np.ones(n, dtype=int) * num_chgpts
    # ψ = ψ.at[:η[0]].set(0)
    ψ[:η[0]] = 0
    for l in range(1, num_chgpts):
        # ψ = ψ.at[η[l-1]:η[l]].set(l)
        ψ[η[l-1]:η[l]] = l
    assert np.diff(ψ).min() >= 0, "ψ should be non-decreasing"
    return ψ

def η_to_ψ_jax(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.

    Involves many tricks in order to avoid jax errors. 
    """
    # Compatible with the -1 entries:
    # num_chgpts = jnp.where(η != -1, 1, 0).sum() # Checking for -1's causes jax errors
    # assert jnp.all(η >= -1), "η should be non-negative" # Causes jax errors. To do this check, would have to first iterate through η and check for -1's.
    num_chgpts = η.flatten().shape[0]
    ψ = jnp.ones(n, dtype=int) * num_chgpts
    # ψ = ψ.at[:η[0]].set(0)
    # ψ[:η[0]] = 0
    ψ = jnp.where(jnp.arange(n) < η[0], 0, ψ)
    cond_0 = jnp.where(jnp.arange(n) < η[0], 0, ψ)
    for l in range(1, num_chgpts):
        
        # A -1 in the change point configuration indicates end of change points
        if η[l] == -1:
            break

        cond_1 = jnp.where(jnp.arange(n) >= η[l-1], l, cond_0) # Tricks to avoid jax errors
        cond_2 = jnp.where(jnp.arange(n) < η[l], cond_1, l + 1)
        ψ = cond_2
        cond_0 = cond_2
    # assert jnp.diff(ψ).min() >= 0, "ψ should be non-decreasing" # Creates jax errors
    return ψ
η_to_ψ_jax_mapped = jax.vmap(η_to_ψ_jax, (0, None), 0)

def η_to_ψ_jax_combinations(η, n, comb): 
    """ Can only deal with >0 change points. Does not work if η has all -1's."""

    if len(comb) == 1:
        return jnp.where(jnp.arange(n) < η[0], 0, comb[0])

    ψ = jnp.ones(n, dtype=int)
    cond_0 = jnp.where(jnp.arange(n) < η[0], 0, ψ)

    for i in range(0, len(comb)-1):
        l = comb[i]
        l_next = comb[i+1]
        
        cond_1 = jnp.where(jnp.arange(n) >= η[i], l, cond_0) # Tricks to avoid jax errors
        cond_2 = jnp.where(jnp.arange(n) < η[i+1], cond_1, l_next)
        ψ = cond_2
        cond_0 = cond_2
    return ψ

def η_to_ψ_jax_combinations_mapped(η_arr, n, L):
    """ Returns an array of signal configurations ψ_arr, along with an array η_idx indicating the index of the η configuration that generated each ψ configuration (useful for gof computations and point estimation). 

    
    Tested on Jupyter notebook with (L = 4, L^* = 3), (L = 4, L^* = 2), (L = 4, L^* = 1).
    """

    num_chgpts_mask = jnp.array([jnp.where(η_arr != -1, 1, 0).sum(axis=1)]).T # To be tested
    min_num_chgpts = num_chgpts_mask.min()
    max_num_chgpts = num_chgpts_mask.max()
    assert max_num_chgpts == L - 1, "The maximum number of change points should be L - 1"
    assert max_num_chgpts == η_arr.shape[1], "The maximum number of change points should be equal to the number of columns in η_arr"

    L_star_min = min_num_chgpts + 1
    assert L_star_min >= 1

    η_to_ψ_jax_combinations_vmap = (jax.vmap(η_to_ψ_jax_combinations, (0, None, None), 0)) # The key step that speeds everything up
    if L_star_min == L: # Account for the special case of L_star = L
        combinations = list(itertools.combinations(range(1, L), L-1))
        comb = combinations[0]
        ψ_arr = η_to_ψ_jax_combinations_vmap(η_arr, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
        η_idx = jnp.arange(η_arr.shape[0])
        return ψ_arr, η_idx
    assert L_star_min == 1, "L_star_min should either be equal to 1 or equal to L in order for the code to work"


    combinations = []
    max_num_chgpts = η_arr.shape[1]
    for num_chgpts in range(0, max_num_chgpts):
        combinations.append(list(itertools.combinations(range(1, L), num_chgpts + 1)))

    for num_chgpts in range(0, max_num_chgpts+1): 

        # Deal with the special case of zero change points
        # if num_chgpts == 0:
        #     ψ_arr = jnp.zeros(n, dtype=int)
        #     η_idx = jnp.array([0])
        #     continue

        # Identify the configurations with the correct number of change points
        mask = jnp.where(num_chgpts_mask == num_chgpts, 1, 0).flatten()

        # Apply the mask to η_arr
        η_arr_masked = η_arr[mask.astype(bool)]

        # Apply the function to the configurations with the correct number of change points, and stack the results
        combs = combinations[num_chgpts-1]
        if num_chgpts == 0: # Special case of zero change points
            # combs = [(0, )]
            ψ_arr = jnp.multiply(jnp.ones((L, n), dtype=int), jnp.arange(0, L).reshape(L, 1))
            η_idx = jnp.repeat(jnp.where(mask)[0], L)
            continue

        for comb in combs:
            ψ = η_to_ψ_jax_combinations_vmap(η_arr_masked, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
            if 'ψ_arr' not in locals():
                ψ_arr = ψ
            else:
                ψ_arr = jnp.vstack((ψ_arr, ψ))

            if 'η_idx' not in locals():
                η_idx = jnp.where(mask)[0]
            else:
                η_idx = jnp.concatenate((η_idx, jnp.where(mask)[0]))

    return ψ_arr, η_idx


    # combinations = []
    # L_star_list = list(range(L_star_min, L))
    # for L_star in L_star_list:
    #     combinations.append(list(itertools.combinations(range(1, L), L_star)))

    # num_chgpts_mask = jnp.array([jnp.where(η_arr != -1, 1, 0).sum(axis=1)]).T # To be tested


    # for i in range(0, len(combinations)):
    #     L_star = L_star_list[i]
    #     num_chgpts = L_star - 1

    #     # Identify the configurations with the correct number of change points
    #     mask = jnp.where(num_chgpts_mask == num_chgpts, 1, 0).flatten()

    #     # Apply the mask to η_arr
    #     η_arr_masked = η_arr[mask.astype(bool)]

    #     # Apply the function to the configurations with the correct number of change points, and stack the results
    #     combs = combinations[i]
    #     if num_chgpts == 0: # Special case of zero change points
    #         ψ_arr = jnp.multiply(jnp.ones((L, n), dtype=int), jnp.arange(0, L).reshape(L, 1))
    #         η_idx = jnp.repeat(jnp.where(mask)[0], L)
    #         continue

    #     for comb in combs:
    #         ψ = η_to_ψ_jax_combinations_vmap(η_arr_masked, n, comb) # Could eventually replace this with a jax.lax.scan for memory efficiency
    #         if 'ψ_arr' not in locals():
    #             ψ_arr = ψ
    #         else:
    #             ψ_arr = jnp.vstack((ψ_arr, ψ))

    #         if 'η_idx' not in locals():
    #             η_idx = jnp.where(mask)[0]
    #         else:
    #             η_idx = jnp.concatenate((η_idx, jnp.where(mask)[0])) 

    # return ψ_arr, η_idx

def η_to_ψ(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.

    Does not involve jax, but can take η with -1s.
    """
    # Compatible with the -1 entries:
    num_chgpts = np.sum(η != -1)
    ψ = np.ones(n, dtype=int) * num_chgpts
    # ψ = ψ.at[:η[0]].set(0)
    ψ[:η[0]] = 0
    for l in range(1, num_chgpts):
        # ψ = ψ.at[η[l-1]:η[l]].set(l)
        ψ[η[l-1]:η[l]] = l
    assert np.diff(ψ).min() >= 0, "ψ should be non-decreasing"
    return ψ

def η_arr_to_C_s(η_arr, n):
    """
    Convert η_arr (num_configs x (L-1) matrix) into C_s 
    (num_configs x n matrix). Does not involve jax, but can take η with -1s.
    """
    num_configs = η_arr.shape[0]
    C_s = np.zeros((num_configs, n), dtype=int)
    for i_config in range(num_configs):
        η = η_arr[i_config, :]
        C_s[i_config, :] = η_to_ψ(η, n)
    return C_s

def ψ_to_η(ψ, L):
    """
    Input C: a length-n vector storing a possible chgpt config i.e. 
    the signal index underlying each of the n observations    

    Returns: indices of changepoints, where the first observation of a 
    new signal is counted as the location of the changepoint.
    """
    if jnp.all(ψ == ψ[0]):
        return jnp.array([-1 for ℓ in range(L - 1)])

    idx_before_chgpt = -2 * jnp.ones(L - 1)

    # @jax.jit
    def jit_subfn(ψ):
        diff_vec = jnp.abs(jnp.diff(ψ.reshape(1, -1), axis = 1))
        diff_vec = jnp.minimum(diff_vec, 1).squeeze() # if jumped from signal 0 to 2, still count as a chgpt
        return diff_vec
    
    jit_res = (jit_subfn(ψ) == 1)
    idx_before_chgpt_size = (jit_res).sum()
    for j in range(0, idx_before_chgpt_size): 
        idx_before_chgpt = idx_before_chgpt.at[j].set(jnp.where(jit_res)[0][j])
    # idx_before_chgpt = jnp.where(jit_subfn(ψ) == 1)[0] # idx of last observation from previous signal
    # idx_before_chgpt = jnp.array(jnp.nonzero(jit_subfn(ψ, L))[0])
    # idx_before_chgpt = 
    # Replace the above line with jax.lax.select to avoid jax errors
    # idx_before_chgpt = jnp.where(jnp.abs(diff_vec) == 1, jnp.arange(diff_vec.shape[0]), -1)
    # idx_before_chgpt = jax.lax.select(diff_vec == 1, jnp.arange(diff_vec.shape[0]), -1)
    

    assert len(idx_before_chgpt) > 0
    res = idx_before_chgpt + 1 

    # Pad the results with -1s so that it has length L - 1
    # res = jnp.concatenate((res, -1 * jnp.ones(L - 1 - len(res))), axis = 0)
    return res.astype(int)
