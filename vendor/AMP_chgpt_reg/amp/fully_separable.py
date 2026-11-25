__all__ =[
    "psd_mat",
    "q_sep", 
    "define_f",
    "define_g",
    "f_full",
    "g_full",
    "run_GAMP"
]

import numpy as np # Does not require jax
from scipy.stats import multivariate_normal 
from scipy.special import logsumexp
import numpy.random as nprandom
from tqdm.auto import tqdm
from . import ϵ_0

def psd_mat(L):
    """ 
    Returns a random positive semi-definite matrix of size L-by-L.
    """
    A = nprandom.rand(L, L)
    res = A @ A.transpose()
    while not np.all(np.linalg.eigvals(res) >= 0):
        A = nprandom.rand(L, L)
        res = A @ A.transpose()
    return res
    
class q_sep_():
    """ The heterogeneous output function.
    Generalized q which takes in the C_s matrix, each row consisting of a vector indicating which signal influences data point i. 
    Non-Separable setting with L>=2 signals. 
    Samples a dataset with signal heterogeneity uniform among those dictated by the rows of C_s. 
    """

    def __init__(self, Θ, ϕ, σ):
        self.Θ = Θ
        self.n = self.Θ.shape[0]
        self.ϕ = ϕ
        self.σ = σ

    def sample(self):
        self.Ψ = np.random.binomial(1, 1 - self.ϕ, size=(self.n, 1)) # Stores signal index for each row of X
        η = np.random.normal(loc = 0, scale = self.σ, size=(self.n, 1))
        return q_sep(self.Θ, self.Ψ, η)

def q_sep(θ, Ψ, ε):
    """ 
    Separable mixtures setting with L>=2 signals
    q applies row-wise to matrix inputs 
    """
    (n, L) = θ.shape
    assert Ψ.shape == (n, 1) and ε.shape == (n, 1)
    assert np.all(np.logical_and(0 <= Ψ, Ψ <= L-1))
    Y = θ[np.arange(n), Ψ.T].T + ε
    assert Y.shape == (n, 1)
    return Y

def define_f(δ, L, ν̂_B, κ_B, ρ):
    """ 
    Defines the optimal separable f based on the SE parameters, 
    for zero mean signal and L>=2.
    
    ν̂_B is ν̂_B^{t+1}
    κ_B is κ_B^{t+1,t+1} in our overleaf notes.
    
    f_j is a function handle, which takes in a row vector s in R^L
    and outputs a row vector in R^L.
    """
    assert ν̂_B.shape == (L, L) and κ_B.shape == (L, L)
    tmp = ν̂_B.T @ ρ @ ν̂_B + 1/δ * κ_B
    χ = (ρ @ ν̂_B) @ np.linalg.pinv(tmp) # pinv is forsure correct
    # assert np.allclose(χ, χ_1)
    assert χ.shape == (L, L)
    def f_j(s):
        assert s.shape == (1, L), "s must be a row vector"
        return s @ χ.T
    return f_j

def f_full(B_, δ, L, ν̂, κ_B, ρ):
        """ Should be triple checked. """
        f_ = define_f(δ, L, ν̂, κ_B, ρ)
        return np.apply_along_axis(lambda s: f_(s.reshape((1, L))).reshape(L, ), 1, B_)

def form_Σ(idx, ϕ, L, σ, ρ, ν, κ_T):
    """
    Construct the (L+1)-by-(L+1) covariance matrix of
    [(V_{Θ,i}^t)^T, bar{Y_i] in R^{L+1} conditioned on 
    c_i = idx, where idx in [0,L-1]. Checked. 
    """
    assert idx in range(L)

    # Form Σᶻⱽ in the notes.
    Σ = np.zeros((2*L, 2*L))
    Σ[0:L, 0:L] = ρ
    Σ[0:L, L:2*L] = ν
    Σ[L:2*L, 0:L] = ν.T
    Σ[L:2*L, L:2*L] = ν.T @ np.linalg.inv(ρ) @ ν + κ_T

    Σᵢ = np.zeros((2*L + 1, 2*L + 1))
    Σᵢ[0:2*L, 0:2*L] = Σ
    Σᵢ[2*L, 0:2*L] = Σ[idx, :]
    Σᵢ[0:2*L, 2*L] = Σ[idx, :]
    Σᵢ[2*L, 2*L] = ρ[idx, idx] + σ**2
    # assert np.all(np.linalg.eigvals(Σᵢ) >= 0) # semi-positive definite, too strict a condition,
    return Σᵢ[L:, L:] # Return just the bottom right principal matrix

def form_Σ_λ(idx, V, u, ϕ, L, σ, ρ, ν, κ_T):
    """
    Calculate λ_idx = E[Z_{B,i} | V_{Θ,i}^t=V, \bar{Y}_i=u, c_i = idx] 
    for idx in [0,L-1].
        
    Input V is 1-by-L, u is scalar.
    Output λ is L-by-1. Checked. 
    """
    assert idx in range(L)
    assert V.shape == (1, L) and np.isscalar(u)
    Σᵢ = form_Σ(idx, ϕ, L, σ, ρ, ν, κ_T)
    term1 = np.concatenate((ν, ρ[:, idx].reshape(L, 1)), axis=1) 
    # term2 = np.linalg.solve(Σᵢ,np.append(V.T, u)) # Try to use pinv instead of solve for now, 
    term2 = np.linalg.pinv(Σᵢ) @ np.append(V.T, u) 
    λ = (term1 @ term2).reshape(L, 1)
    return Σᵢ, λ

def E_Z(V, u, ϕ, L, σ, ρ, ν, κ_T):
    """ 
    Calculate E[Z_{B,i} | V_{Theta,i}^t=V, bar{Y}_i=u].
    V is 1-by-L, u is scalar.
    Output is 1-by-L, and can be positive or negative.

    Tested.
    """
    assert V.shape == (1, L) and np.isscalar(u)
    # Might have to redefine log_N_1, log_N_2 later more explicity, for differentiability:
    def log_N(x, Σ):
        """
        Log of the pdf of N_{L+1}(0, Σ) evaluated at x in R^{L+1}.
        """
        assert x.shape == (L+1, 1)
        # assert np.all(np.linalg.eigvals(Σ + ϵ_0 * np.eye(L+1)) >= 0) # semi-positive definite. This is too strict. 
        # return np.log(multivariate_normal.pdf(x.flatten(), mean = np.zeros((L+1,)), cov = Σ + ϵ_0 * np.eye(L+1), allow_singular = True))
        try:
            mv = multivariate_normal.logpdf(x.flatten(), mean = np.zeros((L+1,)), cov = Σ + ϵ_0 * np.eye(L+1), allow_singular = True) # I don't think allow_singular is implemented here 
        except:
            print("Σ eigenvals: ", np.linalg.eigvals(Σ))
        mv = multivariate_normal.logpdf(x.flatten(), mean = np.zeros((L+1,)), cov = Σ + ϵ_0 * np.eye(L+1), allow_singular = True) # I don't think allow_singular is implemented here 
        return mv
        # else:
        #     mv = multivariate_normal.logpdf(x.flatten(), mean = np.zeros((L+1,)), cov = Σ + 0.1 * np.eye(L+1), allow_singular = True) # I don't think allow_singular is implemented here 
        #     return mv
        # return multivariate_normal.logpdf(x.flatten(), mean = np.zeros((L+1,)), cov = Σ + ϵ_0 * np.eye(L+1), allow_singular = True) # I don't think allow_singular is implemented here
    
    x = np.append(V.T, u).reshape(L+1, 1) # Tested
    
    # To do: form_Σ_λ rand log_N can be vectorized even further to return 
    # tensors storing the results for all L idx at once:
    Σ_1, λ_1 = form_Σ_λ(0, V, u, ϕ, L, σ, ρ, ν, κ_T) 
    Σ_2, λ_2 = form_Σ_λ(1, V, u, ϕ, L, σ, ρ, ν, κ_T)
    log_N1 = log_N(x, Σ_1)
    log_N2 = log_N(x, Σ_2)

    # Use logsumexp for numerical stability. Return sign information. Tested. 
    λ_p_mat = np.concatenate((λ_1 * ϕ, λ_2 * (1 - ϕ)), axis=1) # L-by-L 
    assert λ_p_mat.shape == (L, L)
    log_N_mat = np.tile([log_N1, log_N2], (L, 1))
    assert log_N_mat.shape == (L, L)
    log_num_arr, sign_arr = logsumexp(a = log_N_mat, axis=1, b = λ_p_mat, return_sign = True) 
    assert log_num_arr.shape == (L,) and sign_arr.shape == (L,)
    log_denom = logsumexp([log_N1, log_N2], b = [ϕ, 1 - ϕ])
    res = np.multiply(sign_arr, np.exp(log_num_arr - log_denom))
    return res # Why does this not require a transpose? 

def define_g(ϕ, L, σ, ρ, ν, κ_T): 
    """ 
    Defines g given the state evolution iterates ν, κ_T, ν̂, κ_B.
    ν_T is ν_Θ^t
    κ_T is κ_Θ^{t,t}
    ν̂_B is ν̂_B^{t+1}
    κ_B is κ_B^{t+1,t+1} in our overleaf notes.
    """
    
    def g(V, u, ϕ, L, σ, ρ, ν, κ_T):
        """ 
        Calculate gt applied to one row: R^L * R -> R^L.
        V is 1-by-L, u is scalar.
        Output is 1-by-L.
        """
        ξ = np.linalg.pinv(ν.T @ np.linalg.inv(ρ) @ ν + κ_T)
        return (E_Z(V, u, ϕ, L, σ, ρ, ν, κ_T) - V @ ξ.T @ ν.T ) @ np.linalg.pinv(ρ - ν.T @ ξ.T @ ν)

    return lambda V_, u_: g(V_, u_, ϕ, L, σ, ρ, ν, κ_T)
    
def g_full(Θ_, Y_, ϕ, L, σ, ρ, ν, κ_T):
    """ To be triple checked. """
    g_ = define_g(ϕ, L, σ, ρ, ν, κ_T)
    g_wrapper = lambda x: g_(x[0:L].reshape((1, L)), x[L]).reshape(L, )
    return np.apply_along_axis(g_wrapper, 1, np.concatenate((Θ_, Y_), axis=1))

def run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed=None, psi = None, eta = None, true_theta = None):
  if seed is not None:
      nprandom.seed(seed)

  n = int(δ * p)

  B̂ = B̂_0
  ν = np.zeros((L, L)) 
  κ_T = ρ
  ν̂ = np.zeros((L, L)) # Not necessary
  F = np.eye(L) # Checked
  R̂ = np.zeros((n, L)) # Checked

  for t in tqdm(range(T)):

      if verbose: 
        print("ν: ", ν)
        print("κ_T: ", κ_T)

      ## -- AMP -- ##
      Θ_t = X @ B̂ - R̂ @ F.T

      ## -- g and its parameters -- ##
      try:
        R̂ = g_full(Θ_t, Y, ϕ, L, σ, ρ, ν, κ_T)
      except:
        print('=== EARLY R̂ STOPPAGE ===')
        break

      if (np.isnan(R̂).any() or np.isinf(R̂).any()):
        print('=== EARLY R̂ STOPPAGE ===')
        break

      Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
      C = (np.linalg.pinv(Ω) @ (1/n * Θ_t.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T # Replacing this with autodiff might make it more numerically stable. 
      if verbose: print("C: ", C)
      B_t = X.T @ R̂ - B̂ @ C.T

      ## -- f and its parameters -- ##
    #   Z_B = nprandom.multivariate_normal(np.zeros(L), ρ, size=n)
    #   κ_T_symm = (κ_T + κ_T.T) / 2
    #   G_Θ = nprandom.multivariate_normal(np.zeros(L), κ_T_symm, size=n)
    #   V_Θ = Z_B @ np.linalg.inv(ρ) @ ν + G_Θ
    # #   V_Θ = true_theta @ np.linalg.inv(ρ) @ ν + G_Θ
    # #   R̂_2 = g_full(V_Θ, q_sep_(true_theta, ϕ, σ).sample(), ϕ, L, σ, ρ, ν, κ_T)
    #   R̂_2 = g_full(V_Θ, q_sep_(Z_B, ϕ, σ).sample(), ϕ, L, σ, ρ, ν, κ_T)
    #   ν̂ = 1/n * R̂_2.T @ R̂_2 # More accurate for large n and small p. 
    #   η = np.random.normal(0.0, σ, (n, 1)) # noise
    #   if psi is not None: 
    #     R̂_2 = g_full(V_Θ, q_sep(Z_B, psi, η), ϕ, L, σ, ρ, ν, κ_T)
    #     ν̂ = 1/n * R̂_2.T @ R̂_2 # More accurate for large n and small p. 
      ν̂ = 1/n * R̂.T @ R̂
      if verbose: print("ν̂: ", ν̂)
      κ_B = ν̂
      
      B̂ = f_full(B_t, δ, L, ν̂, κ_B, ρ)
      χ = (ρ @ ν̂) @ np.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) 
      F = 1/δ * χ.T 

      # Closed form gaussian case
      χ = (ρ @ ν̂) @ np.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) 
      ν = ρ @ ν̂ @ χ.T 
      κ_T = ν - ν.T @ np.linalg.inv(ρ) @ ν

    #   ν = 1/δ * 1/p * B̂.T @ B̂
    #   κ_T = ν - ν.T @ np.linalg.inv(ρ) @ ν  # Checked

  return B̂, ν, ν̂