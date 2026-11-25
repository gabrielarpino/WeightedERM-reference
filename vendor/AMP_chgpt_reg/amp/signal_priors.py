# Don't use jnp here because we want it to be as general as possible. 
import itertools
import jax
import jax.numpy as jnp
import numpy.random as nprandom
import numpy as np

# Debugging nans
# from jax import config; config.update("jax_debug_nans", True)

class SignalPrior:
    """ The signal prior super class, for type indexing. """
    def __init__(self):
        self.L = 2
        self.cov = np.eye(self.L)
        self.ρ_B = 1.0 * self.cov

    def sample(self, num_rows):
        pass

class SparseDiffSignal_old(SignalPrior):
    """Sparse difference prior. For now only produces L=2 signals."""
    def __init__(self, δ, ρ_1, σ_w, α):
        # Two length-p signals.
        # Signal 1 ~ N(0, δρ_1).
        # In signal 2, each entry β_(2,j) = β_(1,j) with prob. 1-α, 
        # and = β_(1,j)+w with prob. α where w ~ N(0, σ_w^2).
        self.δ = δ
        self.ρ_1 = ρ_1
        self.σ_w = σ_w
        self.α = α
        self.η = np.sqrt((δ * ρ_1) / (δ * ρ_1 + σ_w**2))
        assert self.η < 1, "η should be less than 1"
        # β_{1,j}, β_{2,j} ~iid (1-α)N(0,δρ_B(same)) + αN(0, δρ_B(diff))
        self.ρ_B_same = ρ_1 * np.ones((2, 2))
        self.ρ_B_diff = ρ_1 * np.array([[1, self.η], [self.η, 1]])
        self.ρ_B = (1-α) * self.ρ_B_same + α * self.ρ_B_diff
        self.cov = self.δ * self.ρ_B
        self.L = 2
        assert np.allclose(self.ρ_B, ρ_1 * \
                           np.array([[1, 1-α*(1-self.η)], \
                                      [1-α*(1-self.η), 1]])), "ρ_B not as expected"
    
    def sample(self, num_rows):
        β_1 = nprandom.normal(loc=0, scale=np.sqrt(self.δ * self.ρ_1), 
                              size=(num_rows, 1))
        
        β_2 = β_1.copy()
        # Add noise to β_2:
        bern = nprandom.binomial(1, self.α, size=(num_rows, 1))
        noise = nprandom.normal(loc=0, scale=self.σ_w, size=(num_rows, 1))
        β_2 = β_1 + bern * noise
        # rescale to same variance:
        β_2[bern == 1] = β_2[bern == 1] * self.η
        # Stack β_1 and β_2:
        B = np.hstack((β_1, β_2))
        return B

class SparseDiffSignal(SignalPrior):
    """
    Sparse difference prior. Generalised version of SparseDiffSignal.
    Can produce L>=2 signals.

    Takes in var_1 instead of δ and ρ_1.
    """
    def __init__(self, var_1, σ_w, α, L):
        # For every pair of neighbouring length-p signals.
        # Signal 1 ~ N(0, δρ_1).
        # In signal 2, each entry β_(2,j) = β_(1,j) with prob. 1-α, 
        # and = β_(1,j)+w with prob. α where w ~ N(0, σ_w^2).

        self.var_1 = var_1
        self.σ_w = σ_w
        self.α = α
        self.L = L
        self.η = np.sqrt(var_1 / (var_1 + σ_w**2))
        assert self.η < 1, "η should be less than 1"
        # β_{1,j}, β_{2,j} ~iid (1-α)N(0,δρ_B(same)) + αN(0, δρ_B(diff))

        # Create 2^{L-1} binary arrays each of length (L-1), indicating whether 
        # each transition is a change or not.
        self.is_change_seq = jnp.array(list(itertools.product([0, 1], repeat=self.L-1)))
        assert self.is_change_seq.shape == (2**(self.L-1), self.L-1)
        num_changes_seq = np.sum(self.is_change_seq, axis=1)
        self.prob_seq = α ** num_changes_seq * \
            (1-α) ** (self.L-1- num_changes_seq)
        self.cov_seq = self.cov_cases()
        self.cov = np.einsum('i,ijk->jk', self.prob_seq, self.cov_seq)
        assert self.cov.shape == (self.L, self.L)
        assert np.allclose(self.cov, self.cov.T), "cov should be symmetric"
    
    def cov_cases(self):
        """
        Create 2^{L-1} LxL covariances for the 2^{L-1} different cases
        considering each transition between signals as either change or no-change."""
        
        def cov_one_case(is_change):
            """is_change is a length-(L-1) binary array indicating whether
            each transition is a change or not."""
            cov = jnp.eye(self.L)
            for i_row in range(self.L-1): # fill in each row in the upper triangular part
                for i_col in range(i_row+1, self.L):
                    cov = cov.at[i_row, i_col].set(
                        cov[i_row, i_col-1] * self.η ** is_change[i_col-1])
                   
                    # Copy upper triangular part to the lower triangular part:
                    cov = cov.at[i_col, i_row].set(cov[i_row, i_col])
            return cov * self.var_1
        cov_seq = jax.vmap(cov_one_case, 0, 0)(self.is_change_seq)
        assert cov_seq.shape == (2**(self.L-1), self.L, self.L)
        return cov_seq

    def sample(self, num_rows):
        β_1 = nprandom.normal(loc=0, scale=np.sqrt(self.var_1), 
                              size=(num_rows, 1))
        B = np.zeros((num_rows, self.L))
        B[:, 0] = β_1.reshape(-1)
        for l in range(1, self.L):
            β_2 = β_1.copy()
            # Add noise to β_2:
            bern = nprandom.binomial(1, self.α, size=(num_rows, 1))
            noise = nprandom.normal(loc=0, scale=self.σ_w, size=(num_rows, 1))
            β_2 = β_1 + bern * noise
            # rescale to same variance:
            β_2[bern == 1] = β_2[bern == 1] * self.η
            B[:, l] = β_2.reshape(-1)
            β_1 = β_2.copy()
        return B
    

class SparseGaussianSignal(SignalPrior):
    def __init__(self, α, δ, σ_l_arr) -> None:
        """β^{(l)} ~iid (1-α)δ_0 + αN(0, σ_l^2) 
        We require α>0 because otherwise the signal is always zero.
        δ = n/p is the oversampling ratio.
        σ_l_arr stores the std dev, not variance, of L signals.
        Then σ_l^2 = δ/α * ρ_l."""
        assert α > 0 and α <= 1, "α must be in (0,1]"
        assert δ > 0, "δ must be positive"
        assert np.all(σ_l_arr > 0), "σ_l must be positive" 
        self.α = α
        self.σ_l_arr = σ_l_arr
        self.L = len(σ_l_arr)
        self.ρ_B = α/δ * np.diag(σ_l_arr**2) # zero off-diagonal
        self.cov = δ * self.ρ_B
        # ρ_B[l,l] is the variance of the l-th signal.

    def sample(self, num_rows):
        """Returns signal matrix B of dim num_rows x L."""
        bern = nprandom.binomial(1, self.α, size=(num_rows, self.L))
        B = np.zeros((num_rows, self.L))
        for l in range(self.L):
            B[bern[:, l] == 1, l] = np.random.randn(np.sum(bern[:, l])) * self.σ_l_arr[l]
        return B

class GaussianSignal(SignalPrior):
    def __init__(self, B_cov):
        self.cov = B_cov
        self.L = self.cov.shape[0]

    def set_cov(self, B_cov):
        self.cov = B_cov
        
    def sample(self, num_rows):
        if self.cov is None: 
            print("Warning: no covariance matrix specified. Sampling from N(0, I_2).")
            self.cov = np.eye(2)
        return nprandom.multivariate_normal(mean=np.zeros(self.cov.shape[0]), 
                                            cov=self.cov, size=num_rows)