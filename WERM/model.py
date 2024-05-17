import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
seed = 0

import numpy.random as nprandom

""" Non-separable AMP to simulate Weighted ERM. Takes in a sequence of loss functions ℓᵢ, assumes no penalty on the signal for now """

def ζ(z):
    return jnp.log(1 + jnp.exp(z))

def ζ_prime(z):
    return jnp.power(1 + jnp.exp(-z), -1 * jnp.ones(z.shape))

def h_logistic(Θ):
    ε = jax.random.uniform(jax.random.PRNGKey(seed), shape=Θ.shape)
    return jnp.greater(ζ_prime(Θ), ε).astype(int)

def h_linear(θ, ε):
    ε = jax.random.normal(jax.random.PRNGKey(seed), shape=θ.shape)
    return θ + ε

def LR_j_sample(Θ, c_j, η_j, L):
    """Logistic regression with a changepoint, output at index j. Sample output yj according to signal index cj. θ=XB.
    Tested."""
    if Θ.ndim == 1:
        Θ = Θ.reshape((1, L))
    num_samples = Θ.shape[0]
    # η = jnp.array(nprandom.normal(loc=0, scale=σ, size=(num_samples, 1))) # noise
    # Sample uniform noise from 0 to 1
    # η = jnp.array(nprandom.uniform(size=(num_samples, 1)))
    
    η = jnp.array(η_j.reshape((num_samples, 1)))
    Ψ = jnp.zeros((num_samples, L))
    Ψ = Ψ.at[:, c_j].set(1)

    # print("eval: ", ζ_prime(jnp.multiply(Θ, (Ψ == 1).astype(int)).sum(axis=1).reshape(
    #     (num_samples, 1))))
    # print("noise: ", η)
    Y = jnp.greater(ζ_prime(jnp.multiply(Θ, (Ψ == 1).astype(int)).sum(axis=1).reshape(
        (num_samples, 1))), η).astype(int)
    return Y

def LR(Θ, C):
    """ Logistic Regression with a Change Point. This takes in Θ ∈ R^{nxL} and C ∈ {0,L-1}^{nxL} and returns Y ∈ R^{nx1}.
    Needs to be differentiable to go into SE_fixed_C.  """

    L = Θ.shape[1]
    η = jnp.array(nprandom.uniform(size=(Θ.shape[0], 1)))
    return jax.vmap(LR_j_sample, in_axes=(0, 0, 0, None), \
                    out_axes=0)(Θ, C, η, L).reshape((Θ.shape[0], 1))

def h_i(Z_i, c_i, h, L):

    if Z_i.ndim == 1:
        Z_i = Z_i.reshape((1, -1))
    num_samples = Z_i.shape[0]
    Ψ = jnp.zeros((num_samples, L))
    Ψ = Ψ.at[:, c_i].set(1)

    return h(jnp.multiply(Z_i, (Ψ == 1).astype(int)))
h_i_mapped = jax.vmap(h_i, in_axes=(0, 0, 0, None, None), out_axes=0)

def h_CP(h, Z, C):
    L = Z.shape[1]
    return h_i_mapped(Z, C, h, L).reshape((Z.shape[0], 1))