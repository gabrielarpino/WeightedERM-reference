import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
import numpy as np
import jax.scipy.optimize as jso
import scipy.optimize
from WERM.config import η_to_ψ
from WERM.model import *

def prox_obj_no_idx(t, args): 
    """ Should always just involve scalars. """
    z, b, π_i = args
    # return (b * π_i * ζ(t) + (1/2) * jnp.linalg.norm(t - z)**2)
    return (b * π_i * ζ(t) + (1/2) * (t - z)**2) # The norm can cause interpretability problems, as this should always just take in scalars

def prox(z, b, π_i):
    """ Proximal operator with respect to the loss b π_i M """

    # def prox_obj(t): 
    #     """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
    #     return (b * π_i * M_fixed(t) + (1/2) * jnp.linalg.norm(t - z)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer

    # res = jso.minimize(prox_obj, jnp.array([1.0]), method="BFGS", options={"maxiter": 10000}) # Currently, this optimization is not always successful, even after increasing maxiter.
    # return res.x[0] # was giving nans

    # Solve using jaxopt, so that we can differentiate through the solution

    # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-5, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-7, stepsize = 1e-2, implicit_diff=True) # Extremely fast, and seems to give just as good results
    args = (z, b, π_i)
    res = gd.run(0.85, args).params
    return res
prox_mapped = jax.vmap(prox, in_axes=(0, None, 0))

def prox_prime(z, b, π_i):
    """ Tested, and yields the correct result in linear regression with change points. """
    ℓ_double_prime = jax.grad(jax.grad(ζ))
    prox_plug = prox(z, b, π_i)
    return jnp.power(b * π_i * ℓ_double_prime(prox_plug) + 1, -1)
prox_prime_mapped = jax.vmap(prox_prime, in_axes=(0, None, 0))

def ḡ_mapped(U, V, b, π):
    return prox_mapped(U.flatten() + b * jnp.multiply(π.flatten(), V.flatten()), b, π.flatten())

def logistic_fp(π, ρ, L, η_, δ, n):
    """ Solves the fixed point equations for MLE in the logistic model with change points."""

    num_samples = n

    # η = jnp.array([0, *((η_ * num_samples).astype(int)), num_samples])
    if η_.size == 0:
        ψ = jnp.zeros(num_samples).astype(int) 
    else:
        ψ = η_to_ψ((η_ * num_samples).astype(int), num_samples) # Seems to be correct

    G = np.random.randn(num_samples)
    Z = np.random.multivariate_normal(np.zeros(L), ρ, num_samples)
    ȳ = logistic_reg(Z, ψ).flatten()

    def v_Θ(μ̃, σ̃):
        return Z @ μ̃ + (σ̃ / (δ**0.5)) * G

    def obj_1(x):
        b, μ̃, σ̃ = x[0], x[1:-1], x[-1]

        term1 = (ȳ.flatten() - ζ_prime( prox_mapped(v_Θ(μ̃, σ̃) + b * jnp.multiply(π, ȳ), b, π) ).flatten() ).reshape(-1, 1)
        term2 = jnp.multiply(π.reshape(-1, 1), Z)
        
        inner = jnp.multiply( term2 , term1 )

        res = jnp.linalg.inv(ρ) @ jnp.mean(inner, axis = 0).reshape((L, ))
        return res

    def obj_2(x):
        b, μ̃, σ̃ = x[0], x[1:-1], x[-1]

        ζ_prime_prox = ζ_prime( prox_mapped(v_Θ(μ̃, σ̃) + b * jnp.multiply(π, ȳ), b, π) )
        inner = jnp.multiply( π**2, (ȳ.flatten() - ζ_prime_prox.flatten())**2 )
        return σ̃ - jnp.sqrt(δ**2 * b**2 * jnp.mean(inner))

    def obj_3(x):
        b, μ̃, σ̃ = x[0], x[1:-1], x[-1]
        ζ_double_prime_prox = jax.vmap(jax.grad(ζ_prime))( prox_mapped(v_Θ(μ̃, σ̃) + b * jnp.multiply(π, ȳ), b, π) ).flatten()
        inner = jnp.power( b * jnp.multiply(π , ζ_double_prime_prox) + 1, -1)
        return 1 - 1/δ - jnp.mean(inner)

    def fsolve_obj(x): 
        return jnp.array([*obj_1(x), obj_2(x), obj_3(x)])
    fsolve_obj_jit = jax.jit(fsolve_obj)

    # Solve using fsolve, faster than minimize with the jacobian
    J_fsolve = jax.jacrev(fsolve_obj_jit, argnums=0)
    x0 = jnp.array([0.66, *(0.85*jnp.ones(L)), 0.75])
    res, infodict, ier, mesg  = scipy.optimize.fsolve(fsolve_obj_jit, x0, fprime = J_fsolve, full_output = True, maxfev = 1000, xtol = 1.49012e-08)
    print("Optimization Results: x = ", res, ", ier = ", ier, ", mesg = ", mesg, ", infodict = ", infodict)
    b, μ̃, σ̃ = res[0], res[1:-1], res[-1]
    
    return μ̃, σ̃, b

