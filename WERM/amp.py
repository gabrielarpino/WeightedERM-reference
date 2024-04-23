import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
import numpy as np
import jax.scipy.optimize as jso

""" Non-separable AMP to simulate Weighted ERM. Takes in a sequence of loss functions ℓᵢ, assumes no penalty on the signal for now """

def f(w, c):
    return -w/c
f_mapped = jax.vmap(f, in_axes=(0, None))

def ḟ(w, c):
    return -1/c
ḟ_mapped = jax.vmap(ḟ, in_axes=(0, None))

def ḡ(u, v, b, ℓ):
    """ Proximal Operator of f"""

    def obj_(z, u, v, b): return ℓ(z, v) + (1/(2*b)) * jnp.linalg.norm(z - u)**2
    # Solve using jaxopt, so that we can differentiate through the solution
    gd = jaxopt.GradientDescent(fun=obj_, maxiter=500, implicit_diff=True)
    init_params = u
    return gd.run(init_params, args = (u, v, b)).params # Keep getting batch tracer is not callable. 

    # solve using scipy
    # return jso.minimize(obj_, u, args = (u, v, b, ℓ), method="BFGS").x
ḡ_mapped = jax.vmap(ḡ, in_axes=(0, 0, None, 0))

def g(u, v, b, ℓ):
    return (ḡ(u, v, b, ℓ) - u) / b
g_mapped = jax.vmap(g, in_axes=(0, 0, None, 0))

def ġ(u, v, b, ℓ):
    """ To be tested """
    ℓ_double_prime = jax.grad(jax.grad(ℓᵢ))
    return (b * ℓ_double_prime(u, v) + 1) ** -1
ġ_mapped = jax.vmap(ġ, in_axes=(0, 0, None, 0))

def dgdz(u, v, b, ℓ):
    """ To be tested """
    ℓ_prime = jax.grad(ℓᵢ)
    return b * ℓ_prime(u, v)

# Can already try running the regular AMP. 

def amp(X, y, n, p, ℓ_list, max_iter=100):
    """ AMP Algorithm """

    ŝ = jnp.zeros(n)
    b = jnp.ones(n) * 0.1 # initial value of b must be greater than zero
    β̂ = np.random.normal(size=p) # initial value of β must be greater than zero
    for _ in tqdm(range(max_iter)):

        # θ update steps
        θ = X @ β̂ - b * ŝ
        print("θ.shape: ", θ.shape)
        print("y.shape: ", y.shape)
        θ̂ = ḡ_mapped(θ.reshape(-1, 1), y, b, ℓ_list)
        ŝ = g_mapped(θ, y, b, ℓ_list)
        c = jnp.mean(ġ_mapped(θ, y, b, ℓ_list), axis=0)

        # β update steps
        β = X.T @ ŝ - c * β̂
        β̂ = f_mapped(β, c)
        b = p/n * jnp.mean(ḟ_mapped(β, c), axis=0)

    return β̂, θ̂