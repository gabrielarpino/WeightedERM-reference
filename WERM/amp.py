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

def obj_(z, args): 
    """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
    u, v, b, π_i = args
    return (π_i * ℓ_fixed(z, v) + (1/(2*b)) * jnp.linalg.norm(z - u)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer
def ḡ(u, v, b, π_i):
    """ Proximal Operator of f"""

    # Solve using jaxopt, so that we can differentiate through the solution

    # gd = jaxopt.GradientDescent(fun=obj_, maxiter=500, implicit_diff=True)
    # init_params = 0.5*u
    # args = (u, v, b, π_i)
    
    # test_grad = jax.grad(obj_)(1., (1.5, 1.5, 1.5, 1.5))
    # jax.debug.print("test_grad: {x}", x = test_grad)
    # return gd.run(init_params, args).params 
    # solve using scipy
    args = (u, v, b, π_i)
    res = jso.minimize(obj_, 0.5*jnp.array([u]), (args, ), method="BFGS")
    # jax.debug.print("res: {x}", x =res.status)
    # jax.debug.print("res success: {x}", x =res.success)
    return res.x[0] # was giving nans
ḡ_mapped = jax.vmap(ḡ, in_axes=(0, 0, None, 0))

def g(u, v, b, π_i):
    return (ḡ(u, v, b, π_i) - u) / b
g_mapped = jax.vmap(g, in_axes=(0, 0, None, 0))

def ġ(u, v, b, π_i):
    """ To be tested """
    ℓ_double_prime = jax.grad(jax.grad(ℓ_fixed))
    return (b * π_i * ℓ_double_prime(u, v) + 1) ** -1
ġ_mapped = jax.vmap(ġ, in_axes=(0, 0, None, 0))

def dgdz(u, v, b, ℓ):
    """ To be tested """
    ℓ_prime = jax.grad(ℓᵢ)
    return b * ℓ_prime(u, v)

# Can already try running the regular AMP. 

def amp(X, y, n, p, π, ℓ, max_iter=10):
    """ AMP Algorithm, π is the marginal probability of each signal being present, ℓ is the loss function (same for every i). 
    Works, but iterates blow up to nan eventually, possibly due to numerics. 
    """

    # Define the global loss ℓ_fixed once ℓ is passed in, as functions are not valid JAX types to vmap over
    global ℓ_fixed
    ℓ_fixed = ℓ

    ŝ = jnp.zeros(n)
    b = 0.5 # initial value of b must be greater than zero
    β̂ = 0.5*np.random.normal(size=p) # initial value of β must be greater than zero
    for _ in tqdm(range(max_iter)):
        # θ update steps
        θ = X @ β̂ - b * ŝ
        θ_test = ḡ(θ[0], y[0], b, π[0])
        print("θ_test args: ", θ[0], y[0], b, π[0])
        print("θ_test: ", θ_test)
        θ̂ = ḡ_mapped(θ, y, b, π).flatten()
        ŝ = g_mapped(θ, y, b, π).flatten()
        c = jnp.mean(ġ_mapped(θ, y, b, π), axis=0)

        # β update steps
        β = X.T @ ŝ - c * β̂
        β̂ = f_mapped(β, c)
        b = p/n * jnp.mean(ḟ_mapped(β, c), axis=0)

    return β̂, θ̂